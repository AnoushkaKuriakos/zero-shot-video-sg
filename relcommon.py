"""relcommon.py — relation-stage helpers layered over common.py."""
import json
import os

import numpy as np

import common as C


def get_predicate_list(anno):
    """The closed predicate vocabulary, from sailvos3d.json. Tries the known
    key layouts, else collects predicate names from per-video triplets."""
    for key in ('relations', 'predicates', 'relation_classes', 'predicate_classes'):
        v = anno.get(key)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return list(v)
    seen = []
    for entry in C._iter_video_entries(anno):
        for r in entry.get('relations', []):
            p = _triplet_fields(r)[2] if not isinstance(r, dict) else \
                (r.get('predicate') or r.get('relation'))
            if isinstance(p, str) and p not in seen:
                seen.append(p)
    if not seen:
        raise KeyError('No predicate list found in sailvos3d.json; keys: '
                       + str(list(anno.keys())))
    return seen


def _triplet_fields(r):
    """Normalize one GT relation entry -> (s_oid, o_oid, pred, intervals)."""
    if isinstance(r, dict):
        s = r.get('subject', r.get('s', r.get('subject_id')))
        o = r.get('object', r.get('o', r.get('object_id')))
        p = r.get('predicate', r.get('relation', r.get('p')))
        iv = r.get('intervals', r.get('durations', r.get('time', [])))
    else:
        s, p, o = r[0], r[1], r[2]   # SAIL-VOS: [subject, predicate, object, ...]
        iv = r[3] if len(r) > 3 else []
    if iv and isinstance(iv[0], (int, float)):
        iv = [iv]
    iv = [[int(a), int(b)] for a, b in iv] if iv else []
    return s, o, p, iv


def get_video_gt_relations(anno, vid, predicates):
    """-> list of (s_oid, o_oid, pred_idx, [[t1,t2],...]) with predicate
    resolved to an index into `predicates` (names or ints both accepted)."""
    entry = C.get_video_entry(anno, vid)
    out = []
    if not entry:
        return out
    for r in entry.get('relations', []):
        s, o, p, iv = _triplet_fields(r)
        if isinstance(p, str):
            if p not in predicates:
                continue
            p = predicates.index(p)
        out.append((int(s), int(o), int(p), iv))
    return out


def interval_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union > 0 else 0.0


def intervals_overlap_score(ivs_a, ivs_b):
    """Best temporal IoU between any pair of intervals from two lists."""
    best = 0.0
    for a in ivs_a:
        for b in ivs_b:
            best = max(best, interval_iou(a, b))
    return best


def parse_json_reply(text):
    """Pull the first JSON array/object out of an LLM reply (tolerates
    ```json fences and prose around it). Returns python obj or None."""
    t = text.strip()
    if '```' in t:
        for chunk in t.split('```'):
            chunk = chunk.strip()
            if chunk.startswith('json'):
                chunk = chunk[4:].strip()
            if chunk[:1] in '[{':
                t = chunk
                break
    start = min([i for i in (t.find('['), t.find('{')) if i >= 0], default=-1)
    if start < 0:
        return None
    for end in range(len(t), start, -1):
        try:
            return json.loads(t[start:end])
        except json.JSONDecodeError:
            continue
    return None


def centroid_of(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())
