#!/usr/bin/env python3
"""
Step 8 — Recall@K / mean Recall@K of both paths against SAIL-VOS GT.

  tube -> GT object   majority best-IoU over every --stride-th frame
                      (IoU >= --iou). Saved to OUT/<vid>/tube2gt.json (the
                      comparison video uses it to colour correct edges).
  hit                 mapped subject, mapped object and predicate match a GT
                      triplet, and temporal IoU (union of intervals) >= --t-iou
  ranking             by votes (then frames); each GT is matched at most once.
                      SAIL-VOS stores one GT entry per interval ("A in B" over
                      0-49 and 103-217 = two entries), so one predicted
                      relation may satisfy every entry of its triplet that it
                      overlaps (PVSG-style: a triplet owns several intervals).
                      --one-to-one restores the old rule (one GT per
                      prediction), which made the second entry unreachable
  R@K                 pooled over all GT triplets of all videos
  mR@K                recall per predicate, pooled over videos, then averaged
                      (the old code averaged per-video mR values)
  ceiling             % of GT triplets whose subject AND object got a tube —
                      the most any captioner can reach with these masks

    python eval_relations.py --data-root $DATA --videos fam_1_mcs_2
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
from pycocotools import mask as maskUtils

import sgcore as S

KS = (20, 50, 100)


def _load_gt(anno, vid, predicates):
    """GT triplets as (s_oid, o_oid, predicate_id, [[t1,t2],...]).
    sailvos3d.json stores them as [subject, predicate, object, [t1, t2]]; the
    local relcommon.py reads exactly that order."""
    raw = S.R().get_video_gt_relations(anno, vid, predicates)
    return [(int(s), int(o), int(p), norm_ivs(iv)) for s, o, p, iv in raw]


def tube_to_gt(vid, data_root, stride, iou_thr):
    tubes = S.Tubes(S.vdir(vid, 'masks.txt'))
    files = S.C().gt_files_of(os.path.expanduser(data_root), vid)
    votes = defaultdict(Counter)
    for fi in range(0, len(files), stride):
        items = tubes.per_frame.get(fi, [])
        if not items:
            continue
        gt = np.load(files[fi])
        gt = gt[..., 0] if gt.ndim > 2 else gt
        gms = {int(g): gt == g for g in np.unique(gt) if g != 0}
        for tid, _, rle in items:
            pm = maskUtils.decode(rle).astype(bool)
            if pm.shape != gt.shape:              # FIX: was a crash
                pm = cv2.resize(pm.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
            best, bg = 0.0, None
            for g, gm in gms.items():
                inter = np.logical_and(pm, gm).sum()
                u = pm.sum() + gm.sum() - inter
                v = inter / u if u else 0.0
                if v > best:
                    best, bg = v, g
            if bg is not None and best >= iou_thr:
                votes[tid][bg] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def norm_ivs(ivs):
    if not ivs:
        return []
    if isinstance(ivs[0], (int, float, np.integer)):
        return [[int(ivs[0]), int(ivs[1])]]
    return [[int(a), int(b)] for a, b in ivs]


def tiou(a, b):
    fa = {f for x, y in a for f in range(x, y + 1)}
    fb = {f for x, y in b for f in range(x, y + 1)}
    return len(fa & fb) / len(fa | fb) if fa | fb else 0.0


def evaluate(rels, gt, t2g, t_iou, one_to_one=False):
    """-> {gt_index: rank} for matched GT triplets."""
    hit, used = {}, set()
    for rank, r in enumerate(rels, 1):
        gs, go = t2g.get(r['subject']), t2g.get(r['object'])
        if gs is None or go is None:
            continue
        for gi, (s, o, pid, ivs) in enumerate(gt):
            if gi in used or (gs, go, r['predicate_id']) != (s, o, pid):
                continue
            if ivs and t_iou > 0 and tiou(r['intervals'], ivs) < t_iou:
                continue
            used.add(gi); hit[gi] = rank
            if one_to_one:
                break
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--t-iou', type=float, default=0.1)
    ap.add_argument('--one-to-one', action='store_true',
                    help='a prediction can match only one GT entry (old rule)')
    args = ap.parse_args()
    anno = S.C().load_anno(os.path.expanduser(args.data_root))
    predicates = S.predicate_list(args.data_root)

    pooled = {p: {'n': 0, 'hits': {K: 0 for K in KS},
                  'per_pred': defaultdict(lambda: [0, 0]), 'preds': 0} for p in S.PATHS}
    rows, total_gt, reach = [], 0, 0
    for vid in args.videos:
        gt = _load_gt(anno, vid, predicates)
        if not gt:
            print(f'[{vid}] no GT relations — skipped'); continue
        t2g = tube_to_gt(vid, args.data_root, args.stride, args.iou)
        json.dump({str(k): v for k, v in t2g.items()}, open(S.vdir(vid, 'tube2gt.json'), 'w'))
        mapped = set(t2g.values())
        r_ok = sum(1 for s, o, _, _ in gt if s in mapped and o in mapped)
        total_gt += len(gt); reach += r_ok
        print(f'[{vid}] {len(gt)} GT triplets; {len(t2g)} tubes mapped to GT objects; '
              f'ceiling {r_ok}/{len(gt)} ({r_ok / len(gt):.0%}) have both objects segmented')
        for path in S.PATHS:
            rp = S.vdir(vid, f'relations_{path}.json')
            if not os.path.exists(rp):
                continue
            rels = json.load(open(rp))['relations']
            hit = evaluate(rels, gt, t2g, args.t_iou, args.one_to_one)
            P = pooled[path]
            P['n'] += len(gt); P['preds'] += len(rels)
            for K in KS:
                P['hits'][K] += sum(1 for r in hit.values() if r <= K)
            for gi, (_, _, pid, _) in enumerate(gt):
                P['per_pred'][pid][1] += 1
                P['per_pred'][pid][0] += int(gi in hit and hit[gi] <= max(KS))
            row = {'video': vid, 'path': path, 'gt': len(gt), 'pred': len(rels),
                   **{f'R@{K}': round(sum(1 for r in hit.values() if r <= K) / len(gt), 4)
                      for K in KS}}
            rows.append(row)
            json.dump({'hits': {str(k): v for k, v in hit.items()},
                       'gt': [[s, o, predicates[p], iv] for s, o, p, iv in gt]},
                      open(S.vdir(vid, f'eval_{path}.json'), 'w'), indent=1)

    print(f'\n{"path":<26}{"#pred":>7}' + ''.join(f'{"R@" + str(K):>8}' for K in KS)
          + f'{"mR@" + str(max(KS)):>9}')
    for path in S.PATHS:
        P = pooled[path]
        if not P['n']:
            continue
        mr = np.mean([h / t for h, t in P['per_pred'].values()])
        print(f'{S.PATH_NAME[path]:<26}{P["preds"]:>7}'
              + ''.join(f'{P["hits"][K] / P["n"]:>8.3f}' for K in KS) + f'{mr:>9.3f}')
    if total_gt:
        print(f'{"(segmentation ceiling)":<26}{"":>7}{reach / total_gt:>8.3f}')
        print('\nper predicate (hits@100 / GT):')
        preds_seen = sorted({p for P in pooled.values() for p in P['per_pred']})
        for pid in preds_seen:
            cells = '  '.join(f'{path}:{pooled[path]["per_pred"][pid][0]}/'
                              f'{pooled[path]["per_pred"][pid][1]}'
                              for path in S.PATHS if pooled[path]['n'])
            print(f'  {predicates[pid]:<22}{cells}')
    if rows:
        cp = os.path.join(S.OUT, 'eval_summary.csv')
        with open(cp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f'\nper-video table -> {cp}')


if __name__ == '__main__':
    main()
