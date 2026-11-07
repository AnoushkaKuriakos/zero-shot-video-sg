#!/usr/bin/env python3
"""
Construction clean-up after SAM3 (README to-do 0b). SAM3 runs one prompt per class, so one
machine often gets several tubes under different, wrong classes. Two CPU-only steps, run on
a COPY of the experiment folder ($SG_OUT), after the graph exists:

1. Relabel from DAM. The first sentence of a tube's DAM description is matched against a
   word table per class; the class named EARLIEST in that sentence wins ("A large vehicle,
   likely a bulldozer" -> bulldozer). Only thing tubes are relabelled, and only to another
   thing class. Stuff tubes (ground, sky, dirt pile) keep their label: DAM describes the
   salient machine inside a large stuff region (151341 "sky" tubes are sky, DAM says
   excavator). A description that names no listed class ("likely a drill") keeps the label.
2. Merge duplicates. Two tubes are the same object when they are both visible in >= 50% of
   the frames of the shorter tube (and at least 10 frames) and their mean mask IoU over
   those frames is >= 0.5. Only thing tubes in the graph (dossiers.json) are considered. Best
   pairs first; the tube with the larger mean mask area (the one covering more of the
   object) keeps its id and (relabelled) class; masks are
   unioned per frame. Relation ids are rewritten, self-relations dropped and duplicate
   (s, p, o) rows joined (intervals unioned, votes and windows summed).

Rules come from class names, DAM wording and mask overlap; they were fixed before looking at
the activity labels. Edits masks.txt, dossiers.json, windows.json and relations_ref.json in
$SG_OUT/<vid>/ and writes $SG_OUT/relabel_log.json.

    cp -r exp/construction exp/construction_relabel
    SG_OUT=$PWD/exp/construction_relabel python tools/relabel_dam.py --videos 150625 151341 140555

Any domain (2026-11-07, now the last step of the pipeline): --data-root reads the thing and stuff
classes from sailvos3d.json; a thing class without an entry in WORDS is matched by its own name
(underscores as spaces, optional plural). --plausible re-applies the commonsense filter to the
relabelled relations (classes may have changed).
"""
import argparse, json, os, re, sys
from collections import defaultdict
import numpy as np
from pycocotools import mask as M

WORDS = {   # construction word table; multi-word names first within a class
    'excavator': r'\b(?:excavators?|diggers?|backhoes?)\b',
    'dump_truck': r'\b(?:dump trucks?|haul trucks?|trucks?|lorry|lorries)\b',
    'wheel_loader': r'\b(?:wheel loaders?|front loaders?|loaders?)\b',
    'bulldozer': r'\b(?:bulldozers?|dozers?)\b',
    'person': r'\b(?:person|people|workers?|man|men|woman|women)\b',
    'car': r'\b(?:cars?|suvs?|sedans?|pickups?|vans?)\b',
}
STUFF = {'ground', 'sky', 'soil_pile'}
MIN_COVIS, MIN_IOU, MIN_SHARED = 0.5, 0.5, 10
UNION = False
MERGE = False      # relabel only by default; merging is irreversible (a wrong merge deletes relations)      # union of two detections of one object is larger than the object


def configure(data_root):
    """Word table and stuff classes for this data set's class list."""
    global STUFF
    a = json.load(open(os.path.join(os.path.expanduser(data_root), 'sailvos3d.json')))
    things, stuff = a['objects']['thing'], a['objects']['stuff']
    STUFF = set(stuff)
    for c in list(WORDS):
        if c not in things:
            del WORDS[c]
    for c in things:
        if c not in WORDS:
            WORDS[c] = r'\b' + re.escape(c.replace('_', ' ')) + r'(?:s|es)?\b'


def first_sentence(text):
    return re.split(r'(?<=[.!?])\s', (text or '').strip())[0].lower()


def dam_class(text):
    """Class named earliest in the first sentence; in a compound noun the last word is the head, so when one
    class word directly precedes another ("car door", "truck driver") the second one counts."""
    s = first_sentence(text)
    hits = sorted((m.start(), m.end(), c) for c, pat in WORDS.items() for m in [re.search(pat, s)] if m)
    for a0, a1, ca in hits:
        nxt = [c for b0, _, c in hits if b0 == a1 + 1 and c != ca]
        return nxt[0] if nxt else ca
    return None


def load_masks(path):
    rows = [l.rstrip('\n').split(' ', 5) for l in open(path) if l.strip()]
    return [r for r in rows if len(r) == 6 and r[0].lstrip('-').isdigit()]


def rle(r):
    return {'size': [int(r[3]), int(r[4])], 'counts': r[5].encode()}


def union_intervals(iv):
    out = []
    for a, b in sorted(map(tuple, iv)):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def process(vdir, vid, log, plaus=None):
    doss = json.load(open(os.path.join(vdir, 'dossiers.json')))
    vocab = json.load(open(os.path.join(vdir, 'vocab.json')))
    rows = load_masks(os.path.join(vdir, 'masks.txt'))
    # class name <-> class id from the masks/dossiers themselves
    cid_of = {d['class_name']: int(d['class_id']) for d in doss.values() if 'class_id' in d}
    classes = vocab.get('classes') or []
    def cid(name):
        if name in cid_of:
            return cid_of[name]
        return classes.index(name)

    # 1. relabel
    change = {}
    for k, d in doss.items():
        old = d.get('class_name')
        if old in STUFF:
            continue
        new = dam_class(d.get('description'))
        if new and new != old and new not in STUFF:
            change[int(k)] = new
            log.append({'video': vid, 'step': 'relabel', 'tube': int(k), 'from': old, 'to': new,
                        'text': first_sentence(d.get('description'))[:120]})
            d['class_name'], d['class_id'] = new, cid(new)

    # 2. duplicates (only with --merge)
    per = defaultdict(dict)                         # tid -> frame -> row
    for r in rows:
        per[int(r[1])][int(r[0])] = r
    tids = sorted(per)
    area = {t: float(np.mean([M.area(rle(r)) for r in per[t].values()])) for t in tids}
    # thing tubes only: stuff regions nest (a dirt pile is part of the ground), so their
    # overlap does not mean they are the same object
    graph_t = [t for t in tids if str(t) in doss and doss[str(t)].get('class_name') not in STUFF]
    cand = []
    for i, a in enumerate(graph_t if MERGE else []):
        for b in graph_t[i + 1:]:
            if doss[str(a)].get('class_name') != doss[str(b)].get('class_name'):
                continue                            # only tubes of the same (relabelled) class are merged
            shared = sorted(set(per[a]) & set(per[b]))
            if len(shared) < max(MIN_SHARED, MIN_COVIS * min(len(per[a]), len(per[b]))):
                continue
            ious = [M.iou([rle(per[a][f])], [rle(per[b][f])], [0])[0, 0] for f in shared]
            m = float(np.mean(ious)) if ious else 0.0
            if m >= MIN_IOU:
                cand.append((m, a, b, len(shared)))
    parent = {}
    def root(t):
        while t in parent:
            t = parent[t]
        return t
    for m, a, b, n in sorted(cand, reverse=True):
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        keep, drop = (ra, rb) if area[ra] >= area[rb] else (rb, ra)
        parent[drop] = keep
        for f, r in per[drop].items():
            if f in per[keep] and UNION:           # union masks into the kept tube
                k = per[keep][f]
                u = M.merge([rle(k), rle(r)], intersect=False)
                per[keep][f] = k[:5] + [u['counts'].decode() if isinstance(u['counts'], bytes) else u['counts']]
            elif f not in per[keep]:               # otherwise the kept tube keeps its own mask; the dropped
                per[keep][f] = [r[0], str(keep)] + r[2:]   # tube only fills frames where the kept one is missing
        del per[drop]
        log.append({'video': vid, 'step': 'merge', 'keep': keep, 'drop': drop, 'iou': round(m, 3),
                    'shared_frames': n,
                    'keep_class': doss.get(str(keep), {}).get('class_name'),
                    'drop_class': doss.get(str(drop), {}).get('class_name')})
    idmap = {t: root(t) for t in tids}

    # write masks (class column from the possibly relabelled dossier)
    out = []
    for t in sorted(per):
        c = doss.get(str(t), {}).get('class_id')
        for f in sorted(per[t]):
            r = list(per[t][f]); r[1] = str(t)
            if c is not None:
                r[2] = str(c)
            out.append(' '.join(r) + '\n')
    out.sort(key=lambda l: (int(l.split(' ', 2)[0]), int(l.split(' ', 2)[1])))
    open(os.path.join(vdir, 'masks.txt'), 'w').writelines(out)

    for t, r in idmap.items():
        if t != r and str(t) in doss:
            doss[str(r)].setdefault('merged_from', []).append(
                {'tube': t, 'class_name': doss[str(t)].get('class_name'),
                 'description': doss[str(t)].get('description')})
            del doss[str(t)]
    json.dump(doss, open(os.path.join(vdir, 'dossiers.json'), 'w'), indent=1)

    wp = os.path.join(vdir, 'windows.json')
    wins = json.load(open(wp))
    for w in wins:
        objs = list(dict.fromkeys(idmap.get(int(o), int(o)) for o in w['objects']))
        w['objects'] = objs
        w['classes'] = {str(o): doss[str(o)]['class_id'] for o in objs if str(o) in doss}
        if 'obj_ref' in w:
            w['obj_ref'] = {str(idmap.get(int(o), int(o))): f for o, f in w['obj_ref'].items()}
    json.dump(wins, open(wp, 'w'))

    rp = os.path.join(vdir, 'relations_ref.json')
    if os.path.exists(rp):
        R = json.load(open(rp))
        joined = {}
        n_self = 0
        for rel in R['relations']:
            s, o = idmap.get(rel['subject'], rel['subject']), idmap.get(rel['object'], rel['object'])
            if s == o:
                n_self += 1
                continue
            key = (s, rel['predicate'], o)
            if key in joined:
                j = joined[key]
                j['intervals'] = union_intervals(j['intervals'] + rel['intervals'])
                j['votes'] += rel.get('votes', 0)
                j['windows'] = sorted(set(j['windows']) | set(rel.get('windows', [])))
            else:
                j = dict(rel, subject=s, object=o)
                j['intervals'] = union_intervals(rel['intervals'])
                joined[key] = j
        for j in joined.values():
            j['n_frames'] = sum(b - a + 1 for a, b in j['intervals'])
        if plaus:                                   # classes may have changed: re-check the predicate rules
            name = lambda t: doss.get(str(t), {}).get('class_name')
            keep = {k: j for k, j in joined.items() if plaus(name(k[0]), k[1], name(k[2])) is None}   # None = accepted
            log.append({'video': vid, 'step': 'plausible', 'dropped': len(joined) - len(keep)})
            joined = keep
        log.append({'video': vid, 'step': 'relations', 'before': len(R['relations']),
                    'after': len(joined), 'self_dropped': n_self})
        R['relations'] = sorted(joined.values(), key=lambda j: -j.get('votes', 0))
        json.dump(R, open(rp, 'w'), indent=1)
    return change, {t: r for t, r in idmap.items() if t != r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--data-root', default='', help='read thing/stuff classes from <root>/sailvos3d.json')
    ap.add_argument('--plausible', action='store_true', help='re-apply the commonsense filter afterwards')
    ap.add_argument('--merge', action='store_true', help='also merge duplicate tubes (off by default)')
    args = ap.parse_args()
    global MERGE
    MERGE = args.merge
    if args.data_root:
        configure(args.data_root)
    plaus = None
    if args.plausible:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from plausible import plausible as plaus
    out = os.environ.get('SG_OUT')
    assert out and os.path.isdir(out), 'set SG_OUT to the experiment COPY to edit'
    log = []
    for vid in args.videos:
        vdir = os.path.join(out, vid)
        if not os.path.exists(os.path.join(vdir, 'dossiers.json')):
            print(vid, 'skipped (no dossiers.json)'); continue
        ch, mg = process(vdir, vid, log, plaus)
        print(vid, 'relabel', ch, 'merge', mg, flush=True)
    json.dump(log, open(os.path.join(out, 'relabel_log.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
