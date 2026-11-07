#!/usr/bin/env python3
"""
Construction demo metrics from a finished scene graph (exp/construction), checked against
the Earthmoving Equipment activity labels (Roberts & Golparvar-Fard 2019).

Rules from relations to activities (written from predicate meaning, not fitted):
  excavator state per frame, first match wins:
    dump   excavator  dumping into | pouring onto | loading   dump_truck / soil_pile
    dig    excavator  digging in                               ground / soil_pile
    swing  excavator  swinging toward                          anything
    move   excavator  moving on | driving on                   anything
    idle   none of the above
  truck "being loaded" per frame: any excavator -> truck relation in
    {loading, dumping into, pouring onto}, or truck 'waiting for' excavator
GT (Labels/<vid>/*.txt, '<start> <end> <code>' on source frames; doc.txt legend
0 idle, 1 swing bucket, 2 load bucket, 3 dump, 4 move). For trucks we read code 2 as
"being loaded" (the truck side of a load); other codes = not being loaded.

Outputs $SG_OUT/construction_metrics.json and a text report: per-video timeline shares,
frame-level agreement with GT (with the majority-class baseline), loading episodes, and the
relation counts by predicate.

    SG_OUT=$PWD/exp/construction python tools/construction_metrics.py --data-root data/construction
"""
import argparse
import json
import os
from collections import Counter

EXC_RULES = [('dump', {'dumping into', 'pouring onto', 'loading'}, {'dump_truck', 'soil_pile'}),
             ('dig', {'digging in'}, {'ground', 'soil_pile'}),
             ('swing', {'swinging toward'}, None),
             ('move', {'moving on', 'driving on'}, None)]
LOAD_P = {'loading', 'dumping into', 'pouring onto'}
GT_EXC = {'0': 'idle', '1': 'swing', '2': 'dig', '3': 'dump', '4': 'move'}


def tube_classes(vdir):
    d = json.load(open(os.path.join(vdir, 'dossiers.json')))
    return {int(k): v.get('class_name') for k, v in d.items()}


def active(rel, f):
    return any(a <= f <= b for a, b in rel['intervals'])


def gt_series(path, fmap):
    lab = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 3 and p[0].lstrip('-').isdigit():
            lab[(int(p[0]), int(p[1]))] = p[2]
    out = []
    for sf in fmap:
        code = next((c for (a, b), c in lab.items() if a <= sf <= b), None)
        out.append(code)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='*')
    a = ap.parse_args()
    root, out = os.path.expanduser(a.data_root), os.environ['SG_OUT']
    vids = a.videos or json.load(open(os.path.join(root, 'sailvos3d.json')))['split']['val']
    report, lines = {}, []
    tot = Counter()
    for vid in vids:
        vdir = os.path.join(out, vid)
        rp = os.path.join(vdir, 'relations_ref.json')
        if not os.path.exists(rp):
            lines.append(f'{vid}: no graph'); continue
        rels = json.load(open(rp))['relations']
        cls = tube_classes(vdir)
        for r in rels:
            r['s_cls'], r['o_cls'] = cls.get(r['subject']), cls.get(r['object'])
        fmap = json.load(open(os.path.join(root, 'activity', vid, 'frame_map.json')))['source_frame']
        n = len(fmap)
        exc = [r for r in rels if r['s_cls'] == 'excavator']
        state = []
        for f in range(n):
            s = 'idle'
            for name, preds, objs in EXC_RULES:
                if any(r['predicate'] in preds and (objs is None or r['o_cls'] in objs) and active(r, f)
                       for r in exc):
                    s = name; break
            state.append(s)
        loaded = [any(((r['s_cls'] == 'excavator' and r['o_cls'] == 'dump_truck' and r['predicate'] in LOAD_P)
                       or (r['s_cls'] == 'dump_truck' and r['o_cls'] == 'excavator' and r['predicate'] == 'waiting for'))
                      and active(r, f) for r in rels) for f in range(n)]
        # GT
        gexc = [GT_EXC.get(c) for c in gt_series(os.path.join(root, 'activity', vid, 'excavator.txt'), fmap)]
        trucks = sorted(p for p in os.listdir(os.path.join(root, 'activity', vid)) if p.startswith('truck'))
        gload = [False] * n
        for t in trucks:
            for f, c in enumerate(gt_series(os.path.join(root, 'activity', vid, t), fmap)):
                gload[f] = gload[f] or c == '2'
        # excavator working (any non-idle) vs GT working
        work_p = [s != 'idle' for s in state]
        work_g = [g not in (None, 'idle') for g in gexc]
        acc = lambda p, g: sum(x == y for x, y in zip(p, g)) / n
        maj = lambda g: max(Counter(g).values()) / n
        ep = sum(1 for f in range(n) if loaded[f] and (f == 0 or not loaded[f - 1]))
        tp = sum(p and g for p, g in zip(loaded, gload)); fp = sum(p and not g for p, g in zip(loaded, gload))
        fn = sum(g and not p for p, g in zip(loaded, gload))
        r = {'frames': n, 'seconds': n / 5, 'relations': len(rels),
             'predicates': dict(Counter(x['predicate'] for x in rels).most_common()),
             'tubes_by_class': dict(Counter(cls.values())),
             'pred_excavator_share': {k: round(v / n, 3) for k, v in Counter(state).items()},
             'gt_excavator_share': {k: round(v / n, 3) for k, v in Counter(gexc).items()},
             'excavator_state_acc': round(acc(state, gexc), 3), 'excavator_state_majority': round(maj(gexc), 3),
             'working_acc': round(acc(work_p, work_g), 3), 'working_majority': round(maj(work_g), 3),
             'truck_loading_pred_share': round(sum(loaded) / n, 3), 'truck_loading_gt_share': round(sum(gload) / n, 3),
             'truck_loading_precision': round(tp / (tp + fp), 3) if tp + fp else None,
             'truck_loading_recall': round(tp / (tp + fn), 3) if tp + fn else None,
             'loading_episodes_pred': ep}
        report[vid] = r
        for k in ('frames', 'relations'):
            tot[k] += r[k]
        lines.append(f'{vid}: {json.dumps(r)}')
    json.dump(report, open(os.path.join(out, 'construction_metrics.json'), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
