#!/usr/bin/env python3
"""Export a spatio-temporal scene graph per video: objects with an (x, y, t) track, relations with time.

    SG_OUT=$PWD/exp/v3final python tools/export_stsg.py --data-root $DATA [--videos ...]
writes $SG_OUT/<vid>/stsg.json:
  {"video", "fps", "duration_s", "width", "height",
   "objects":   [{"id", "class", "description" (DAM, from dossiers.json), "track": [[t_s, x, y, x0, y0, x1, y1], ...]}],   x, y = box centre,
                 all coordinates normalised to 0-1 (x left->right, y top->bottom); one row per visible frame
   "relations": [{"subject", "predicate", "object", "intervals_s": [[t0, t1], ...], "votes"}]}
Depth is not available, so there is no z. Tracks come from the SAM3 masks the relations were found on.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcore as S

FPS = 5


def export(vid, data_root):
    tubes = S.Tubes(S.vdir(vid, 'masks.txt', make=False))
    names = S.class_names(vid, data_root)
    H, W = tubes.hw
    R = json.load(open(S.vdir(vid, 'relations_ref.json', make=False)))['relations']
    used = sorted({r['subject'] for r in R} | {r['object'] for r in R})
    T = len(S.frames_of(data_root, vid))
    dpath = S.vdir(vid, 'dossiers.json', make=False)
    doss = json.load(open(dpath)) if os.path.exists(dpath) else {}
    objs = []
    for t in used:
        tr = []
        for f in sorted(tubes.frames_of.get(t, {})):
            x, y, w, h = tubes.frames_of[t][f][1]
            tr.append([round(f / FPS, 2), round((x + w / 2) / W, 4), round((y + h / 2) / H, 4),
                       round(x / W, 4), round(y / H, 4), round((x + w) / W, 4), round((y + h) / H, 4)])
        c = tubes.cls.get(t, -1)
        objs.append({'id': t, 'class': names[c] if 0 <= c < len(names) else 'object',
                     'description': doss.get(str(t), {}).get('description', ''), 'track': tr})
    rels = [{'subject': r['subject'], 'predicate': r['predicate'], 'object': r['object'],
             'intervals_s': [[a / FPS, (b + 1) / FPS] for a, b in r['intervals']], 'votes': r.get('votes', 1)}
            for r in R]
    out = {'video': vid, 'fps': FPS, 'duration_s': T / FPS, 'width': W, 'height': H,
           'objects': objs, 'relations': rels}
    json.dump(out, open(S.vdir(vid, 'stsg.json'), 'w'))
    return len(objs), len(rels), sum(len(o['track']) for o in objs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='*')
    a = ap.parse_args()
    vids = a.videos or sorted(os.listdir(os.path.join(a.data_root, 'images')))
    for v in vids:
        if not os.path.exists(S.vdir(v, 'relations_ref.json', make=False)):
            print(v, 'no relations_ref.json'); continue
        print(v, '%d objects, %d relations, %d (x, y, t) points' % export(v, a.data_root))
