#!/usr/bin/env python3
"""
Hybrid graph = caption pipeline triplets + direct-VLM triplets, fused together.
Concatenates <A>/<vid>/triplets_aligned_ref.jsonl and <B>/<vid>/triplets_aligned_ref.jsonl into
$SG_OUT/<vid>/triplets_aligned_ref.jsonl. B's window ids are offset by 100000 so that fuse_relations.py
counts one vote per (method, window): a relation both methods assert in a window gets 2 votes.
This rule was fixed before scoring (2026-11-04) and is not tuned. Copies masks/vocab/windows from A.
Then run fuse_relations.py (--plausible for SAIL-VOS) and the evaluator on $SG_OUT.

    SG_OUT=$PWD/exp/hybrid python tools/make_hybrid.py --a exp/v3final --b exp/vlm_direct --videos ...
"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True, help='caption-pipeline experiment dir')
ap.add_argument('--b', required=True, help='direct-VLM experiment dir')
ap.add_argument('--videos', nargs='+', required=True)
args = ap.parse_args()
out = os.environ['SG_OUT']
for v in args.videos:
    os.makedirs(os.path.join(out, v), exist_ok=True)
    for f in ['masks.txt', 'vocab.json', 'windows.json', 'dossiers.json', 'tube2gt.json']:
        src, dst = os.path.join(args.a, v, f), os.path.join(out, v, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    n = [0, 0]
    with open(os.path.join(out, v, 'triplets_aligned_ref.jsonl'), 'w') as fo:
        for i, d in enumerate([args.a, args.b]):
            for line in open(os.path.join(d, v, 'triplets_aligned_ref.jsonl')):
                r = json.loads(line)
                if i:
                    r['window_id'] = 100000 + r['window_id']
                    r['sources'] = ['vlm_direct']
                fo.write(json.dumps(r) + '\n'); n[i] += 1
    print(f'[{v}] hybrid: {n[0]} pipeline + {n[1]} direct window triplets')
