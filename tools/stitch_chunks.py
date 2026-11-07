#!/usr/bin/env python3
"""
Join SAM3 runs on overlapping time chunks (run_sam3.py --frame-range) into one masks.txt.
A track continues into the next chunk when its mean mask IoU over the shared frames is
>= --iou, best pairs first. In shared frames the earlier chunk's masks are kept.

    python tools/stitch_chunks.py --out DIR/<vid> DIR/c0/<vid>/masks.txt DIR/c1/<vid>/masks.txt ...
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np
from pycocotools import mask as M


def load(path):
    rows = [l.rstrip('\n').split(' ', 5) for l in open(path) if l.strip()]
    return [r for r in rows if len(r) == 6 and r[0].lstrip('-').isdigit()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('chunks', nargs='+')
    args = ap.parse_args()
    merged, next_id, joined = [], 1, 0
    prev = {}                                   # frame -> {final id: rle}, last chunk only
    for path in args.chunks:
        rows = load(path)
        cur = defaultdict(dict)
        for r in rows:
            cur[int(r[0])][r[1]] = {'size': [int(r[3]), int(r[4])], 'counts': r[5].encode()}
        shared = sorted(set(cur) & set(prev))
        score = defaultdict(list)
        for f in shared:
            a, b = list(prev[f]), list(cur[f])
            if a and b:
                ious = M.iou([cur[f][x] for x in b], [prev[f][x] for x in a], [0] * len(a))
                for i, x in enumerate(b):
                    for j, y in enumerate(a):
                        score[(x, y)].append(ious[i, j])
        pairs = sorted(((sum(v) / len(shared), k) for k, v in score.items()), reverse=True)
        idmap, used = {}, set()
        for s, (x, y) in pairs:
            if s >= args.iou and x not in idmap and y not in used:
                idmap[x] = y; used.add(y); joined += 1
        for r in rows:
            if r[1] not in idmap:
                idmap[r[1]] = next_id; next_id += 1
        last = max(prev, default=-1)
        prev = defaultdict(dict)
        for r in rows:
            f, tid = int(r[0]), idmap[r[1]]
            prev[f][tid] = {'size': [int(r[3]), int(r[4])], 'counts': r[5].encode()}
            if f > last:
                merged.append([r[0], str(tid)] + r[2:])
    os.makedirs(args.out, exist_ok=True)
    merged.sort(key=lambda r: (int(r[0]), int(r[1])))
    with open(os.path.join(args.out, 'masks.txt'), 'w') as fh:
        fh.writelines(' '.join(r) + '\n' for r in merged)
    print(f'{len(args.chunks)} chunks, {joined} tracks joined, {next_id - 1} tracks, {len(merged)} rows')


if __name__ == '__main__':
    main()
