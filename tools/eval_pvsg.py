#!/usr/bin/env python3
"""
R@K / mR@K under the PVSG / PSG4D protocol (OpenPVSG tools/rel_test_full.py +
utils/rel_metrics.py), so our numbers sit next to PSG4D's Table 2 (PSG4D-GTA val =
our 13 videos). Differences from eval_relations.py:

  triplet match   subject CLASS, object CLASS and predicate equal the GT triplet's
                  (eval_relations.py maps tubes to GT instances by majority vote)
  vIoU            over the frames of the GT intervals where both GT objects are
                  visible: a frame is hit when it lies in a predicted interval and the
                  predicted subject and object masks both have IoU >= 0.5 with the GT
                  masks of that frame; vIoU = hit frames / GT frames
  hit             vIoU >= 0.5 (first such prediction in rank order; each GT once)
  soft recall     the hit's vIoU instead of 1 (PSG4D's "soft recall")
  weak            vIoU >= 0.1
  mR@K            per-predicate recall averaged over predicates with GT

    SG_OUT=$PWD/out python tools/eval_pvsg.py --data-root $DATA --videos ...
Writes $SG_OUT/eval_pvsg.txt and $SG_OUT/eval_pvsg.csv.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as maskUtils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_relations as E
import sgcore as S

KS = (20, 50, 100)


def frames_of_ivs(ivs, n):
    return {f for a, b in ivs for f in range(max(a, 0), min(b, n - 1) + 1)}


def video_hits(vid, anno, predicates, data_root, path):
    """-> list of (predicate_id, rank or None, vIoU, weak rank or None) per GT triplet."""
    gt = E._load_gt(anno, vid, predicates)
    rp = S.vdir(vid, f'relations_{path}.json')
    if not gt or not os.path.exists(rp):
        return [], 0
    import json
    rels = json.load(open(rp))['relations']
    v = next(x for x in anno['data'] if x['video_id'] == vid)
    gcat = {o['object_id']: o['category'] for o in v['object']}
    names = S.class_names(vid, data_root)
    tubes = S.Tubes(S.vdir(vid, 'masks.txt'))
    tcat = {t: names[c] if 0 <= c < len(names) else None for t, c in tubes.cls.items()}
    files = S.C().gt_files_of(os.path.expanduser(data_root), vid)
    n = len(files)

    # candidate (GT, prediction) pairs and the (frame, tube, gt object) IoUs they need
    cands = defaultdict(list)            # gi -> [(rank, rel)]
    need = defaultdict(set)              # frame -> {(tid, gid)}
    for gi, (s, o, pid, ivs) in enumerate(gt):
        fs = frames_of_ivs(ivs, n)
        for rank, r in enumerate(rels):
            if r['predicate_id'] != pid or gcat.get(s) is None or gcat.get(o) is None:
                continue
            if tcat.get(r['subject']) != gcat[s] or tcat.get(r['object']) != gcat[o]:
                continue
            cands[gi].append((rank, r))
            pf = frames_of_ivs(r['intervals'], n)
            for f in fs & pf:
                need[f] |= {(r['subject'], s), (r['object'], o)}
    gt_frames = defaultdict(set)         # gi -> frames where both GT objects are visible
    vis = defaultdict(set)               # gid -> visible frames (only within needed spans)
    span_frames = set()
    for s, o, _, ivs in gt:
        span_frames |= frames_of_ivs(ivs, n)
    gids = {x for s, o, _, _ in gt for x in (s, o)}
    iou = {}
    for f in sorted(span_frames):
        g = np.load(files[f]); g = g[..., 0] if g.ndim > 2 else g
        present = set(np.unique(g).tolist()) & gids
        for gid in present:
            vis[gid].add(f)
        if f not in need:
            continue
        dec = {}
        for tid, gid in need[f]:
            if gid not in present:
                iou[(f, tid, gid)] = 0.0; continue
            if tid not in dec:
                rle = next((r for t, _, r in tubes.per_frame.get(f, []) if t == tid), None)
                m = maskUtils.decode(rle).astype(bool) if rle is not None else None
                if m is not None and m.shape != g.shape:
                    m = cv2.resize(m.astype(np.uint8), (g.shape[1], g.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) > 0
                dec[tid] = m
            m = dec[tid]
            if m is None:
                iou[(f, tid, gid)] = 0.0; continue
            gm = g == gid
            inter = np.logical_and(m, gm).sum()
            iou[(f, tid, gid)] = inter / (m.sum() + gm.sum() - inter)
    out = []
    for gi, (s, o, pid, ivs) in enumerate(gt):
        real = frames_of_ivs(ivs, n) & vis[s] & vis[o]
        hit_rank, hit_v, weak_rank = None, 0.0, None
        for rank, r in cands[gi]:
            if not real:
                break
            pf = frames_of_ivs(r['intervals'], n)
            h = sum(1 for f in real & pf
                    if iou.get((f, r['subject'], s), 0) >= 0.5 and iou.get((f, r['object'], o), 0) >= 0.5)
            vi = h / len(real)
            if vi >= 0.1 and weak_rank is None:
                weak_rank = rank
            if vi >= 0.5:
                hit_rank, hit_v = rank, vi
                break
        out.append((pid, hit_rank, hit_v, weak_rank))
    return out, len(rels)


def summarize(per_gt):
    res = {}
    for K in KS:
        tot, hit, soft, weak = defaultdict(int), defaultdict(int), defaultdict(float), defaultdict(int)
        for pid, rk, vi, wk in per_gt:
            tot[pid] += 1
            if rk is not None and rk < K:
                hit[pid] += 1; soft[pid] += vi
            if wk is not None and wk < K:
                weak[pid] += 1
        n = sum(tot.values()) or 1
        preds = [p for p in tot if tot[p]]
        res[K] = dict(R=sum(hit.values()) / n, mR=np.mean([hit[p] / tot[p] for p in preds]) if preds else 0,
                      softR=sum(soft.values()) / n,
                      softmR=np.mean([soft[p] / tot[p] for p in preds]) if preds else 0,
                      weakR=sum(weak.values()) / n)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--paths', nargs='+', default=list(S.PATHS))
    args = ap.parse_args()
    anno = S.C().load_anno(os.path.expanduser(args.data_root))
    predicates = S.predicate_list(args.data_root)
    lines, rows = [], []
    for path in args.paths:
        pooled, nv = [], 0
        n_pred = n_hit_pred = n_at_gt = 0
        for vid in args.videos:
            h, npred = video_hits(vid, anno, predicates, args.data_root, path)
            if not h:
                continue
            nv += 1; pooled += h
            # measures that output volume cannot inflate: a prediction counts once, and
            # R@GT gets as many guesses as the video has GT triplets
            n_pred += npred
            n_hit_pred += len({rk for _, rk, _, _ in h if rk is not None})
            n_at_gt += sum(1 for _, rk, _, _ in h if rk is not None and rk < len(h))
            r = summarize(h)[100]
            rows.append([vid, path, len(h), npred, f"{r['R']:.4f}", f"{r['softR']:.4f}", f"{r['weakR']:.4f}"])
            print(f'[{vid}] {path}: R@100 {r["R"]:.3f} soft {r["softR"]:.3f} weak {r["weakR"]:.3f}', flush=True)
        if not pooled:
            continue
        res = summarize(pooled)
        lines.append(f'{S.PATH_NAME[path]}  ({nv} videos, {len(pooled)} GT triplets)')
        for K in KS:
            x = res[K]
            lines.append(f'  @{K:<3d} R {100*x["R"]:6.2f}  mR {100*x["mR"]:6.2f}  | soft R {100*x["softR"]:6.2f}'
                         f'  soft mR {100*x["softmR"]:6.2f}  | weak R {100*x["weakR"]:6.2f}')
        lines.append(f'  volume: {n_pred} predictions for {len(pooled)} GT ({n_pred / len(pooled):.1f}x) | '
                     f'precision {100 * n_hit_pred / max(n_pred, 1):.2f} | R@GT {100 * n_at_gt / len(pooled):.2f}')
    head = ('PVSG/PSG4D protocol (class-level triplet, per-frame mask IoU>=0.5, vIoU>=0.5), in %.\n'
            'PSG4D-GTA trained, R/mR@100: PSG4DFormer RGB-D 7.22/4.02 (PSG4D Table 2); best reported 4D-LLM 18.43/12.41 (PSG-4D-LLM Table 2)\n')
    txt = head + '\n'.join(lines) + '\n'
    print(txt)
    open(os.path.join(S.OUT, 'eval_pvsg.txt'), 'w').write(txt)
    with open(os.path.join(S.OUT, 'eval_pvsg.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['video', 'path', 'gt', 'pred', 'R@100', 'softR@100', 'weakR@100']] + rows)


if __name__ == '__main__':
    main()
