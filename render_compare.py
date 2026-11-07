#!/usr/bin/env python3
"""
Step 9 — comparison video: SoM scene graph vs PixelRefer scene graph (and
the ground truth, if eval_relations.py has run), unfolding with the footage.

  top      the frame, every tracked object outlined and labelled name#id
  bottom   one panel per graph. Nodes are the objects that appear in any
           relation (same position in every panel, same colour as the SoM
           marks); a node is dim when the object is not visible. An edge is
           drawn while its interval covers the current frame:
             green  = also a GT relation right now (needs eval)
             orange = not in GT (or GT unavailable)
             cyan   = GT panel
           The list under each graph spells out the active triplets.

Output: OUT/<vid>/compare_<vid>.mp4 (H.264 if ffmpeg is installed, so it plays
in browsers and slides; plain mp4v otherwise)

    python render_compare.py --data-root $DATA --videos fam_1_mcs_2
"""
import argparse
import json
import math
import os
import shutil
import subprocess

import cv2
import numpy as np

import sgcore as S
from som_render import color

PANEL_W, PANEL_H, GRAPH_H = 470, 430, 300
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, ORANGE, CYAN, DIM = (90, 220, 90), (40, 160, 255), (230, 210, 60), (90, 90, 90)


def label(tid, dossiers, names, tubes):
    d = dossiers.get(tid)
    if d:
        return f"{d['class_name']}#{tid}"
    c = tubes.cls.get(tid, -1)
    return f"{names[c] if 0 <= c < len(names) else 'obj'}#{tid}"


def active(rels, f):
    return [r for r in rels if any(a <= f <= b for a, b in r['intervals'])]


def arrow(img, p, q, col, text, k):
    (x1, y1), (x2, y2) = p, q
    d = math.hypot(x2 - x1, y2 - y1) or 1
    sh = 18 / d                                    # stop at the node rim
    a = (int(x1 + (x2 - x1) * sh), int(y1 + (y2 - y1) * sh))
    b = (int(x2 - (x2 - x1) * sh), int(y2 - (y2 - y1) * sh))
    cv2.arrowedLine(img, a, b, col, 2, cv2.LINE_AA, tipLength=min(1, 12 / d))
    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2 + 14 * k
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
    cv2.rectangle(img, (mx - tw // 2 - 3, my - th - 3), (mx + tw // 2 + 3, my + 3), (25, 25, 25), -1)
    cv2.putText(img, text, (mx - tw // 2, my), FONT, 0.42, col, 1, cv2.LINE_AA)


def panel(title, rels, f, pos, labels, visible, gt_now):
    img = np.full((PANEL_H, PANEL_W, 3), 22, np.uint8)
    act = active(rels, f)
    cv2.putText(img, title, (12, 24), FONT, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(img, f'{len(act)} active / {len(rels)} relations', (12, 44), FONT, 0.42,
                (160, 160, 160), 1, cv2.LINE_AA)
    for tid, (x, y) in pos.items():
        c = color(tid) if tid in visible else DIM
        cv2.circle(img, (x, y), 14, c, -1 if tid in visible else 2, cv2.LINE_AA)
        cv2.putText(img, labels[tid], (x - 30, y + 30), FONT, 0.38,
                    (220, 220, 220) if tid in visible else DIM, 1, cv2.LINE_AA)
    seen = {}
    for r in act:
        s, o = r['subject'], r['object']
        if s not in pos or o not in pos:
            continue
        k = seen.get((s, o), 0); seen[(s, o)] = k + 1
        col = CYAN if gt_now is None else (GREEN if (s, o, r['predicate']) in gt_now else ORANGE)
        arrow(img, pos[s], pos[o], col, r['predicate'], k)
    y = GRAPH_H + 58
    for r in act[:6]:
        col = CYAN if gt_now is None else (GREEN if (r['subject'], r['object'], r['predicate'])
                                            in gt_now else ORANGE)
        t = f"{labels.get(r['subject'], r['subject'])} {r['predicate']} {labels.get(r['object'], r['object'])}"
        cv2.putText(img, t[:60], (12, y), FONT, 0.42, col, 1, cv2.LINE_AA)
        y += 18
    if len(act) > 6:
        cv2.putText(img, f'+{len(act) - 6} more', (12, y), FONT, 0.4, DIM, 1, cv2.LINE_AA)
    return img


def layout(tids):
    cx, cy, r = PANEL_W // 2, 60 + (GRAPH_H - 30) // 2, min(PANEL_W, GRAPH_H) // 2 - 45
    n = max(1, len(tids))
    return {t: (int(cx + r * math.cos(2 * math.pi * i / n - math.pi / 2)),
                int(cy + r * math.sin(2 * math.pi * i / n - math.pi / 2)))
            for i, t in enumerate(sorted(tids))}


def gt_relations(vid, t2g):
    """GT triplets re-expressed on tube ids (first tube mapped to each GT id)."""
    for path in S.PATHS:
        p = S.vdir(vid, f'eval_{path}.json')
        if os.path.exists(p):
            gt = json.load(open(p))['gt']
            break
    else:
        return None
    inv = {}
    for t, g in sorted(t2g.items()):
        inv.setdefault(g, t)
    return [{'subject': inv[s], 'object': inv[o], 'predicate': p, 'intervals': iv or [[0, 10 ** 9]]}
            for s, o, p, iv in gt if s in inv and o in inv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--width', type=int, default=960, help='frame width in the video')
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()
    for vid in args.videos:
        tubes = S.Tubes(S.vdir(vid, 'masks.txt'))
        frames = S.frames_of(args.data_root, vid)
        names = S.class_names(vid, args.data_root)
        dossiers = S.load_dossiers(vid)
        graphs = []
        for path in S.PATHS:
            p = S.vdir(vid, f'relations_{path}.json')
            if os.path.exists(p):
                graphs.append((S.PATH_NAME[path], json.load(open(p))['relations']))
        if not graphs:
            print(f'[{vid}] no relations_*.json — run fuse first'); continue
        t2g_p = S.vdir(vid, 'tube2gt.json')
        t2g = {int(k): v for k, v in json.load(open(t2g_p)).items()} if os.path.exists(t2g_p) else {}
        gt = gt_relations(vid, t2g) if t2g else None
        if gt is not None:
            graphs.append(('Ground truth (on SAM3 tubes)', gt))
        tids = {t for _, rels in graphs for r in rels for t in (r['subject'], r['object'])}
        pos = layout(tids)
        labels = {t: label(t, dossiers, names, tubes) for t in tids}

        H0, W0 = cv2.imread(frames[0]).shape[:2]
        sc = args.width / W0
        fh = int(H0 * sc)
        cw = max(args.width, PANEL_W * len(graphs))
        out = S.vdir(vid, f'compare_{vid}.mp4')
        tmp = out + '.mp4v.mp4'
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (cw, fh + PANEL_H))
        for f in range(0, len(frames), args.stride):
            im = cv2.resize(cv2.imread(frames[f]), (args.width, fh), interpolation=cv2.INTER_AREA)
            visible = set()
            for tid, _, _ in tubes.per_frame.get(f, []):
                m = tubes.mask(tid, f)
                if m is None or not m.any():
                    continue
                visible.add(tid)
                m = cv2.resize(m.astype(np.uint8), (args.width, fh), interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cnts, -1, color(tid), 2)
                ys, xs = np.nonzero(m)
                cv2.putText(im, label(tid, dossiers, names, tubes), (int(xs.min()), max(14, int(ys.min()) - 4)),
                            FONT, 0.45, color(tid), 1, cv2.LINE_AA)
            cv2.putText(im, f'{vid}  frame {f}', (10, 22), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            gt_now = ({(r['subject'], r['object'], r['predicate']) for r in active(gt, f)}
                      if gt is not None else set())       # no GT -> every edge orange
            canvas = np.full((fh + PANEL_H, cw, 3), 12, np.uint8)
            x0 = (cw - args.width) // 2
            canvas[:fh, x0:x0 + args.width] = im
            for i, (title, rels) in enumerate(graphs):
                is_gt = title.startswith('Ground')
                canvas[fh:, i * PANEL_W:(i + 1) * PANEL_W] = panel(
                    title, rels, f, pos, labels, visible, None if is_gt else gt_now)
            vw.write(canvas)
        vw.release()
        if shutil.which('ffmpeg'):
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-crf', '23', out], check=False)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            os.remove(tmp)
        else:
            os.replace(tmp, out)
        print(f'[{vid}] {len(graphs)} graphs -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
