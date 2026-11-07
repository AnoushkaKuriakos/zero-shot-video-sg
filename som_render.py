#!/usr/bin/env python3
"""
Path A, step 3a — Set-of-Mark keyframes, NON-pairwise: every object of the
window is marked at once on each keyframe (mark number = track id).

Drawing follows microsoft/SoM task_adapter/utils/visualizer.py:
  * masks drawn largest-first (inference_sam_m2m_auto sorts by area, desc),
    so small objects stay on top
  * translucent fill (SoM demo default alpha 0.1; we default 0.25 because
    our frames are downscaled by the VLM) + mask outline
  * number placed at the max of the distance transform of the mask (the
    median of all max points, as _draw_number_in_mask does), white text on a
    dark box
  plus: numbers never overlap each other, and size scales with the image.

Output: OUT/<vid>/som/w<id>_f<frame>.jpg + som_index.json {window_id: [paths]}

    python som_render.py --data-root $DATA --videos fam_1_mcs_2
"""
import argparse
import json
import os

import cv2
import numpy as np

import sgcore as S

FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
           (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
           (170, 110, 40), (128, 0, 0), (170, 255, 195), (0, 0, 128)]


def color(tid):                     # BGR
    r, g, b = PALETTE[tid % len(PALETTE)]
    return (b, g, r)


def number_anchor(mask):
    """SoM's rule: point(s) farthest from the mask boundary, take the median."""
    m = np.pad(mask.astype(np.uint8), ((1, 1), (1, 1)), 'constant')
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 0)[1:-1, 1:-1]
    ys, xs = np.where(dt == dt.max())
    return int(xs[len(xs) // 2]), int(ys[len(ys) // 2])


def overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def draw_number(im, text, anchor, col, scale, thick, taken):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    pad = max(3, th // 3)
    bw, bh = tw + 2 * pad, th + base + 2 * pad
    H, W = im.shape[:2]
    for dx, dy in [(0, 0), (0, bh + 3), (0, -bh - 3), (bw + 3, 0), (-bw - 3, 0),
                   (bw + 3, bh + 3), (-bw - 3, -bh - 3)]:
        x0 = int(np.clip(anchor[0] + dx - bw // 2, 0, max(0, W - bw - 1)))
        y0 = int(np.clip(anchor[1] + dy - bh // 2, 0, max(0, H - bh - 1)))
        r = (x0, y0, x0 + bw, y0 + bh)
        if not any(overlaps(r, o) for o in taken):
            break
    cv2.rectangle(im, r[:2], r[2:], (0, 0, 0), -1)
    cv2.rectangle(im, r[:2], r[2:], col, max(1, thick // 2))
    cv2.putText(im, text, (r[0] + pad, r[3] - pad - base // 2), FONT, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
    taken.append(r)


def render(im, items, alpha, tag_frac):
    """items: [(tid, mask)]. Largest first, as SoM does."""
    H, W = im.shape[:2]
    items = sorted(items, key=lambda t: -int(t[1].sum()))
    line = max(2, round(W / 640))
    for tid, m in items:
        c = np.array(color(tid), np.float32)
        im[m] = (im[m].astype(np.float32) * (1 - alpha) + c * alpha).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, color(tid), line)
    scale = tag_frac * max(H, W) / 22.0
    thick = max(1, round(scale * 2))
    taken = []
    for tid, m in reversed(items):              # small objects pick first
        draw_number(im, str(tid), number_anchor(m), color(tid), scale, thick, taken)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--alpha', type=float, default=0.25)
    ap.add_argument('--tag-frac', type=float, default=0.025,
                    help='number height as a fraction of the long image side')
    args = ap.parse_args()
    for vid in args.videos:
        tubes = S.Tubes(S.vdir(vid, 'masks.txt'))
        frames = S.frames_of(args.data_root, vid)
        odir = S.vdir(vid, 'som')
        index = {}
        for w in S.load_windows(vid):
            paths = []
            for f in w['keyframes']:
                im = cv2.imread(frames[f])
                items = [(t, tubes.mask(t, f)) for t in w['objects'] if tubes.visible(t, f)]
                items = [(t, m) for t, m in items if m is not None and m.any()]
                p = os.path.join(odir, f'w{w["window_id"]:04d}_f{f:06d}.jpg')
                cv2.imwrite(p, render(im, items, args.alpha, args.tag_frac),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                paths.append(p)
            index[str(w['window_id'])] = paths
        json.dump(index, open(S.vdir(vid, 'som_index.json'), 'w'), indent=1)
        print(f'[{vid}] rendered {sum(map(len, index.values()))} keyframes for '
              f'{len(index)} windows -> {odir}')


if __name__ == '__main__':
    main()
