#!/usr/bin/env python3
"""
Step 1 — import SAM3 masks and cut the video into TIME windows.

Replaces tube_doctor + tube_geometry. There are no pairs: a window is a time
span, and every object visible in it (up to --max-objects) is sent to the
captioner together, in ONE call. Both paths (SoM and PixelRefer) read the same
windows.json, so they see exactly the same frames and the same objects.

Per window:
  span        [t1, t2]           --win frames, a new window every --stride
  objects     [tid, ...]         visible >= --min-vis frames with area
                                 >= --min-area px; things before stuff, then
                                 by total visible area; at most --max-objects
  keyframes   [f, ...]           --keyframes frames spread over the span (SoM
                                 draws its marks on these)
  clip        [f, ...]           --clip-frames frames spread over the span
                                 (PixelRefer's video input)
  ref_frame   f                  frame showing the most objects (PixelRefer
                                 region masks are taken here)
  obj_ref     {tid: f}           ref_frame if the object is visible there,
                                 else its largest frame in the span

    python make_windows.py --data-root $DATA --videos fam_1_mcs_2 \
        --masks-from ../work_dirs/frontend_sam3_orig
"""
import argparse
import json
import os
import shutil

import sgcore as S


def build(vid, tubes, T, stuff_ids, args):
    windows = []
    starts = list(range(0, max(1, T - args.win + 1), args.stride))
    if starts[-1] + args.win < T:
        starts.append(T - args.win)          # last window reaches the end
    for s in starts:
        span = [s, min(T, s + args.win) - 1]
        fr = range(span[0], span[1] + 1)
        cand = []
        for tid in tubes.tids():
            vis = [f for f in fr if tubes.area(tid, f) >= args.min_area]
            if len(vis) >= args.min_vis:
                tot = sum(tubes.area(tid, f) for f in vis)
                cand.append((tid in stuff_ids, -tot, tid, vis))
        if args.no_stuff:
            cand = [c for c in cand if not c[0]]
        cand.sort()
        cand = cand[:args.max_objects]
        if len(cand) < 2:
            continue                          # no relation possible
        objs = sorted(c[2] for c in cand)
        vis_of = {c[2]: set(c[3]) for c in cand}
        busy = [f for f in fr if any(f in vis_of[t] for t in objs)]
        ref = max(busy, key=lambda f: (sum(f in vis_of[t] for t in objs),
                                       sum(tubes.area(t, f) for t in objs)))
        obj_ref = {}
        for t in objs:
            obj_ref[str(t)] = ref if ref in vis_of[t] else \
                max(vis_of[t], key=lambda f: tubes.area(t, f))
        windows.append({
            'window_id': len(windows), 'span': span, 'objects': objs,
            'keyframes': S.uniform(busy, args.keyframes),
            'clip': S.uniform(list(fr), args.clip_frames),
            'ref_frame': ref, 'obj_ref': obj_ref,
            'classes': {str(t): tubes.cls[t] for t in objs},
        })
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--masks-from', default='',
                    help='dir holding <vid>/masks.txt from run_sam3.py; copied '
                         'into OUT/<vid>/ (skip if already there)')
    ap.add_argument('--win', type=int, default=30)
    ap.add_argument('--stride', type=int, default=15)
    ap.add_argument('--keyframes', type=int, default=4)
    ap.add_argument('--clip-frames', type=int, default=16)
    ap.add_argument('--min-vis', type=int, default=3)
    ap.add_argument('--min-area', type=int, default=400)
    ap.add_argument('--max-objects', type=int, default=10,
                    help='cap per window; SoM gets cluttered beyond ~10 marks')
    ap.add_argument('--no-stuff', action='store_true',
                    help='drop stuff classes (wall, road, ...) from windows')
    args = ap.parse_args()
    stuffs = S.stuff_names(args.data_root)

    for vid in args.videos:
        dst = S.vdir(vid, 'masks.txt')
        if args.masks_from:
            src_dir = os.path.join(os.path.expanduser(args.masks_from), vid)
            shutil.copy2(os.path.join(src_dir, 'masks.txt'), dst)
            if os.path.exists(os.path.join(src_dir, 'vocab.json')):
                shutil.copy2(os.path.join(src_dir, 'vocab.json'), S.vdir(vid, 'vocab.json'))
            print(f'[{vid}] copied masks from {src_dir}')
        if not os.path.exists(dst):
            raise SystemExit(f'[{vid}] no {dst}; pass --masks-from')
        tubes = S.Tubes(dst)
        names = S.class_names(vid, args.data_root)
        stuff_ids = {t for t, c in tubes.cls.items()
                     if 0 <= c < len(names) and names[c] in stuffs}
        frame_list = S.frames_of(args.data_root, vid)
        T = len(frame_list)
        last = max(tubes.per_frame) if tubes.per_frame else -1
        if T == 0:
            # frames_of() returned nothing. Print what path it searched so the
            # user can fix DATA or common.py, then fall back to mask extent.
            import common as _C
            import inspect
            src = inspect.getsource(_C.frames_of)
            print(f'[{vid}] ERROR: frames_of() returned 0 images from DATA={args.data_root}')
            print(f'       frames_of source:\n{src}')
            T = last + 1
            print(f'[{vid}] Falling back to mask extent: T={T} frames (describe_dam / '
                  f'som_render will skip — fix DATA first for the real run)')
        elif last >= T:
            print(f'[{vid}] WARNING masks reach frame {last} but only {T} images')
        ws = build(vid, tubes, T, stuff_ids, args)
        json.dump(ws, open(S.vdir(vid, 'windows.json'), 'w'), indent=1)
        n_obj = [len(w['objects']) for w in ws]
        print(f'[{vid}] {len(tubes.tids())} tubes, {T} frames -> {len(ws)} windows '
              f'(= calls per path), objects/window mean '
              f'{sum(n_obj) / max(1, len(n_obj)):.1f} max {max(n_obj, default=0)}')


if __name__ == '__main__':
    main()
