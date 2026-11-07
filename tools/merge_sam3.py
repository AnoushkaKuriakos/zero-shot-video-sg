#!/usr/bin/env python3
"""
Merge a partial SAM3 run (run_sam3.py --class-list ...) into a full one.

  --mode replace  drop the base tracks of every class in the extra run, then add the
                  extra tracks (e.g. a person-only rerun with a higher --max-objects)
  --mode add      keep the base tracks and add the extra ones (e.g. ground synonyms)

Extra track ids are shifted above the base ids. The base masks.txt is backed up once
to masks_before_merge.txt; reruns merge from that backup, so they are idempotent.

    python tools/merge_sam3.py --base $MASKS --extra ~/.../frontend_sam3_person \
        --videos mic_4_int fin_c_ext --mode replace
    python tools/merge_sam3.py --base $MASKS --extra ... --out exp/ground --mode add
"""
import argparse
import os
import shutil


def read_rows(path):
    return [l.rstrip('\n').split(' ', 5) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='dir with <vid>/masks.txt (full run)')
    ap.add_argument('--extra', required=True, help='dir with <vid>/masks.txt (partial run)')
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--mode', choices=['replace', 'add'], default='replace')
    ap.add_argument('--out', default='', help='write <out>/<vid>/masks.txt instead of into --base')
    args = ap.parse_args()

    for vid in args.videos:
        bdir, edir = os.path.join(args.base, vid), os.path.join(args.extra, vid)
        bpath, epath = os.path.join(bdir, 'masks.txt'), os.path.join(edir, 'masks.txt')
        if not os.path.exists(epath):
            print(f'[{vid}] no {epath} — skipped'); continue
        backup = os.path.join(bdir, 'masks_before_merge.txt')
        if not args.out and not os.path.exists(backup):
            shutil.copy2(bpath, backup)
        base = read_rows(backup if os.path.exists(backup) else bpath)
        extra = read_rows(epath)
        ecids = {r[2] for r in extra}   # a class with no new tracks keeps its base tracks
        if args.mode == 'replace':
            dropped = {r[1] for r in base if r[2] in ecids}
            base = [r for r in base if r[2] not in ecids]
        else:
            dropped = set()
        off = max((int(r[1]) for r in base), default=0)
        merged = base + [[r[0], str(int(r[1]) + off)] + r[2:] for r in extra]
        merged.sort(key=lambda r: (int(r[0]), int(r[1])))
        odir = os.path.join(args.out, vid) if args.out else bdir
        os.makedirs(odir, exist_ok=True)
        with open(os.path.join(odir, 'masks.txt'), 'w') as f:
            f.writelines(' '.join(r) + '\n' for r in merged)
        if args.out and os.path.exists(os.path.join(bdir, 'vocab.json')):
            shutil.copy2(os.path.join(bdir, 'vocab.json'), odir)
        print(f'[{vid}] {args.mode}: -{len(dropped)} base tracks, '
              f'+{len({r[1] for r in extra})} extra tracks -> {odir}/masks.txt')


if __name__ == '__main__':
    main()
