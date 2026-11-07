#!/usr/bin/env python3
"""
Step 2 — Describe Anything (DAM) dossier for every tube (one call per TUBE,
never per pair). Shared by both paths: its short description goes into the
object legend of both the SoM prompt and the PixelRefer prompt.

Follows NVlabs/describe-anything examples/dam_video_with_sam2.py exactly:
  model nvidia/DAM-3B-Video, conv_mode v1, prompt_mode 'full+focal_crop',
  8 frames sampled uniformly, masks as PIL 'L' images (0/255),
  query 'Video: <image>x8\\nGiven the video in the form of a sequence of
  frames above, describe the object in the masked region in the video in
  detail.', temperature 0.2, top_p 0.5, num_beams 1, max_new_tokens 512.
  Image checkpoint (--dam-model nvidia/DAM-3B) uses one frame and
  '<image>\\nDescribe the masked region in detail.' like examples/dam_with_sam.py.

We sample the 8 frames from the frames where the tube is at least median
size (small, clipped views describe badly); fewer than 8 -> repeated.

    python describe_dam.py --data-root $DATA --videos fam_1_mcs_2 \
        --dam-repo ~/describe-anything
"""
import argparse
import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image

import sgcore as S

VIDEO_QUERY = ('Video: ' + '<image>' * 8 + '\nGiven the video in the form of a '
               'sequence of frames above, describe the object in the masked '
               'region in the video in detail.')
IMAGE_QUERY = '<image>\nDescribe the masked region in detail.'


def first_sentence(text, max_words=18):
    s = re.split(r'(?<=[.!?])\s', text.strip(), maxsplit=1)[0]
    w = s.split()
    return ' '.join(w[:max_words]).rstrip(',;') + ('...' if len(w) > max_words else '')


def pick_frames(tubes, tid, k):
    fr = tubes.frames_of[tid]
    fs = sorted(fr)
    med = float(np.median([fr[f][0] for f in fs]))
    big = [f for f in fs if fr[f][0] >= med] or fs
    sel = S.uniform(big, k)
    while len(sel) < k:                       # DAM-3B-Video expects 8 views
        sel = (sel * 2)[:k]
    return sorted(sel)


class DAM:
    def __init__(self, args):
        if args.dam_repo:
            sys.path.insert(0, os.path.expanduser(args.dam_repo))
        import torch
        from dam import DescribeAnythingModel, disable_torch_init
        disable_torch_init()
        model_path = args.dam_model
        if not os.path.exists(model_path):    # dam's loader calls repo_exists(), which fails offline
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_path, local_files_only=True)
        self.dam = DescribeAnythingModel(
            model_path=model_path, conv_mode='v1',
            prompt_mode='full+focal_crop').to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.video = 'Video' in args.dam_model

    def describe(self, images, masks, args):
        q = VIDEO_QUERY if self.video else IMAGE_QUERY
        if not self.video:
            images, masks = images[:1], masks[:1]
        return self.dam.get_description(
            images if self.video else images[0], masks if self.video else masks[0],
            q, streaming=False, temperature=args.temperature, top_p=args.top_p,
            num_beams=1, max_new_tokens=args.max_new_tokens)


class FakeDAM:
    video = True

    def describe(self, images, masks, args):
        assert len(images) == len(masks) == 8
        assert all(m.mode == 'L' and np.asarray(m).max() == 255 for m in masks)
        return 'A person in a dark jacket stands still. They hold something small.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--dam-model', default='nvidia/DAM-3B-Video')
    ap.add_argument('--dam-repo', default='', help='clone of NVlabs/describe-anything '
                    '(not needed if `pip install -e` was done)')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--top-p', type=float, default=0.5)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--fake', action='store_true', help='no model (plumbing test)')
    args = ap.parse_args()
    model = FakeDAM() if args.fake else DAM(args)

    for vid in args.videos:
        tubes = S.Tubes(S.vdir(vid, 'masks.txt'))
        names = S.class_names(vid, args.data_root)
        frames = S.frames_of(args.data_root, vid)
        opath = S.vdir(vid, 'dossiers.json')
        out = {int(k): v for k, v in json.load(open(opath)).items()} \
            if os.path.exists(opath) else {}
        used = {t for w in S.load_windows(vid) for t in w['objects']}
        for tid in sorted(used):
            if tid in out and out[tid].get('description'):
                continue                                  # resume
            sel = pick_frames(tubes, tid, 8)
            imgs, msks = [], []
            for f in sel:
                im = cv2.imread(frames[f])
                imgs.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                msks.append(Image.fromarray(tubes.mask(tid, f).astype(np.uint8) * 255))
            cid = tubes.cls[tid]
            try:
                desc = model.describe(imgs, msks, args).strip()
            except Exception as e:
                print(f'[{vid}] tid {tid}: DAM failed ({e!r})')
                desc = ''
            out[tid] = {'class_id': cid,
                        'class_name': names[cid] if 0 <= cid < len(names) else 'object',
                        'frames': sel, 'description': desc,
                        'short': first_sentence(desc) if desc else ''}
            json.dump({str(k): v for k, v in out.items()}, open(opath, 'w'), indent=1)
            print(f'[{vid}] tid {tid} ({out[tid]["class_name"]}): {out[tid]["short"]}', flush=True)
        print(f'[{vid}] {len(out)} dossiers -> {opath}')


if __name__ == '__main__':
    main()
