#!/usr/bin/env python3
"""
Path B, step 3b+4b — PixelRefer (VideoRefer suite successor) captions each
window from PLAIN frames + object masks. No painted marks. ONE call per
window with ALL of its objects as region tokens.

Follows alibaba-damo-academy/PixelRefer, PixelRefer/notebooks/
pixelrefer-infer-video.ipynb (multi-object cell) and
evaluation/pixelrefer/infer_videorefer_q.py:
    model, processor, tokenizer = model_init(model_path)
    video = load_video(<frame dir>, fps=None, max_frames=K, frame_ids=[...])
    masks = torch uint8 [N, H, W]   (one mask per object, full resolution)
    mask_ids = [index into frame_ids for each mask]
    prompt  = '<video>\\nThere are N objects in the video: <object0><region>,
               <object1><region>, ... .' + instruction
    mm_infer(video, prompt, model=model, tokenizer=tokenizer, masks=masks,
             mask_ids=mask_ids, modal='video')          (greedy by default)

Differences from the notebook, and why:
  * the notebook loads an .mp4; SAIL-VOS is a folder of BMPs, and
    load_video(<dir>) is PixelRefer's own folder loader. We give it a temp
    folder of symlinks to the window's frames. That loader hard-codes 3 fps
    for timestamps and samples uniformly when fps=None, so we pass fps=None
    and max_frames=--clip-frames.
  * masks come from the SAM3 tubes: every object on the window's ref_frame,
    except objects absent there, which use their own best frame (mask_ids
    then points at that frame — the same mechanism the notebook uses with
    frame_ids).
  * the model answers with "object0 ..."; we rewrite objectK -> "mark <tid>"
    so extraction downstream is IDENTICAL for both paths.

Env: PixelRefer pins transformers 4.46.3 / torch 2.4 — run this in its own
env. It only needs sgcore.py, numpy, opencv, pycocotools from here.

Output: OUT/<vid>/captions_ref.jsonl
  {window_id, span, objects, caption (marks), raw_caption, prompt,
   frame_ids, mask_ids}

    python pixelrefer_caption.py --data-root $DATA --videos fam_1_mcs_2 \
        --pixelrefer-root ~/PixelRefer/PixelRefer --model Alibaba-DAMO-Academy/PixelRefer-7B
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

import numpy as np

import sgcore as S


def build_prompt(w, dossiers, names, version='v1', predicates=(), motion=''):
    objs = w['objects']
    ref = lambda t, k: f'object{k}'
    regions = ', '.join(f'<object{k}><region>' for k in range(len(objs)))
    head = f'<video>\nThere are {len(objs)} objects in the video: {regions}.\n'
    legend = S.legend(objs, dossiers, names, ref)
    instr = S.caption_instruction(
        version, predicates, n_min=3, n_max=8, ref_style='"objectK" using its index',
        ex_a='object0', ex_b='object1')
    return head + legend + ('\n\n' + motion if motion else '') + '\n\n' + instr


def window_motion(w, tubes, names, stuff):
    import motion as M
    idx = {int(t): k for k, t in enumerate(w['objects'])}
    return M.facts_block(M.motion_facts(w, tubes, names, stuff, ref=lambda t: f'object{idx[int(t)]}'))


def to_marks(text, objs):
    """object3 / Object 3 / <object3> -> 'mark <tid of object 3>'."""
    def sub(m):
        k = int(m.group(1))
        return f'mark {objs[k]}' if k < len(objs) else m.group(0)
    text = re.sub(r'<\s*object\s*(\d+)\s*>(\s*<region>)?', sub, text, flags=re.I)
    text = re.sub(r'\bobject\s*(\d+)\b', sub, text, flags=re.I)
    return text.replace('<region>', '')


def window_inputs(w, tubes, frames, tmp):
    """Temp frame folder for the span + masks/mask_ids for all objects."""
    s0, s1 = w['span']
    for i, f in enumerate(range(s0, s1 + 1)):
        ext = os.path.splitext(frames[f])[1]
        os.symlink(os.path.abspath(frames[f]), os.path.join(tmp, f'{i:05d}{ext}'))
    ref_frames = sorted({w['obj_ref'][str(t)] for t in w['objects']})
    frame_ids = [f - s0 for f in ref_frames]           # indices inside tmp dir
    masks, mask_ids = [], []
    for t in w['objects']:
        f = w['obj_ref'][str(t)]
        m = tubes.mask(t, f)
        if m is None or not m.any():
            raise ValueError(f'object {t} has an empty mask at frame {f}')
        masks.append(m.astype(np.uint8))
        mask_ids.append(ref_frames.index(f))
    return frame_ids, np.stack(masks, 0), mask_ids


class PixelRefer:
    def __init__(self, args):
        sys.path.insert(0, os.path.expanduser(args.pixelrefer_root))
        # flash_attn is not installed — intercept the check in transformers and
        # fall back to sdpa (PyTorch built-in) for every sub-model that requests it.
        try:
            import transformers.modeling_utils as _mu
            _orig = _mu.PreTrainedModel._check_and_enable_flash_attn_2
            @classmethod
            def _safe_flash_check(cls, config, *a, **kw):
                try:
                    return _orig.__func__(cls, config, *a, **kw)
                except ImportError:
                    config._attn_implementation = 'sdpa'
                    return config
            _mu.PreTrainedModel._check_and_enable_flash_attn_2 = _safe_flash_check
        except Exception:
            pass
        pkg = __import__(args.pkg)
        from importlib import import_module
        mm = import_module(f'{args.pkg}.mm_utils')
        pkg.disable_torch_init()
        self.model, self.processor, self.tokenizer = pkg.model_init(args.model, attn_implementation=getattr(args,'attn_impl','sdpa'))
        self.mm_infer, self.load_video = pkg.mm_infer, mm.load_video
        # PXR_GPU_FRAC: cap PyTorch's share of VRAM so a huge window raises a normal
        # CUDA OOM (caught per window below) instead of growing past tools/guard.sh's
        # MAX_GPU_MIB and getting the whole job killed (2026-11-04 v5 run, 48.4 GB).
        frac = os.environ.get('PXR_GPU_FRAC')
        if frac:
            import torch
            torch.cuda.set_per_process_memory_fraction(float(frac))

    def __call__(self, tmp, frame_ids, masks, mask_ids, prompt, args):
        import torch
        video = self.load_video(tmp, fps=None, max_frames=args.clip_frames,
                                frame_ids=frame_ids)
        return self.mm_infer(video, prompt, model=self.model, tokenizer=self.tokenizer,
                             masks=torch.from_numpy(masks).to(torch.uint8),
                             mask_ids=mask_ids, modal='video',
                             do_sample=False, max_new_tokens=args.max_new_tokens)


class FakePixelRefer:
    """Checks inputs the way PixelRefer's own code would use them."""

    def __call__(self, tmp, frame_ids, masks, mask_ids, prompt, args):
        files = sorted(os.listdir(tmp))
        n = prompt.count('<region>')
        assert n == len(masks) == len(mask_ids), (n, len(masks), len(mask_ids))
        assert all(0 <= i < len(frame_ids) for i in mask_ids)
        assert all(0 <= f < len(files) for f in frame_ids)
        assert masks.dtype == np.uint8 and masks.ndim == 3
        assert all(m.sum() > 0 for m in masks)       # resize_image_mask needs rows
        return ('object0 is holding object1. object1 is next to object0.'
                + (' Object2 is looking at object0.' if n > 2 else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--pixelrefer-root', default='~/PixelRefer/PixelRefer',
                    help='folder containing the pixelrefer/ package')
    ap.add_argument('--pkg', default='pixelrefer', choices=['pixelrefer', 'pixelreferlite'])
    ap.add_argument('--model', default='Alibaba-DAMO-Academy/PixelRefer-7B')
    ap.add_argument('--clip-frames', type=int, default=16)
    ap.add_argument('--max-new-tokens', type=int, default=400)
    ap.add_argument('--max-windows', type=int, default=0)
    ap.add_argument('--attn-impl', default='sdpa', choices=['sdpa','flash_attention_2','eager'], help='attention implementation; sdpa needs no extra install')
    ap.add_argument('--fake', action='store_true')
    S.add_prompt_arg(ap)
    args = ap.parse_args()
    predicates = S.predicate_list(args.data_root)
    stuff = S.stuff_names(args.data_root) if args.prompt == 'v5' else set()
    model = FakePixelRefer() if args.fake else PixelRefer(args)

    for vid in args.videos:
        tubes = S.Tubes(S.vdir(vid, 'masks.txt'))
        frames = S.frames_of(args.data_root, vid)
        names = S.class_names(vid, args.data_root)
        dossiers = S.load_dossiers(vid)
        opath = S.vdir(vid, 'captions_ref.jsonl')
        done = {r['window_id']: r for r in S.io_jsonl(opath)}
        rows, n_err = [], 0
        ws = S.load_windows(vid)
        ws = ws[:args.max_windows] if args.max_windows else ws
        for w in ws:
            if S.caption_matches(done.get(w['window_id']), w, args.prompt):
                rows.append(done[w['window_id']]); continue
            mot = window_motion(w, tubes, names, stuff) if args.prompt == 'v5' else ''
            prompt = build_prompt(w, dossiers, names, args.prompt, predicates, mot)
            tmp = tempfile.mkdtemp(prefix='pxr_')
            clip_used = args.clip_frames
            try:
                frame_ids, masks, mask_ids = window_inputs(w, tubes, frames, tmp)
                try:
                    raw = model(tmp, frame_ids, masks, mask_ids, prompt, args).strip()
                except Exception as e:
                    # Out of GPU memory: retry this one window once with half the clip
                    # frames (recorded per row as 'clip_frames'); anything else fails.
                    if 'out of memory' not in str(e).lower() or args.fake:
                        raise
                    import torch
                    torch.cuda.empty_cache()
                    clip_used = max(4, args.clip_frames // 2)
                    print(f'[{vid}] window {w["window_id"]}: CUDA OOM, retrying with {clip_used} frames', flush=True)
                    raw = model(tmp, frame_ids, masks, mask_ids, prompt,
                                argparse.Namespace(**{**vars(args), 'clip_frames': clip_used})).strip()
            except Exception as e:
                n_err += 1
                print(f'[{vid}] window {w["window_id"]}: PixelRefer failed ({e!r})')
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            rows.append({'window_id': w['window_id'], 'span': w['span'],
                         'objects': w['objects'], 'caption': to_marks(raw, w['objects']),
                         'raw_caption': raw, 'prompt': prompt, 'prompt_version': args.prompt,
                         'frame_ids': frame_ids, 'mask_ids': mask_ids, 'clip_frames': clip_used})
            S.io_jsonl(opath, rows)
            print(f'[{vid}] ref w{w["window_id"]}: {rows[-1]["caption"][:90]!r}', flush=True)
        S.io_jsonl(opath, rows)
        print(f'[{vid}] {len(rows)} PixelRefer captions ({n_err} failed, rerun resumes) -> {opath}')


if __name__ == '__main__':
    main()
