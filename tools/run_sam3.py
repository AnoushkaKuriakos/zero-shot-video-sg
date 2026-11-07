#!/usr/bin/env python3
"""
Frontend A: SAM 3 — promptable concept segmentation. One model does detection,
segmentation AND tracking: prompt with a class name, get identity-consistent
masklets across the whole video. Writes the same contract as frontend B:

    <out>/<vid>/masks.txt          frame tid cid h w rle
    <out>/<vid>/vocab.json
    <out>/<vid>/qualitative/*.jpg  (--viz-every)

Notes:
  * SAM3's video predictor takes a JPEG folder or MP4, not BMPs — we convert
    each video's frames to a temp JPEG folder once (kept under <out>/<vid>/_jpg
    unless --keep-jpg is off).
  * One session per prompted class; track ids from different classes are
    remapped into a single id space.
  * The session API is `handle_request` / `handle_stream_request` with request
    dicts (start_session / add_prompt / propagate). Because Meta has iterated
    on exact field names (SAM 3 -> 3.1), the parsing below is deliberately
    tolerant and, on the first frame it cannot parse, prints the raw structure
    and points you at examples/sam3_video_predictor_example.ipynb in the sam3
    repo — one look at that notebook tells you which key to add.

MEMORY (this version): masks are streamed. Each propagated frame is encoded to
RLE the moment it arrives and the full-resolution arrays are dropped, so peak
RAM is one frame's masks + the RLE strings (a few KB per mask) instead of
every frame of every object. On top of that the session is started/propagated
with SAM3's own memory switches:
  evict_cached_frame_outputs=True   SAM3 drops its cached per-frame outputs
                                    after yielding them (sam3_base_predictor)
  --offload-video-cpu / --offload-state-cpu   keep frames / tracker state on
                                    CPU instead of GPU (start_session options)
  propagation_direction='forward' when prompting frame 0 (the default 'both'
                                    re-yields frame 0 on the backward pass)

Run:
    python tools/run_sam3.py --data-root $DATA \
        --out work/sam3 --videos mic_4_int fin_c_ext
"""
import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C


# --------------------------------------------------------------------------- #
# frame conversion                                                            #
# --------------------------------------------------------------------------- #
def bmp_to_jpg_folder(frames, jpg_dir, quality=95):
    os.makedirs(jpg_dir, exist_ok=True)
    existing = len([f for f in os.listdir(jpg_dir) if f.endswith('.jpg')])
    if existing == len(frames):
        return jpg_dir
    for i, fp in enumerate(frames):
        dst = os.path.join(jpg_dir, f'{i:05d}.jpg')
        if os.path.exists(dst):
            continue
        im = cv2.imread(fp)
        cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpg_dir


# --------------------------------------------------------------------------- #
# tolerant response parsing                                                   #
# --------------------------------------------------------------------------- #
def _describe(x, indent=0, max_depth=3):
    """Print the structure of an arbitrary response object."""
    pad = '  ' * indent
    try:
        import torch
        is_tensor = isinstance(x, torch.Tensor)
    except ImportError:
        is_tensor = False
    if is_tensor or isinstance(x, np.ndarray):
        arr = x.detach().cpu().numpy() if is_tensor else x
        extra = ''
        if arr.size and arr.dtype != object and arr.dtype.kind in 'fiub':
            extra = f' min={arr.min():.3g} max={arr.max():.3g}'
        print(f'{pad}{type(x).__name__} shape={tuple(arr.shape)} '
              f'dtype={arr.dtype}{extra}')
    elif isinstance(x, dict):
        print(f'{pad}dict({len(x)}) keys={list(x.keys())[:12]}')
        if indent < max_depth:
            for k in list(x.keys())[:8]:
                print(f'{pad} [{k!r}]:')
                _describe(x[k], indent + 1, max_depth)
    elif isinstance(x, (list, tuple)):
        print(f'{pad}{type(x).__name__}(len={len(x)})')
        if x and indent < max_depth:
            _describe(x[0], indent + 1, max_depth)
    elif isinstance(x, (str, bytes)):
        print(f'{pad}{type(x).__name__}(len={len(x)}) {x[:40]!r}...')
    else:
        print(f'{pad}{type(x).__name__} = {x!r}')


def _to_bool_mask(m, hw=None):
    """Accepts RLE dict / numpy / torch / nested [1,H,W]; returns bool HxW."""
    if m is None:
        return None
    if isinstance(m, (str, bytes)) and hw is not None:
        # bare RLE counts string (Meta's demo code emits these)
        counts = m.encode('ascii') if isinstance(m, str) else m
        from pycocotools import mask as mu
        try:
            return mu.decode({'size': list(hw), 'counts': counts}).astype(bool)
        except Exception:
            return None
    if isinstance(m, dict) and 'counts' in m:
        counts = m['counts']
        if isinstance(counts, str):
            counts = counts.encode('ascii')
        from pycocotools import mask as mu
        return mu.decode({'size': list(m['size']), 'counts': counts}).astype(bool)
    try:
        import torch
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
    except ImportError:
        pass
    m = np.asarray(m)
    while m.ndim > 2:
        m = m[0]
    if m.dtype != bool:
        # logits (have negatives) -> 0 threshold; probs/binary -> 0.5
        m = (m > 0) if m.size and m.min() < 0 else (m > 0.5)
    if hw is not None and m.shape != hw:
        m = cv2.resize(m.astype(np.uint8), (hw[1], hw[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
    return m


def parse_frame_output(item, hw):
    """Return (frame_index, [(obj_id, mask_bool), ...]) or None.
    Handles the shapes seen across sam3 releases."""
    if item is None or not isinstance(item, dict):
        return None
    fi = None
    for k in ('frame_index', 'frame_idx', 'frame'):
        if k in item:
            fi = int(item[k]); break
    if 'outputs' in item and isinstance(item['outputs'], dict):
        item = item['outputs']           # sam3: {'frame_index': i, 'outputs': {...}}
        if fi is None:
            for k in ('frame_index', 'frame_idx', 'frame'):
                if k in item:
                    fi = int(item[k]); break
    if fi is None:
        return None

    pairs = []
    # shape 1: parallel ids + masks ('masks' may be a list, ndarray or tensor;
    # never use `or` here — bool() on a tensor raises)
    ids = item.get('object_ids')
    if ids is None:
        ids = item.get('obj_ids')
    if ids is None:
        ids = item.get('out_obj_ids')
    ms = item.get('masks')
    if ms is None:
        ms = item.get('out_binary_masks')      # sam3 video predictor's key
    if ms is None:
        ms = item.get('pred_masks')
    if ms is None:
        ms = item.get('out_mask_logits')
    if ids is not None and ms is not None and len(ids) == 0:
        return fi, pairs                 # valid frame, zero detections
    if ids is not None and ms is not None:
        for oid, m in zip(ids, ms):
            mb = _to_bool_mask(m, hw)
            if mb is not None:
                pairs.append((int(oid), mb))
        return fi, pairs
    # shape 2: list of per-object dicts
    objs = item.get('results') or item.get('objects') or item.get('masklets')
    if isinstance(objs, (list, tuple)):
        for o in objs:
            if not isinstance(o, dict):
                continue
            oid = o.get('object_id', o.get('obj_id', o.get('id')))
            m = o.get('mask', o.get('masks', o.get('segmentation',
                                                   o.get('rle'))))
            mb = _to_bool_mask(m, hw)
            if oid is not None and mb is not None:
                pairs.append((int(oid), mb))
        return fi, pairs
    # shape 3: {'object_masks'/'masks': {obj_id: mask}} dict
    od = item.get('object_masks')
    if od is None and isinstance(item.get('masks'), dict):
        od = item.get('masks')
    if isinstance(od, dict):
        for oid, m in od.items():
            mb = _to_bool_mask(m, hw)
            if mb is not None:
                pairs.append((int(oid), mb))
        return fi, pairs
    # shape 4: the outputs dict itself is keyed by object id
    # (int keys, values are masks / RLEs / {'mask': ...} dicts)
    id_like = {k: v for k, v in item.items()
               if isinstance(k, int) or (isinstance(k, str) and k.isdigit())}
    if id_like:
        for oid, v in id_like.items():
            if isinstance(v, dict) and 'counts' not in v:
                v = v.get('mask', v.get('masks', v.get('rle',
                          v.get('segmentation'))))
            mb = _to_bool_mask(v, hw)
            if mb is not None:
                pairs.append((int(oid), mb))
        return fi, pairs
    # dict had content but nothing matched: signal "unparsed" so the caller
    # warns and dumps structure instead of silently reporting 0 detections
    if any(k not in ('frame_index', 'frame_idx', 'frame') for k in item):
        return None
    return fi, pairs


def propagate(predictor, session_id, direction='both', evict=True):
    """Yield raw per-frame items ONE AT A TIME (never materialised as a list).

    Official API (facebookresearch/sam3, sam3_base_predictor.py):
        predictor.handle_stream_request({'type': 'propagate_in_video',
            'session_id': sid, 'propagation_direction': ...,
            'evict_cached_frame_outputs': True})
        -> yields {'frame_index': i, 'outputs': {'out_obj_ids', 'out_binary_masks', ...}}
    Older request names are tried only if that raises before yielding
    anything; the non-streaming handle_request fallback is last and warns,
    because it returns every frame at once (the old OOM)."""
    last_err = None
    errs = []                # every attempt, so the first (real) error is not hidden
    req_extra = dict(propagation_direction=direction,
                     evict_cached_frame_outputs=evict)
    for rtype in ('propagate_in_video', 'propagate', 'propagate_in_session'):
        req = dict(type=rtype, session_id=session_id, **req_extra)
        if hasattr(predictor, 'handle_stream_request'):
            got_any = False
            try:
                for item in predictor.handle_stream_request(request=req):
                    got_any = True
                    yield item
                # Request accepted without error: an empty stream just means no
                # objects were detected for this prompt (e.g. class not visible).
                # Trying the other request names here only masks that (2026-11-04).
                return
            except Exception as e:
                if got_any:          # failed mid-stream: do NOT restart and duplicate
                    raise
                if rtype == 'propagate_in_video':
                    import traceback; traceback.print_exc()
                errs.append('%s: %r' % (rtype, e))
                last_err = e
    for rtype in ('propagate_in_video', 'propagate', 'propagate_in_session'):
        try:
            resp = predictor.handle_request(request=dict(type=rtype, session_id=session_id))
        except Exception as e:
            errs.append('%s (non-stream): %r' % (rtype, e))
            last_err = e
            continue
        print('WARNING: streaming propagation unavailable; falling back to the '
              'non-streaming request, which holds ALL frames in memory')
        outs = resp.get('outputs', resp) if isinstance(resp, dict) else resp
        for item in ([outs] if isinstance(outs, dict) else list(outs or [])):
            yield item
        return
    raise RuntimeError(
        'Could not propagate — the sam3 session API request type differs in '
        'your installed version (errors: %s). See examples/'
        'sam3_video_predictor_example.ipynb in the sam3 repo.' % ('; '.join(errs),))


# --------------------------------------------------------------------------- #
def run_video(vid, args, anno, all_classes, predictor):
    frames = C.frames_of(args.data_root, vid)
    if args.max_frames:
        frames = frames[:args.max_frames]
    first = 0
    if args.frame_range:                       # masks.txt keeps whole-video frame indices
        first, last = (int(x) for x in args.frame_range.split(':'))
        frames = frames[first:last]
    if not frames:
        print(f'[{vid}] no frames — skipping'); return
    im0 = cv2.imread(frames[0]); hw = im0.shape[:2]

    out_dir = os.path.join(args.out, vid)
    os.makedirs(out_dir, exist_ok=True)
    jpg_dir = bmp_to_jpg_folder(frames, os.path.join(out_dir, '_jpg'))

    if args.class_list:   # [(class, prompt text)], e.g. person or ground:road
        prompts = [tuple(e.split(':', 1)) if ':' in e else (e, e) for e in args.class_list.split(',')]
        vocab = list(dict.fromkeys(c for c, _ in prompts))
    else:
        vocab = (list(all_classes) if args.classes == 'all'
                 else C.get_video_vocab(anno, vid, all_classes))
        prompts = [(c, c) for c in vocab]
    with open(os.path.join(out_dir, 'vocab.json'), 'w') as f:
        json.dump({'vocab': vocab, 'classes': all_classes, 'prompts': prompts}, f, indent=1)

    rows, next_tid, idmap = [], 1, {}
    per_frame_viz = {}
    failed_classes = []
    for ci, (cname, ptext) in enumerate(prompts):
        gcid = all_classes.index(cname)
        t0 = time.time()
        resp = predictor.handle_request(
            request=dict(type='start_session', resource_path=jpg_dir,
                         offload_video_to_cpu=args.offload_video_cpu,
                         offload_state_to_cpu=args.offload_state_cpu))
        sid = resp['session_id'] if isinstance(resp, dict) else resp
        r2 = predictor.handle_request(
            request=dict(type='add_prompt', session_id=sid,
                         frame_index=args.prompt_frame, text=ptext))
        # the add_prompt response already carries the prompted frame's output
        n_frames_done, unparsed_warned = 0, False
        seen = set()

        def consume(item):
            """parse -> RLE-encode -> keep only strings. Masks die here."""
            nonlocal n_frames_done, unparsed_warned, next_tid
            parsed = parse_frame_output(item, hw)
            if parsed is None:
                if not unparsed_warned:
                    print(f'[{vid}/{cname}] WARNING: unparsed item — full '
                          f'structure below (paste this back to fix the '
                          f'parser in one line):')
                    _describe(item)
                    unparsed_warned = True
                return
            fi, pairs = parsed
            if fi in seen:
                return
            seen.add(fi)
            n_frames_done += 1
            for oid, mask in pairs:
                if mask is None or mask.sum() < args.min_area:
                    continue
                key = (ci, int(oid))
                if key not in idmap:
                    idmap[key] = next_tid; next_tid += 1
                tid = idmap[key]
                h, w, counts = C.encode_mask(mask)
                rows.append((fi, tid, gcid, h, w, counts))
                if args.viz_every and fi % args.viz_every == 0:
                    per_frame_viz.setdefault(fi, []).append((tid, cname, h, w, counts))

        if parse_frame_output(r2, hw) is not None:   # prompted frame, if returned
            consume(r2)
        direction = 'forward' if args.prompt_frame == 0 else 'both'
        try:
            for item in propagate(predictor, sid, direction, evict=True):
                consume(item)
                del item                               # nothing else holds masks
        except Exception as e:
            # One class prompt failing (e.g. SAM3's KeyError('maskmem_features') in
            # tracker.remove_object, seen on 150625 "dirt pile", 2026-11-04) must not
            # discard the other classes of the video: drop this class's partial
            # tracks and go on.
            bad = {idmap.pop(k) for k in [k for k in idmap if k[0] == ci]}
            rows[:] = [r for r in rows if r[1] not in bad]
            for fi in per_frame_viz:
                per_frame_viz[fi] = [v for v in per_frame_viz[fi] if v[0] not in bad]
            failed_classes.append(cname)
            print(f'[{vid}/{cname}] CLASS FAILED, skipped ({len(bad)} partial '
                  f'tracks dropped): {e}', flush=True)
            n_frames_done = 0
        if n_frames_done == 0:
            print(f'[{vid}/{cname}] WARNING: propagation yielded no frames')
        try:
            predictor.handle_request(
                request=dict(type='close_session', session_id=sid))
        except Exception:
            pass
        try:
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
        print(f'[{vid}] "{ptext}" ({cname}): {n_frames_done} frames, '
              f'{len([k for k in idmap if k[0] == ci])} tracks '
              f'({time.time()-t0:.0f}s)', flush=True)
        if (n_frames_done and not any(k[0] == ci for k in idmap)
                and not unparsed_warned):
            print(f'[{vid}/{cname}] 0 tracks with clean parsing = SAM3 '
                  f'genuinely found no "{cname}" above --min-area '
                  f'(fine if the class is not visible in this video)')

    if args.viz_every:
        qd = os.path.join(out_dir, 'qualitative'); os.makedirs(qd, exist_ok=True)
        from pycocotools import mask as mu
        for fi, items in per_frame_viz.items():
            im = cv2.imread(frames[fi])
            dec = [(mu.decode({'size': [h, w], 'counts': c.encode() if isinstance(c, str) else c}
                              ).astype(bool), tid, name)
                   for tid, name, h, w, c in items]
            cv2.imwrite(os.path.join(qd, f'{fi:06d}.jpg'), C.overlay_masks(im, dec))

    rows = [(r[0] + first,) + tuple(r[1:]) for r in rows]
    C.write_masks_txt(os.path.join(out_dir, 'masks.txt'), rows)
    if failed_classes:
        with open(os.path.join(out_dir, 'vocab.json'), 'w') as f:
            json.dump({'vocab': vocab, 'classes': all_classes, 'prompts': prompts,
                       'failed_classes': failed_classes}, f, indent=1)
    print(f'[{vid}] DONE — {len(rows)} mask rows, {next_tid-1} tracks '
          f'-> {out_dir}/masks.txt')
    if not args.keep_jpg:
        shutil.rmtree(jpg_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', default='work_dirs/frontend_sam3')
    ap.add_argument('--videos', nargs='*', default=C.DEFAULT_VIDEOS)
    ap.add_argument('--classes', choices=['gt', 'all'], default='gt')
    ap.add_argument('--class-list', default='',
                    help='comma list overriding --classes; CLASS or CLASS:PROMPT (prompt SAM3 '
                         'with another text, tracks stored as CLASS), e.g. ground:road,ground:floor')
    ap.add_argument('--prompt-frame', type=int, default=0,
                    help='frame index to attach the text prompt at')
    ap.add_argument('--frame-range', default='',
                    help='A:B runs only frames A..B-1, for videos too long to track at once')
    ap.add_argument('--min-area', type=int, default=64)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--viz-every', type=int, default=50)
    ap.add_argument('--offload-video-cpu', action='store_true',
                    help='SAM3 start_session offload_video_to_cpu (less GPU RAM)')
    ap.add_argument('--offload-state-cpu', action='store_true',
                    help='SAM3 start_session offload_state_to_cpu (less GPU RAM)')
    ap.add_argument('--max-objects', type=int, default=0,
                    help='cap on concurrently tracked objects per class (0 = SAM3 default, '
                         'unlimited); crowded classes otherwise grow until the GPU spills')
    ap.add_argument('--no-trim-memory', action='store_true',
                    help='keep every past frame\'s tracker output on the GPU (SAM3 default). '
                         'Without trimming GPU memory grows with frames x objects and long '
                         'videos (fin_c_ext, mic_4_int) pass 47 GB')
    ap.add_argument('--keep-jpg', action='store_true',
                    help='keep the converted JPEG folder (reruns are faster)')
    args = ap.parse_args()
    args.data_root = os.path.expanduser(args.data_root)

    anno = C.load_anno(args.data_root)
    _, _, all_classes = C.get_class_lists(anno)
    print(f'{len(all_classes)} classes: {all_classes}')

    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor()   # downloads ckpt from HF on 1st run
    if args.max_objects:
        predictor.model.max_num_objects = args.max_objects
        print(f'max_num_objects = {args.max_objects}')
    if not args.no_trim_memory:
        # SAM3 builds its tracker with trim_past_non_cond_mem_for_eval=False, so every frame's
        # full-res masks + memory features stay on the GPU for the whole video. Trimming keeps
        # only low-res masks / obj pointers for frames outside the attention window
        # (>20*max_obj_ptrs frames back with memory selection), so outputs are unchanged.
        predictor.model.tracker.trim_past_non_cond_mem_for_eval = True
        print('tracker memory trimming on')

    for vid in args.videos:
        try:
            run_video(vid, args, anno, all_classes, predictor)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[{vid}] FAILED: {e} — continuing', file=sys.stderr)


if __name__ == '__main__':
    main()
