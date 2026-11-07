"""
common.py — shared utilities for the PSG4D frontend bake-off.

The contract (identical for every frontend, same as the existing baseline):
  <out>/<video>/masks.txt      one line per (frame, track) :
                               "frame track_id class_id h w rle_counts"
                               rle_counts = COCO compressed RLE counts string
  <out>/<video>/feats.pickle   written by extract_dino_feats.py:
                               {'feats': {tid: [T, D] float16},
                                'present': {tid: [T] bool},
                                'classes': {tid: class_id},
                                'class_names': [...], 'dim': D, 'video': vid}

Ground truth:
  <data_root>/images/<vid>/*.bmp
  <data_root>/visible/<vid>/*.npy   int image, pixel value = GT object id (0 = void,
                                    >1000 = per-video stuff ids after merge_masks)
  <data_root>/sailvos3d.json        class lists, per-video object id -> category,
                                    splits, relations
"""
import glob
import json
import os

import numpy as np
from pycocotools import mask as maskUtils

# The 13 val videos with relation annotations (your extracted set).
DEFAULT_VIDEOS = [
    'fam_1_mcs_2', 'fin_c_ext', 'maude_mcs_2', 'mic_4_int',
    'family_2_mcs_2', 'fam_6_mcs_1', 'family_4_mcs_3_concat', 'maude_mcs_4',
    'maude_mcs_3', 'maude_mcs_5', 'fam_1_ext_alt2', 'sol_1_mcs_1_concat',
    'fbi_2_mcs_2',
]


# --------------------------------------------------------------------------- #
# annotation json                                                             #
# --------------------------------------------------------------------------- #
def load_anno(data_root):
    path = os.path.join(data_root, 'sailvos3d.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'sailvos3d.json not found at {path}')
    with open(path) as f:
        return json.load(f)


def get_class_lists(anno):
    """Return (things, stuffs, all_classes). Tries the known layouts of
    sailvos3d.json / pvsg-style jsons; falls back to scanning per-video
    object categories."""
    obj = anno.get('objects')
    if isinstance(obj, dict):
        things = obj.get('thing') or obj.get('things') or []
        stuffs = obj.get('stuff') or obj.get('stuffs') or []
        if things or stuffs:
            return list(things), list(stuffs), list(things) + list(stuffs)
    things = anno.get('thing_classes') or []
    stuffs = anno.get('stuff_classes') or []
    if things or stuffs:
        return list(things), list(stuffs), list(things) + list(stuffs)
    if isinstance(anno.get('classes'), list):
        c = list(anno['classes'])
        return c, [], c
    # last resort: scan every video's object list
    seen = []
    for entry in _iter_video_entries(anno):
        for o in entry.get('objects') or entry.get('object') or []:
            c = _obj_category(o)
            if c and c not in seen:
                seen.append(c)
    if not seen:
        raise KeyError(
            'Could not find class lists in sailvos3d.json — inspect its keys: '
            + str(list(anno.keys())))
    return seen, [], seen


def _iter_video_entries(anno):
    data = anno.get('data', anno)
    if isinstance(data, dict):
        for vid, entry in data.items():
            if isinstance(entry, dict):
                e = dict(entry)
                e.setdefault('video_id', vid)
                yield e
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                yield entry


def _obj_category(o):
    if isinstance(o, dict):
        return o.get('category') or o.get('class') or o.get('name')
    return None


def _obj_id(o):
    if isinstance(o, dict):
        for k in ('object_id', 'id', 'oid'):
            if k in o:
                return int(o[k])
    return None


def get_video_entry(anno, vid):
    for entry in _iter_video_entries(anno):
        if entry.get('video_id') == vid or entry.get('vid') == vid:
            return entry
    return None


def get_video_objects(anno, vid):
    """{gt_object_id (int) -> category (str)} for one video. Includes stuff
    ids (>1000) if background masks were merged and the json lists them."""
    entry = get_video_entry(anno, vid)
    out = {}
    if entry:
        # sailvos3d.json uses 'object'; pvsg-style jsons use 'objects'
        for o in entry.get('objects') or entry.get('object') or []:
            oid, cat = _obj_id(o), _obj_category(o)
            if oid is not None and cat:
                out[oid] = cat
    return out


def get_video_vocab(anno, vid, all_classes):
    """Class names to prompt with for this video: the GT categories present,
    ordered as in the global class list (fair closed-vocab setting)."""
    cats = set(get_video_objects(anno, vid).values())
    vocab = [c for c in all_classes if c in cats]
    return vocab if vocab else list(all_classes)


# --------------------------------------------------------------------------- #
# frames / GT masks                                                           #
# --------------------------------------------------------------------------- #
def frames_of(data_root, vid):
    fr = sorted(glob.glob(os.path.join(data_root, 'images', vid, '*.bmp')))
    if not fr:  # tolerate other extensions
        fr = sorted(glob.glob(os.path.join(data_root, 'images', vid, '*.*')))
    return fr


def gt_files_of(data_root, vid):
    return sorted(glob.glob(os.path.join(data_root, 'visible', vid, '*.npy')))


# --------------------------------------------------------------------------- #
# RLE / masks.txt                                                             #
# --------------------------------------------------------------------------- #
def encode_mask(mask_bool):
    """bool HxW -> (h, w, counts_str)"""
    rle = maskUtils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    return rle['size'][0], rle['size'][1], rle['counts'].decode('ascii')


def decode_rle(h, w, counts):
    return maskUtils.decode(
        {'size': [int(h), int(w)], 'counts': counts.encode('ascii')}
    ).astype(bool)


def write_masks_txt(path, rows):
    """rows: iterable of (frame, tid, cid, h, w, counts_str)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for fr, tid, cid, h, w, counts in rows:
            f.write(f'{int(fr)} {int(tid)} {int(cid)} {int(h)} {int(w)} {counts}\n')


def find_masks_txt(pred_root, vid):
    for rel in (os.path.join(vid, 'masks.txt'),
                os.path.join(vid, 'quantitive', 'masks.txt'),
                os.path.join(vid, 'quantitative', 'masks.txt')):
        p = os.path.join(pred_root, rel)
        if os.path.exists(p):
            return p
    return None


def read_masks_txt(path, normalize_t0=True):
    """-> per_frame: {frame -> [(tid, cid, rle_dict), ...]} .
    Frames are shifted so the earliest frame becomes 0 (matches how the
    baseline's absolute frame numbers line up with sorted GT files)."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.strip().split(' ')
            if len(p) < 6:
                continue
            rows.append((int(p[0]), int(p[1]), int(p[2]),
                         {'size': [int(p[3]), int(p[4])],
                          'counts': p[5].encode('ascii')}))
    if not rows:
        return {}
    t0 = min(r[0] for r in rows) if normalize_t0 else 0
    per_frame = {}
    for fr, tid, cid, rle in rows:
        per_frame.setdefault(fr - t0, []).append((tid, cid, rle))
    return per_frame


# --------------------------------------------------------------------------- #
# small viz helpers (used by runners' --viz-every and viz_compare.py)         #
# --------------------------------------------------------------------------- #
def color_for_id(tid):
    rng = np.random.RandomState(int(tid) * 9973 % (2 ** 31))
    c = rng.randint(60, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])   # BGR


def overlay_masks(im_bgr, items, alpha=0.55, draw_label=True):
    """items: list of (mask_bool, tid, label_str). Returns a new BGR image."""
    import cv2
    out = im_bgr.copy()
    for mask, tid, label in items:
        if mask is None or not mask.any():
            continue
        color = np.array(color_for_id(tid), dtype=np.float32)
        m = mask.astype(bool)
        out[m] = (out[m].astype(np.float32) * (1 - alpha) + color * alpha
                  ).astype(np.uint8)
        if draw_label:
            ys, xs = np.nonzero(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(out, f'{label}#{tid}', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(out, f'{label}#{tid}', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color_for_id(tid), 1, cv2.LINE_AA)
    return out
