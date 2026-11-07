#!/usr/bin/env python3
"""
Turn your own videos into a data root the pipeline can read:

    <out>/images/<video_id>/000000.jpg ...     frames at --fps
    <out>/sailvos3d.json                       class lists, predicate list, one entry per video

The only domain-specific input is the config: object classes (things = countable objects, stuff =
background), optional SAM3 prompt text per class, and the predicate list. See configs/*.json.

    python tools/make_data_root.py --config configs/construction.json --out data/mysite \
        --videos site_cam1.mp4 site_cam2.mp4 --fps 5 --max-frames 600
Prints the --class-list to pass to tools/run_sam3.py.
"""
import argparse, json, os
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--videos', nargs='+', required=True, help='video files (mp4, avi, ...)')
    ap.add_argument('--fps', type=float, default=5, help='frames per second kept (the report used 5)')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = whole video')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    data = []
    for path in a.videos:
        vid = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(src_fps / a.fps))
        d = os.path.join(a.out, 'images', vid); os.makedirs(d, exist_ok=True)
        i = k = 0; h = w = 0
        while True:
            ok, im = cap.read()
            if not ok or (a.max_frames and k >= a.max_frames):
                break
            if i % step == 0:
                cv2.imwrite(os.path.join(d, f'{k:06d}.jpg'), im); h, w = im.shape[:2]; k += 1
            i += 1
        data.append({'video_id': vid, 'meta': {'height': h, 'width': w, 'fps': src_fps / step, 'num_frames': k},
                     'object': [], 'relations': []})
        print(f'{vid}: {k} frames ({src_fps:.1f} fps source, every {step}th)')
    anno = {'objects': {'thing': cfg['things'], 'stuff': cfg['stuff']}, 'relations': cfg['predicates'],
            'split': {'val': [x['video_id'] for x in data]}, 'data': data, 'sam3_prompts': cfg.get('prompts', {})}
    json.dump(anno, open(os.path.join(a.out, 'sailvos3d.json'), 'w'), indent=1)
    p = cfg.get('prompts', {})
    print('--class-list "' + ','.join(f'{c}:{p[c]}' if c in p else c for c in cfg['things'] + cfg['stuff']) + '"')


if __name__ == '__main__':
    main()
