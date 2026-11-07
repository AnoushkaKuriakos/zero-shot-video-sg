#!/usr/bin/env python3
"""
Build a SAIL-VOS-shaped data root from the Earthmoving Equipment dataset
(Roberts & Golparvar-Fard 2019, Mendeley 10.17632/fyw6ps2d2j.1, CC BY 4.0), so the
unchanged pipeline (run_sam3.py, run_pipeline.sh) runs on construction video.

Only two things differ from PSG4D-GTA: the class list and the predicate vocabulary,
both in the generated sailvos3d.json. No GT relations exist ("relations": []);
the per-machine activity labels are copied to activity/<vid>/ with the frame map
so tools/construction_metrics.py can score derived activities.

    python tools/build_construction_root.py --src <earthmoving>/Data \
        --out data/construction --videos 150625 151341 140555 --step 6 --max-frames 600
"""
import argparse
import json
import os
import shutil
import zipfile

THINGS = ['person', 'excavator', 'dump_truck', 'wheel_loader', 'bulldozer', 'car']
STUFF = ['ground', 'soil_pile', 'sky']
# SAM3 prompt text per class (run_sam3.py --class-list CLASS:PROMPT,...)
PROMPTS = {'dump_truck': 'dump truck', 'wheel_loader': 'wheel loader', 'soil_pile': 'dirt pile'}
PREDICATES = ['approaching', 'behind', 'carrying', 'digging in', 'driving on', 'dumping into',
              'in front of', 'leaving', 'loading', 'moving on', 'next to', 'on', 'parked on',
              'pouring onto', 'standing on', 'swinging toward', 'waiting for', 'walking on',
              'watching', 'working on']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--step', type=int, default=6, help='keep every Nth source frame')
    ap.add_argument('--max-frames', type=int, default=600)
    a = ap.parse_args()
    src, out = os.path.expanduser(a.src), os.path.expanduser(a.out)
    z = zipfile.ZipFile(os.path.join(src, 'video_frames.zip'))
    names = {}
    for n in z.namelist():
        if n.endswith('.jpg'):
            names.setdefault(n.split('/')[1], []).append(n)
    data = []
    for vid in a.videos:
        src_frames = sorted(names[vid])[::a.step][:a.max_frames]
        d = os.path.join(out, 'images', vid)
        os.makedirs(d, exist_ok=True)
        fmap = []
        for i, n in enumerate(src_frames):
            with open(os.path.join(d, f'{i:06d}.jpg'), 'wb') as fh:
                fh.write(z.read(n))
            fmap.append(int(n.rsplit('_I', 1)[1].split('.')[0]))
        act = os.path.join(out, 'activity', vid)
        os.makedirs(act, exist_ok=True)
        for f in os.listdir(os.path.join(src, 'Labels', vid)):
            shutil.copy(os.path.join(src, 'Labels', vid, f), act)
        json.dump({'source_frame': fmap, 'step': a.step}, open(os.path.join(act, 'frame_map.json'), 'w'))
        data.append({'video_id': vid, 'meta': {'height': 272, 'width': 480, 'fps': 30 // a.step,
                                               'num_frames': len(fmap)}, 'object': [], 'relations': []})
        print(vid, len(fmap), 'frames, source', fmap[0], '-', fmap[-1])
    anno = {'objects': {'thing': THINGS, 'stuff': STUFF}, 'relations': PREDICATES,
            'split': {'val': a.videos}, 'data': data, 'sam3_prompts': PROMPTS}
    json.dump(anno, open(os.path.join(out, 'sailvos3d.json'), 'w'), indent=1)
    print('class list for run_sam3.py --class-list:',
          ','.join(f'{c}:{PROMPTS[c]}' if c in PROMPTS else c for c in THINGS + STUFF))


if __name__ == '__main__':
    main()
