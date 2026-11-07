"""
sgcore.py — shared plumbing for the window-level (NON-pairwise) scene-graph
pipeline. Every other script imports this.

What lives here
  * paths:     OUT/<vid>/...   (OUT = $SG_OUT or <this folder>/out)
  * dataset:   SAIL-VOS access goes through the LOCAL common.py /
               relcommon.py (they know sailvos3d.json). This folder is
               self-contained: nothing is imported from bakeoff/
  * masks:     masks.txt reader (raw frame indices, no t0 shift) + a Tubes
               index whose per-frame area/box come straight from the RLE —
               no decoding needed for statistics
  * windows:   time windows over the whole video; ALL objects visible in a
               window go into ONE model call (no pairs anywhere)
  * llm:       text/VLM backend: --backend fake (built-in, scripted) or
               qwen (local vlm_backends.py, Qwen2.5-VL)
"""
import json
import os
import re
import sys

import numpy as np
from pycocotools import mask as maskUtils

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)       # common / relcommon / vlm_backends from HERE only
OUT = os.path.expanduser(os.environ.get('SG_OUT', os.path.join(HERE, 'out')))

PATHS = ('som', 'ref')                  # the two captioning paths
PATH_NAME = {'som': 'Set-of-Mark (Qwen-VL)', 'ref': 'PixelRefer'}


# ------------------------------------------------------------------ dataset
def C():
    import common
    return common


def R():
    import relcommon
    return relcommon


def vdir(vid, *parts, make=True):
    p = os.path.join(OUT, vid, *parts)
    if make:
        os.makedirs(p if not os.path.splitext(p)[1] else os.path.dirname(p),
                    exist_ok=True)
    return p


def frames_of(data_root, vid):
    return C().frames_of(os.path.expanduser(data_root), vid)


def class_names(vid=None, data_root=None):
    """Global class list. Prefer the vocab.json SAM3 wrote next to masks.txt
    (its 'classes' is exactly the list class ids index into)."""
    if vid:
        vp = os.path.join(OUT, vid, 'vocab.json')
        if os.path.exists(vp):
            v = json.load(open(vp))
            if v.get('classes'):
                return list(v['classes'])
    _, _, names = C().get_class_lists(C().load_anno(os.path.expanduser(data_root)))
    return list(names)


def stuff_names(data_root):
    _, stuffs, _ = C().get_class_lists(C().load_anno(os.path.expanduser(data_root)))
    return set(stuffs)


def predicate_list(data_root):
    return list(R().get_predicate_list(C().load_anno(os.path.expanduser(data_root))))


# -------------------------------------------------------------------- masks
def read_masks(path):
    """masks.txt: 'frame tid cid h w rle' per line -> {frame: [(tid,cid,rle)]}.
    Frame indices are kept exactly as written (no normalisation), so they
    index the image list directly."""
    per = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) != 6 or not p[0].lstrip('-').isdigit():
                continue                              # header / blank
            f, t, c, h, w = (int(x) for x in p[:5])
            per.setdefault(f, []).append(
                (t, c, {'size': [h, w], 'counts': p[5].encode()}))
    return per


def encode(mask):
    r = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return r['size'][0], r['size'][1], r['counts'].decode('ascii')


def write_masks(path, rows):
    """rows: (frame, tid, cid, h, w, counts_str). Sorted, atomic."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        for r in sorted(rows, key=lambda r: (r[0], r[1])):
            fh.write(' '.join(str(x) for x in r) + '\n')
    os.replace(tmp, path)


class Tubes:
    """All tracks of one video. Stats from RLE (area, bbox) without decoding."""

    def __init__(self, masks_path):
        self.per_frame = read_masks(masks_path)
        self.frames_of = {}                 # tid -> {f: (area, (x,y,w,h))}
        self.cls = {}
        for f, items in self.per_frame.items():
            for tid, cid, rle in items:
                a = int(maskUtils.area(rle))
                if a == 0:
                    continue
                bb = tuple(float(x) for x in maskUtils.toBbox(rle))
                self.frames_of.setdefault(tid, {})[f] = (a, bb)
                self.cls[tid] = cid
        self.hw = None
        for items in self.per_frame.values():
            if items:
                self.hw = tuple(items[0][2]['size']); break

    def tids(self):
        return sorted(self.frames_of)

    def visible(self, tid, f):
        return f in self.frames_of.get(tid, {})

    def area(self, tid, f):
        return self.frames_of.get(tid, {}).get(f, (0, None))[0]

    def centroid(self, tid, f):
        a = self.frames_of.get(tid, {}).get(f)
        if not a:
            return None
        x, y, w, h = a[1]
        return x + w / 2, y + h / 2

    def mask(self, tid, f):
        for t, _, rle in self.per_frame.get(f, []):
            if t == tid:
                return maskUtils.decode(rle).astype(bool)
        return None

    def covis(self, a, b):
        fa, fb = self.frames_of.get(a, {}), self.frames_of.get(b, {})
        return set(fa) & set(fb)


# ------------------------------------------------------------------ windows
def uniform(seq, k):
    seq = list(seq)
    if len(seq) <= k:
        return seq
    idx = np.linspace(0, len(seq) - 1, k).round().astype(int)
    return [seq[i] for i in sorted(set(idx))]


def load_windows(vid):
    p = os.path.join(OUT, vid, 'windows.json')
    if not os.path.exists(p):
        raise SystemExit(f'[{vid}] no windows.json — run make_windows.py first')
    return json.load(open(p))


def load_dossiers(vid):
    p = os.path.join(OUT, vid, 'dossiers.json')
    return {int(k): v for k, v in json.load(open(p)).items()} if os.path.exists(p) else {}


# ---------------------------------------------------------- shared prompt
# The SAME instruction is given to both captioners, so any difference in the
# graphs comes from how objects are referred to (painted marks vs region
# tokens), not from what they were asked.
CAPTION_INSTRUCTION = (
    'Describe, in {n_min}-{n_max} short plain sentences, what these objects '
    'are doing and how they interact with or relate to each other over the '
    'clip: actions, contact, relative motion and spatial relations. Write one '
    'fact per sentence. Refer to every object ONLY as {ref_style} (for example '
    '"{ex_a} is holding {ex_b}"), never by a description alone. Prefer '
    'specific verbs (holding, sitting on, walking toward, talking to, looking '
    'at, next to) over vague ones. Only state what you can see; if two '
    'objects do not interact, do not invent a relation between them.')


# Prompt versions (--prompt). v1 is the original instruction above; the
# README "Prompt construction" section explains each change and its effect.
# Placeholders: {ref_style} {ex_a} {ex_b} {vocab} (+ {n_min} {n_max} in v1).
CAPTION_PROMPTS = {
    'v1': CAPTION_INSTRUCTION,
    # v1 + the closed SAIL-VOS predicate list as preferred wording
    'v2': CAPTION_INSTRUCTION + (
        ' Whenever one of these relation words fits, use it exactly as written: '
        '{vocab}.'),
    # structured: one relation per sentence, closed vocabulary, explicit asks
    # for the relation types the captioners kept omitting (conversation,
    # persistent states, state changes), and no unlisted entities
    'v3': (
        'Describe how these objects relate to each other during the clip, in '
        '4-12 short sentences. Follow these rules:\n'
        '1. One relation per sentence, written as "<subject> <relation> '
        '<object>", for example "{ex_a} is talking to {ex_b}".\n'
        '2. Refer to every object ONLY as {ref_style}. Both the subject and the '
        'object must be one of the listed objects; never mention anything that '
        'is not listed (no "the road", "the background", "the seat").\n'
        '3. Use these relation words whenever one fits, exactly as written: '
        '{vocab}.\n'
        '4. Conversations: when people face each other, speak or gesture, say '
        'who is talking to whom; if both speak, write both directions '
        '("{ex_a} is talking to {ex_b}. {ex_b} is talking to {ex_a}.").\n'
        '5. Include relations that hold for the whole clip (someone in a '
        'vehicle, standing or sitting on something, holding something), not '
        'only changes.\n'
        '6. If a relation starts during the clip (gets into a car, picks '
        'something up), use the matching relation word, e.g. "entering".\n'
        '7. Only state what you can see; do not invent relations.'),
    # v2 + the two v3 failure fixes for Qwen: a concrete conversation cue
    # (it wrote "looking at" for every conversation) and a guard against
    # enumerating every mark pair ("mark 1 is in mark 7, mark 3 is in ...")
    'v4': CAPTION_INSTRUCTION + (
        ' Whenever one of these relation words fits, use it exactly as written: '
        '{vocab}. When two people face each other, or one speaks or gestures to '
        'the other, write "{ex_a} is talking to {ex_b}" rather than "looking '
        'at". Never repeat a sentence, and do not list the same relation for '
        'every object: each sentence must state a different, visible fact.'),
}
# v5 = v3 + motion facts measured from the masks (motion.py; README "Motion cues (prompt v5)")
CAPTION_PROMPTS['v5'] = CAPTION_PROMPTS['v3'][:-len('7. Only state what you can see; do not invent relations.')] + (
    '7. Use the motion facts listed above, which were measured from the masks, to choose between relations such '
    'as approaching, leaving, following, next to, touching and on; do not contradict them.\n'
    '8. Only state what you can see; do not invent relations.')


def caption_instruction(version, predicates, **kw):
    return CAPTION_PROMPTS[version].format(vocab=', '.join(predicates), **kw)


def add_prompt_arg(ap):
    ap.add_argument('--prompt', default='v1', choices=sorted(CAPTION_PROMPTS),
                    help='caption instruction version (see CAPTION_PROMPTS)')


def legend(objs, dossiers, names, ref):
    """ref(tid, k) -> how the object is named in this path's prompt."""
    lines = []
    for k, tid in enumerate(objs):
        d = dossiers.get(tid, {})
        cname = d.get('class_name') or (names[d['class_id']] if 'class_id' in d else 'object')
        short = d.get('short')
        lines.append(f'{ref(tid, k)} is a {cname}' + (f': {short}' if short else ''))
    return '\n'.join(lines)


def io_jsonl(path, rows=None):
    if rows is None:
        return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    os.replace(tmp, path)


def caption_matches(row, w, version='v1'):
    """Resume guard: a saved caption is reused only if it was made for the
    same window (window ids shift when make_windows.py is rerun with other
    --win/--stride/--max-objects) with the same prompt version."""
    return (bool(row) and row.get('span') == w['span'] and row.get('objects') == w['objects']
            and row.get('prompt_version', 'v1') == version)


MARK_RE = re.compile(r'\bmark\s*#?\s*(\d+)\b', re.I)


def mark_id(entity):
    m = MARK_RE.search(str(entity))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- backends
try:
    import vlm_backends as VB
except Exception as _vb_e:           # e.g. PixelRefer env without Qwen deps
    print(f'[sgcore] vlm_backends not importable ({_vb_e!r}); only --backend fake',
          file=sys.stderr)
    VB = None


def add_llm_args(ap):
    if VB is not None:
        VB.add_backend_args(ap)
        for a in ap._actions:            # make sure 'fake' is always allowed
            if '--backend' in a.option_strings and a.choices is not None \
                    and 'fake' not in a.choices:
                a.choices = list(a.choices) + ['fake']
    else:
        ap.add_argument('--backend', default='fake', choices=['fake'],
                        help='vlm_backends.py not importable: fake only')


def make_llm(args):
    if args.backend == 'fake':
        return FakeLLM()
    return VB.backend_from_args(args)


class FakeLLM:
    """Deterministic stand-in that answers each prompt type in the format the
    real model is asked for, so every parser is exercised end to end."""

    def chat(self, prompt, images=None, system=None, max_tokens=256, temperature=0.0):
        ids = [int(x) for x in re.findall(r'mark (\d+) is a', prompt)]
        if 'predefined predicate lexicon' in prompt:          # Chain-2
            lex = dict(re.findall(r'(\d+)\.([^\d\n]+?)(?= \d+\.| *\n)', prompt))
            q = re.findall(r"Given the lexeme '([^']*)', find", prompt.split('Please answer')[-1])
            out = []
            for lx in q:
                hit = next((f'{n}.{p.strip()}' for n, p in lex.items()
                            if p.strip() and p.strip().split()[0] in lx), '0.None')
                out.append(f'Question: Given the lexeme \'{lx}\', find semantically relevant '
                           f'lexeme in the predefined predicate lexicon. Answer: {hit}')
            return '\n'.join(out)
        if 'extract meaningful triplets' in prompt:          # Chain-1
            sents = re.findall(r"Given the sentence '([^']*)', extract", prompt.split('Please answer')[-1])
            para = 'Step 1' in prompt
            out = []
            for s in sents:
                trip = []
                m = re.search(r'(mark \d+) (?:is )?(\w+(?: \w+)?) (mark \d+)', s, re.I)
                if m:
                    trip = [f'<{m.group(1)}, {m.group(2)}, {m.group(3)}>']
                t = ', '.join(trip) if trip else ''
                if para:
                    out.append(f"Question: Given the sentence '{s}', extract meaningful triplets. Answer: "
                               f"Step 1: The sentence can be paraphrased as: '{s}'. Step 2: Meaningful "
                               f"triplets, where the subject and object are the simple noun, extracted "
                               f"from the paraphrased sentence are: {t}. The meaningful triplets are {t}.")
                else:
                    out.append(f"Question: Given the sentence '{s}', extract meaningful triplets. "
                               f"Answer: Meaningful triplets are {t}.")
            return '\n'.join(out)
        # captioning prompt (SoM path): mention the first marks
        if len(ids) >= 2:
            a, b = ids[0], ids[1]
            s = f'Mark {a} is holding mark {b}. Mark {b} is next to mark {a}.'
            if len(ids) >= 3:
                s += f' Mark {ids[2]} is looking at mark {a}.'
            return s
        return 'Nothing happens between the marked objects.'
