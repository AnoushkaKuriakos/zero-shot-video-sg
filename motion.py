"""
Motion cues measured from the SAM3 masks of one window, as short plain sentences for the caption / direct-VLM
prompts (prompt v5). Thresholds were fixed before any score: README "Motion cues (prompt v5)". Needs only numpy,
cv2 and pycocotools (both the bakeoff and the vref environments have them).

    facts = motion_facts(w, tubes, names, stuff, ref=lambda tid: f'mark {tid}')
"""
import cv2
import numpy as np

CAM_MAX = 0.15          # camera shift (fraction of the diagonal) above which no facts are given
MOVE = 0.03             # compensated centroid shift for "moves"
STILL = 0.01            # ... below which "is still"
SIZE = 0.30             # relative area change for "changes size"
TOUCH = 0.01            # gap for "touches"
GAP_ABS = 0.02          # minimum gap change for "gets closer" / "moves apart"
CLOSER, APART = 0.8, 1.25
MAX_FACTS = 8
MAX_TOUCH = 3
MAX_SIDE = 256


def _small(m):
    h, w = m.shape
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1.0:
        m = cv2.resize(m.astype(np.uint8), (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return m, s


def _gap(ma, mb):
    """Minimum pixel distance between two masks (0 if they overlap), in full-resolution pixels."""
    a, s = _small(ma)
    b, _ = _small(mb)
    if not a.any() or not b.any():
        return None
    if (a & b).any():
        return 0.0
    dist = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist[b].min()) / s


def motion_facts(w, tubes, names, stuff, ref):
    """w: window dict; tubes: sgcore.Tubes; names: class names; stuff: set of stuff class names;
    ref(tid) -> how the prompt names the object. Returns a list of sentences (possibly empty)."""
    if tubes.hw is None:
        return []
    h, wd = tubes.hw
    D = float(np.hypot(h, wd))
    t0, t1 = w['span']
    objs = [int(t) for t in w['objects']]
    cname = lambda t: names[tubes.cls[t]] if 0 <= tubes.cls.get(t, -1) < len(names) else ''
    is_stuff = {t: cname(t) in stuff for t in objs}
    frames = {t: sorted(f for f in tubes.frames_of.get(t, {}) if t0 <= f <= t1) for t in objs}

    def cen(t, f):
        return np.array(tubes.centroid(t, f))

    # camera shift: median shift of stuff tubes visible at both ends of their window frames
    shifts = [cen(t, frames[t][-1]) - cen(t, frames[t][0]) for t in objs
              if is_stuff[t] and len(frames[t]) >= 5]
    cam = np.median(np.array(shifts), axis=0) if shifts else np.zeros(2)
    if np.hypot(*cam) > CAM_MAX * D:
        return []

    moving, still, size = {}, set(), {}
    for t in objs:
        fs = frames[t]
        if is_stuff[t] or len(fs) < 5:
            continue
        d = np.hypot(*(cen(t, fs[-1]) - cen(t, fs[0]) - cam)) / D
        if d > MOVE:
            moving[t] = d
        elif d < STILL:
            still.add(t)
        a0, a1 = tubes.area(t, fs[0]), tubes.area(t, fs[-1])
        if a0 > 0 and abs(a1 - a0) / a0 > SIZE:
            size[t] = (a1 - a0) / a0

    touch, closer, apart = [], [], []
    masks = {}

    def mask(t, f):
        if (t, f) not in masks:
            masks[(t, f)] = tubes.mask(t, f)
        return masks[(t, f)]

    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            if is_stuff[a] and is_stuff[b]:
                continue
            common = sorted(set(frames[a]) & set(frames[b]))
            if len(common) < 3:
                continue
            sample = [common[0], common[len(common) // 2], common[-1]]
            gaps = []
            for f in sample:
                ma, mb = mask(a, f), mask(b, f)
                gaps.append(None if ma is None or mb is None else _gap(ma, mb))
            if any(g is None for g in gaps):
                continue
            if sum(g <= TOUCH * D for g in gaps) >= 2:
                touch.append((a, b))
            elif gaps[2] < CLOSER * gaps[0] and gaps[0] - gaps[2] > GAP_ABS * D:
                closer.append((a, b, gaps[0], gaps[2]))
            elif gaps[2] > APART * gaps[0] and gaps[2] - gaps[0] > GAP_ABS * D:
                apart.append((a, b, gaps[0], gaps[2]))

    facts, named = [], set()
    for a, b, g0, g2 in closer:
        if a in moving and b in still:
            facts.append(f'{ref(a)} moves toward {ref(b)} (gap {g0:.0f} px -> {g2:.0f} px)')
        elif b in moving and a in still:
            facts.append(f'{ref(b)} moves toward {ref(a)} (gap {g0:.0f} px -> {g2:.0f} px)')
        else:
            facts.append(f'{ref(a)} and {ref(b)} get closer (gap {g0:.0f} px -> {g2:.0f} px)')
        named |= {a, b}
    for a, b, g0, g2 in apart:
        if a in moving and b in still:
            facts.append(f'{ref(a)} moves away from {ref(b)} (gap {g0:.0f} px -> {g2:.0f} px)')
        elif b in moving and a in still:
            facts.append(f'{ref(b)} moves away from {ref(a)} (gap {g0:.0f} px -> {g2:.0f} px)')
        else:
            facts.append(f'{ref(a)} and {ref(b)} move apart (gap {g0:.0f} px -> {g2:.0f} px)')
        named |= {a, b}
    for t in objs:
        if t in named and t in moving:
            facts.append(f'{ref(t)} is moving')
        elif t in named and t in still:
            facts.append(f'{ref(t)} is still')
    for t, r in size.items():
        facts.append(f'{ref(t)} {"grows" if r > 0 else "shrinks"} in the image (area {r:+.0%})')
    # README amendment 17:55: touches only between things, at most MAX_TOUCH, listed last
    facts += [f'{ref(a)} touches {ref(b)}' for a, b in touch if not is_stuff[a] and not is_stuff[b]][:MAX_TOUCH]
    return facts[:MAX_FACTS]


def facts_block(facts):
    if not facts:
        return ''
    return 'Motion facts measured from the masks (camera motion removed):\n' + '\n'.join(f'- {f}' for f in facts)
