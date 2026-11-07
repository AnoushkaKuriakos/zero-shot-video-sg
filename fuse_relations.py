#!/usr/bin/env python3
"""
Step 7 — fuse window-level triplets into video-level relations with
intervals.

For each (subject, object, predicate):
  frames   = union over supporting windows of the frames in the window span
             where BOTH objects are visible (from the SAM3 masks)
  bridge   a gap of <= --max-gap frames is filled only if the pair stays
             co-visible through it
  votes    = number of DISTINCT windows that asserted it (the old code
             counted rows, so caption + paraphrase double-counted)
  score    = votes, ties broken by total frames — used for Recall@K ranking
Relations shorter than --min-len frames in total are dropped.
--plausible also drops commonsense violations (plausible.py), logged to
plausibility_log_<path>.json.

Output: OUT/<vid>/relations_<path>.json
  {'video', 'path', 'predicates', 'relations': [{subject, object, predicate,
   predicate_id, intervals: [[t1,t2],...], votes, windows, n_frames}]}

    python fuse_relations.py --data-root $DATA --videos fam_1_mcs_2 --path som
"""
import argparse
import json
from collections import defaultdict

import plausible as PL
import sgcore as S


def intervals(frames):
    fs = sorted(frames)
    if not fs:
        return []
    out, a, b = [], fs[0], fs[0]
    for f in fs[1:]:
        if f == b + 1:
            b = f
        else:
            out.append([a, b]); a = b = f
    out.append([a, b])
    return out


def fuse(spans, covis, max_gap):
    frames = set()
    for t1, t2 in spans:
        frames |= {f for f in range(t1, t2 + 1) if f in covis}
    ivs = intervals(frames)
    merged = []
    for iv in ivs:
        if merged:
            gap = range(merged[-1][1] + 1, iv[0])
            if len(gap) <= max_gap and all(f in covis for f in gap):
                merged[-1][1] = iv[1]; continue
        merged.append(list(iv))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--path', required=True, choices=S.PATHS)
    ap.add_argument('--max-gap', type=int, default=15)
    ap.add_argument('--min-len', type=int, default=5)
    ap.add_argument('--plausible', action='store_true', help='apply the plausible.py filter')
    ap.add_argument('--rules', default='', help='JSON from tools/gen_rules.py: drop relations whose subject or '
                    'object class is not listed for the predicate (LLM-written commonsense rules)')
    ap.add_argument('--score', choices=['votes', 'q'], default='votes',
                    help='q: rank by the sum of label-model posteriors over windows (tools/label_model.py)')
    args = ap.parse_args()
    predicates = S.predicate_list(args.data_root)
    RULES = json.load(open(args.rules))['rules'] if args.rules else None
    if RULES:
        args.plausible = True          # same logging path
        def rule_check(sc, p, oc):
            r = RULES.get(p)
            if r is None or sc is None or oc is None or 'others' in (sc, oc):
                return None
            if sc not in r['subjects']:
                return f'LLM rule: {sc} cannot be the subject of {p}'
            if oc not in r['objects']:
                return f'LLM rule: {oc} cannot be the object of {p}'
            return None
    for vid in args.videos:
        tubes = S.Tubes(S.vdir(vid, 'masks.txt'))
        rows = S.io_jsonl(S.vdir(vid, f'triplets_aligned_{args.path}.jsonl'))
        names = S.class_names(vid, args.data_root)
        cname = lambda t: names[tubes.cls[t]] if 0 <= tubes.cls.get(t, -1) < len(names) else None
        groups = defaultdict(dict)                     # key -> {window_id: span}
        qs = defaultdict(dict)                         # key -> {window_id: q}
        dropped = {}
        for r in rows:
            if args.plausible:
                sc, oc = cname(r['subject']), cname(r['object'])
                why = rule_check(sc, r['predicate'], oc) if RULES else PL.plausible(sc, r['predicate'], oc)
                if why:
                    dropped[f"{sc}#{r['subject']} {r['predicate']} {oc}#{r['object']}"] = why
                    continue
            groups[(r['subject'], r['object'], r['predicate'])][r['window_id']] = r['span']
            if 'q' in r:
                key = (r['subject'], r['object'], r['predicate'])
                qs[key][r['window_id']] = max(qs[key].get(r['window_id'], 0.0), r['q'])
        rels = []
        for (s, o, p), wins in groups.items():
            ivs = fuse(wins.values(), tubes.covis(s, o), args.max_gap)
            n = sum(b - a + 1 for a, b in ivs)
            if n < args.min_len:
                continue
            rel = {'subject': s, 'object': o, 'predicate': p,
                   'predicate_id': predicates.index(p), 'intervals': ivs,
                   'votes': len(wins), 'windows': sorted(wins), 'n_frames': n}
            if args.score == 'q':
                rel['score'] = round(sum(qs[(s, o, p)].values()), 6)   # expected supporting windows
            rels.append(rel)
        if args.score == 'q':
            rels.sort(key=lambda r: (-r['score'], -r['n_frames']))
        else:
            rels.sort(key=lambda r: (-r['votes'], -r['n_frames']))
        out = {'video': vid, 'path': args.path, 'predicates': predicates, 'relations': rels}
        json.dump(out, open(S.vdir(vid, f'relations_{args.path}.json'), 'w'), indent=1)
        if args.plausible:
            json.dump(dropped, open(S.vdir(vid, f'plausibility_log_{args.path}.json'), 'w'), indent=1)
        print(f'[{vid}] {args.path}: {len(rels)} relations from {len(rows)} window triplets'
              + (f', {len(dropped)} implausible dropped' if args.plausible else ''))


if __name__ == '__main__':
    main()
