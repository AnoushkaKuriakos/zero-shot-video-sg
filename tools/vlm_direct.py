#!/usr/bin/env python3
"""
Single-VLM training-free baseline (README to-do 3). Qwen2.5-VL-7B (env bakeoff) sees the
Set-of-Mark keyframes of each window (som_render.py output, same windows as the pipeline),
a legend with ONLY the class name per mark (no DAM descriptions), and the numbered predicate
list, and answers with triplets directly. No captioning step, no LLM4SGG extraction or
alignment: a predicate that is not in the list, or a mark that is not in the window, is
dropped.

Writes $SG_OUT/<vid>/triplets_aligned_ref.jsonl in the pipeline's format ("ref" is only the
file slot fuse_relations.py / eval_pvsg.py read; keep this in its own experiment folder), and
the raw replies to $SG_OUT/<vid>/vlm_direct_raw.jsonl (resumable).

    SG_OUT=$PWD/exp/vlm_direct python tools/vlm_direct.py --data-root $DATA --videos ... --backend qwen
    SG_OUT=$PWD/exp/vlm_direct python fuse_relations.py --data-root $DATA --videos ... --path ref --plausible
    SG_OUT=$PWD/exp/vlm_direct python tools/eval_pvsg.py --data-root $DATA --videos ... --paths ref
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import sgcore as S

SYSTEM = ('You are a careful, concrete video annotator. You report only relations you can '
          'actually see.')
PROMPT_VERSION = 'direct-v1'


def build_prompt(w, dossiers, names, n_img, predicates, motion='', prev=''):
    objs = w['objects']
    legend = []
    for t in objs:
        d = dossiers.get(t, {})
        c = d.get('class_name') or names[w['classes'][str(t)]]
        legend.append(f'mark {t}: {c}')
    plist = '\n'.join(f'{i + 1}. {p}' for i, p in enumerate(predicates))
    return (f'You see {n_img} frames from one video clip, in temporal order. {len(objs)} objects '
            'are painted with a colored mask and a white number on a black box (the mark):\n'
            + '\n'.join(legend) + ('\n\n' + motion + '\nUse these measured facts when choosing predicates; '
                                    'do not contradict them.' if motion else '') +
            ('\n\n' + prev if prev else '') +
            '\n\nList the relations between these marked objects that hold in this clip. Use ONLY '
            'predicates from this list:\n' + plist +
            '\n\nAnswer with one relation per line, exactly in the form\n'
            'mark A | predicate | mark B\n'
            f'for example "mark {objs[0]} | {predicates[0]} | mark {objs[-1]}". Give between 1 and '
            '8 relations, most clearly visible first. Write nothing else.')


LINE = re.compile(r'mark\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*mark\s*(\d+)', re.I)


def parse(reply, objs, predicates):
    pid = {p: i for i, p in enumerate(predicates)}
    out, bad = [], 0
    for m in LINE.finditer(reply):
        s, p, o = int(m.group(1)), m.group(2).strip().lower().strip('."'), int(m.group(3))
        p = re.sub(r'^\d+\.\s*', '', p)                  # "12. next to" -> "next to"
        if s in objs and o in objs and s != o and p in pid:
            out.append((s, p, o))
        else:
            bad += 1
    return list(dict.fromkeys(out)), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--max-tokens', type=int, default=300)
    ap.add_argument('--seed', type=int, default=None, help='torch/random seed (seed-variance runs)')
    ap.add_argument('--motion', action='store_true', help='add motion facts measured from the masks (motion.py)')
    ap.add_argument('--prev-context', action='store_true',
                    help='pilot (SAMJAM idea): add the relations reported in the previous window among marks '
                         'that are also in this window (max 8 lines)')
    S.add_llm_args(ap)
    args = ap.parse_args()
    if args.seed is not None:
        import random, torch
        random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    predicates = S.predicate_list(args.data_root)
    stuff = S.stuff_names(args.data_root) if args.motion else set()
    version = PROMPT_VERSION + ('-motion' if args.motion else '') + ('-prev' if args.prev_context else '')
    llm = S.make_llm(args)
    for vid in args.videos:
        names = S.class_names(vid, args.data_root)
        dossiers = S.load_dossiers(vid)
        index = json.load(open(S.vdir(vid, 'som_index.json')))
        rpath = S.vdir(vid, 'vlm_direct_raw.jsonl')
        done = {r['window_id']: r for r in S.io_jsonl(rpath) if r.get('prompt_version') == version}
        tubes = S.Tubes(S.vdir(vid, 'masks.txt')) if args.motion else None
        raw = []
        prev_trip, prev_span = [], None
        for w in S.load_windows(vid):
            if len(w['objects']) < 2:
                continue
            if w['window_id'] in done:
                raw.append(done[w['window_id']])
                prev_trip, _ = parse(done[w['window_id']]['reply'], done[w['window_id']]['objects'], predicates)
                prev_span = done[w['window_id']]['span']
                continue
            paths = index[str(w['window_id'])]
            mot = ''
            if args.motion:
                import motion as M
                mot = M.facts_block(M.motion_facts(w, tubes, names, stuff, ref=lambda t: f'mark {t}'))
            prev = ''
            if args.prev_context and prev_trip:
                keep = [t for t in prev_trip if t[0] in w['objects'] and t[2] in w['objects']][:8]
                if keep:
                    prev = ('In the previous clip (frames %d-%d), these relations were reported:\n' % tuple(prev_span)
                            + '\n'.join(f'mark {s} | {p} | mark {o}' for s, p, o in keep)
                            + '\nThey may or may not still hold. Report only what you see in this clip.')
            prompt = build_prompt(w, dossiers, names, len(paths), predicates, mot, prev)
            try:
                rep = llm.chat(prompt, images=[cv2.imread(p) for p in paths], system=SYSTEM,
                               max_tokens=args.max_tokens, temperature=args.temperature).strip()
            except Exception as e:
                print(f'[{vid}] window {w["window_id"]}: failed ({e!r})', flush=True)
                continue
            raw.append({'window_id': w['window_id'], 'span': w['span'], 'objects': w['objects'],
                        'reply': rep, 'prompt_version': version, 'motion': mot, 'prev': prev})
            prev_trip, _ = parse(rep, w['objects'], predicates)
            prev_span = w['span']
            S.io_jsonl(rpath, raw)
            print(f'[{vid}] w{w["window_id"]}: {rep[:100]!r}', flush=True)
        S.io_jsonl(rpath, raw)
        rows, n_bad = [], 0
        for r in raw:
            trip, bad = parse(r['reply'], r['objects'], predicates)
            n_bad += bad
            for s, p, o in trip:
                rows.append({'window_id': r['window_id'], 'span': r['span'], 'subject': s,
                             'object': o, 'predicate': p, 'predicate_id': predicates.index(p),
                             'phrases': [p], 'sources': ['vlm_direct']})
        S.io_jsonl(S.vdir(vid, 'triplets_aligned_ref.jsonl'), rows)
        print(f'[{vid}] {len(raw)} windows, {len(rows)} triplets, {n_bad} lines dropped '
              '(predicate not in list or unknown mark)', flush=True)


if __name__ == '__main__':
    main()
