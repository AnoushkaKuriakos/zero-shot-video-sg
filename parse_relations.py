#!/usr/bin/env python3
"""
Step 6 — LLM4SGG Chain-2 (predicate alignment) + LLM4SGG's final merge.

Ported from rlqja1107/torch-LLM4SGG triplet_extraction_process/
  alignment_classes_vg.py         alignment_relation() prompt, the pre-
                                  processing rules, batches of 10, the
                                  "Answer:" parsing and the error-repair pass
  final_preprocess_triplets_vg.py union of original + paraphrased triplets,
                                  drop anything aligned to None, then keep ONE
                                  "fine-grained" predicate per (subject, object):
                                  the least frequent one
Changed, and why:
  * lexicon = the SAIL-VOS predicate list (numbered 1..N) instead of VG-50
  * few-shot examples: theirs, re-numbered to our list. An example whose
    answer is not in our list, or whose lexeme contains a more specific
    predicate of ours ('next to' -> near, when we have 'next to'), is dropped (keeping it would teach the model a
    wrong mapping; whole-word match on multi-word predicates, see the 2026-11-04
    fix below); a "0.None" example is dropped if our list HAS a predicate
    for that verb (e.g. their 'driving' -> None).
  * their example 'containing' -> '0.has' is inconsistent (0 means None);
    it is rendered with has's real number when 'has' is in our list
  * error-repair bug fixed: `aligned_relation_dict[k] = f"{predicate_to_idx[v]}.{v}"`
    raised KeyError when the reply had no number and was not a predicate
  * entity alignment (alignment_entity) is not needed: entities are already
    track ids (marks)
  * the "fine-grained" selection is per window (an image in LLM4SGG); the
    frequency ranking is computed over all windows of all videos in the run.
    --keep-all-predicates disables it (useful because SAIL-VOS GT can hold
    several predicates for one pair).

Alignment decisions are cached in OUT/alignment_<fingerprint>.json and shared
by both paths (same phrase -> same predicate, whichever captioner said it).

Output: OUT/<vid>/triplets_aligned_<path>.jsonl
  {window_id, span, subject, object, predicate, predicate_id, phrases, sources}

    python parse_relations.py --data-root $DATA --videos fam_1_mcs_2 --path som --backend qwen
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict

import sgcore as S

# (lexeme, answer) pairs from alignment_relation() in alignment_classes_vg.py (LLM4SGG).
# The method uses ONE worked example, ('waiting', None): adding the five posture examples below made the 7B model
# prefer positional predicates ("on") and lose "talking to" relations on PSG4D-GTA (Recall@100 24.39 vs 28.46).
# Set ALIGN_EXAMPLES=all to use every example that fits the predicate list.
LLM4SGG_EXAMPLES = [('next to', 'near'), ('are parked in', 'parked on'), ('waiting', None),
            ('sitting', 'sitting on'), ('grazing', 'eating'), ('pointing to', None),
            ('lies on', 'lying on'), ('sitting underneath', 'under'),
            ('placed next to', 'near'), ('looking down at', 'looking at'),
            ('containing', 'has'), ('perched on', 'sitting on'), ('driving', None),
            ('hangs on', 'hanging from')]
EXAMPLES = LLM4SGG_EXAMPLES if os.environ.get('ALIGN_EXAMPLES') == 'all' else [('waiting', None)]


def lexicon(predicates):
    return ' '.join(f'{i + 1}.{p}' for i, p in enumerate(predicates))


def examples(predicates):
    idx = {p: i + 1 for i, p in enumerate(predicates)}
    heads = {p.split()[0] for p in predicates}
    lines = []
    for lx, ans in EXAMPLES:
        if ans is None:
            if lx.split()[0] in heads:
                continue                     # our list has it: None would be wrong
            a = '0.None'
        elif ans in idx:
            # drop if our list has a MORE specific predicate inside the lexeme
            # (VG maps 'next to' -> near; SAIL-VOS has 'next to' itself)
            # whole-word match (2026-11-04 fix): a substring test let 'in'/'on'
            # match inside 'sitting', 'looking', 'lies on' and dropped nearly all
            # Only an exact match or a multi-word predicate counts as more specific;
            # a bare preposition ('on' in 'hangs on', 'lies on') does not.
            if any(p != ans and (p == lx or (' ' in p and f' {p} ' in f' {lx} '))
                   for p in predicates):
                continue
            a = f'{idx[ans]}.{ans}'
        else:
            continue
        lines.append(f"Question: Given the lexeme '{lx},' find semantically relevant lexeme "
                     f"in the predefined predicate lexicon. Answer: {a} ")
    return ''.join(lines)


def alignment_relation(llm, relation, predicates):
    """alignment_classes_vg.alignment_relation, lexicon swapped."""
    question = (
        f"The predefined predicate lexicon containing {len(predicates)} lexemes is numbered "
        f"as follows: {lexicon(predicates)} \n"
        "Given the lexeme, the task is to find semantically relevant lexeme from the "
        "predefined predicate lexicon. However, if there is no semantically relevant lexeme "
        "in the predefined predicate lexicon, please answer 0.None. "
        "Let's take a few examples. \n"
        + examples(predicates) +
        f"Please answer the following {len(relation)} questions. ")
    for r in relation:
        question += (f"Question: Given the lexeme '{r}', find semantically relevant lexeme "
                     f"in the predefined predicate lexicon. Answer:")
    return llm.chat(question, max_tokens=60 * len(relation) + 50, temperature=0)


def preprocess(rel):
    """Their pre-processing: >4 words -> None; strip auxiliaries."""
    if len(rel.split(" ")) > 4:
        return None, None
    w = rel.split(" ")
    if len(w) > 1 and w[0] in ['is', 'for', 'being', 'are', 'can', 'arranged', 'were', 'was']:
        return rel, " ".join(w[1:]).strip()
    if len(w) > 2 and w[0] in ['can', 'attempting', 'attempts', 'about'] and w[1] in ['be', 'to']:
        return rel, " ".join(w[2:]).strip()
    return rel, rel


def repair(v, predicates):
    """Their post-hoc error handling, returning 'N.pred' or '0.None'."""
    idx = {p: i + 1 for i, p in enumerate(predicates)}
    v = v.strip().strip('.').strip()
    parts = v.split('.')
    if len(parts) == 2 and parts[1].strip() in idx:
        return f'{idx[parts[1].strip()]}.{parts[1].strip()}'
    if len(parts) < 2:
        return f'{idx[v]}.{v}' if v in idx else '0.None'            # FIX: was KeyError
    head, name = parts[0].strip(), '.'.join(parts[1:]).strip()
    if name in ('Non', 'No', 'None'):
        return '0.None'
    if head.isdigit() and 1 <= int(head) <= len(predicates):
        p = predicates[int(head) - 1]
        if name in p or p in name:
            return f'{int(head)}.{p}'
    return '0.None'


def fingerprint(predicates, args):
    h = hashlib.sha1(json.dumps([predicates, examples(predicates), args.backend,
                                 str(getattr(args, 'model', ''))]).encode())
    return h.hexdigest()[:12]


def align_all(phrases, predicates, llm, cache, log):
    idx = {p: i + 1 for i, p in enumerate(predicates)}
    todo = []
    for ph in phrases:
        if ph in cache:
            continue
        key, q = preprocess(ph)
        if key is None:
            cache[ph] = '0.None'
        elif q in idx:
            cache[ph] = f'{idx[q]}.{q}'                              # exact match
        else:
            todo.append((ph, q))
    for i in range(0, len(todo), 10):                               # batches of 10
        chunk = todo[i:i + 10]
        try:
            resp = alignment_relation(llm(), [q for _, q in chunk], predicates)
        except Exception as e:
            log.append({'error': repr(e), 'phrases': [p for p, _ in chunk]})
            continue                                                 # not cached: retried next run
        answers = resp.split("Answer:")[1:]
        for (ph, _), res in zip(chunk, answers):
            raw = res[:res.find('\n')].strip() if '\n' in res else res.strip()
            raw = raw.split(' Question')[0].strip()
            cache[ph] = repair(raw, predicates)
            if cache[ph] == '0.None':
                log.append({'phrase': ph, 'llm_answer': raw, 'aligned': None})
        for ph, _ in chunk[len(answers):]:
            log.append({'phrase': ph, 'error': 'no answer in reply'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--path', required=True, choices=S.PATHS)
    ap.add_argument('--keep-all-predicates', action='store_true',
                    help="skip LLM4SGG's one-predicate-per-pair selection")
    S.add_llm_args(ap)
    args = ap.parse_args()
    _llm = []
    llm = lambda: _llm[0] if _llm else _llm.append(S.make_llm(args)) or _llm[0]  # lazy
    predicates = S.predicate_list(args.data_root)
    fp = fingerprint(predicates, args)
    cpath = os.path.join(S.OUT, f'alignment_{fp}.json')
    os.makedirs(S.OUT, exist_ok=True)
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    print(f'{len(predicates)} predicates; alignment cache {cpath} ({len(cache)} entries)')

    raw = {v: S.io_jsonl(S.vdir(v, f'triplets_raw_{args.path}.jsonl')) for v in args.videos}
    log = []
    align_all(sorted({r['predicate_phrase'] for rs in raw.values() for r in rs}),
              predicates, llm, cache, log)
    json.dump(cache, open(cpath, 'w'), indent=1)

    # final_preprocess: union per window, drop None, predicate frequencies
    per_win = {}
    pred_count = defaultdict(int)
    for vid, rs in raw.items():
        comb = defaultdict(lambda: defaultdict(lambda: {'phrases': set(), 'sources': set()}))
        for r in rs:
            a = cache.get(r['predicate_phrase'], '0.None')
            name = a.split('.', 1)[1] if '.' in a else 'None'
            if name not in predicates:
                continue
            key = (r['window_id'], tuple(r['span']))
            d = comb[key][(r['subject'], name, r['object'])]
            d['phrases'].add(r['predicate_phrase']); d['sources'].add(r['source'])
        for key, trips in comb.items():
            for (s, p, o) in trips:
                pred_count[p] += 1
        per_win[vid] = comb
    order = [p for p, _ in sorted(pred_count.items(), key=lambda kv: kv[1], reverse=True)]

    for vid, comb in per_win.items():
        out = []
        for (wid, span), trips in comb.items():
            by_pair = defaultdict(list)
            for (s, p, o), d in trips.items():
                by_pair[(s, o)].append((p, d))
            for (s, o), plist in by_pair.items():
                if not args.keep_all_predicates:
                    # fine-grained = least frequent = largest index in `order`
                    plist = [max(plist, key=lambda x: order.index(x[0]))]
                for p, d in plist:
                    out.append({'window_id': wid, 'span': list(span), 'subject': s, 'object': o,
                                'predicate': p, 'predicate_id': predicates.index(p),
                                'phrases': sorted(d['phrases']), 'sources': sorted(d['sources'])})
        out.sort(key=lambda r: (r['window_id'], r['subject'], r['object']))
        S.io_jsonl(S.vdir(vid, f'triplets_aligned_{args.path}.jsonl'), out)
        json.dump(log, open(S.vdir(vid, f'align_log_{args.path}.json'), 'w'), indent=1)
        print(f'[{vid}] {args.path}: {len(out)} aligned triplets from {len(raw[vid])} raw '
              f'({sum(1 for l in log if l.get("aligned", 1) is None)} phrases -> None, '
              f'{sum("error" in l for l in log)} LLM errors)')


if __name__ == '__main__':
    main()
