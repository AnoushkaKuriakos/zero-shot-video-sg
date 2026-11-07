#!/usr/bin/env python3
"""
Step 5 — LLM4SGG Chain-1: triplet extraction from the window captions.

Ported from rlqja1107/torch-LLM4SGG triplet_extraction_process/
  extract_triplet_with_original_caption.py      (prompt ORIGINAL)
  extract_triplet_with_paraphrased_caption.py   (prompt PARAPHRASED)
Kept as in the repo:
  * both prompts word for word, including their few-shot examples
  * the "Please answer the following {n} questions." batching: LLM4SGG sends
    the 5 COCO captions of an image as 5 questions in one call; we send the
    sentences of a window caption the same way (<= 7 per call, the size of
    their number_dict)
  * the <subject, predicate, object> text answer format and refine_output()
    parsing; ORIGINAL and PARAPHRASED run as two separate calls and their
    triplets are unioned later (final_preprocess_triplets_vg.py)
  * temperature 0
Changed, and why:
  * gpt-3.5-turbo -> your backend (--backend)
  * ONE added sentence + ONE added example (after theirs) telling the model to
    keep "mark N" entities verbatim. Their prompt reduces entities to simple
    nouns ("a beautiful woman" -> "woman"), which would turn "mark 7" into
    "mark" and lose the grounding.
  * refine_output for the original prompt found no triplets in replies without
    "Question" (about half of them), so it now scans the whole reply as a fallback.
  * refine_output for the paraphrase prompt indexes res.split(":")[4]/[6],
    which breaks as soon as the model's wording differs by one colon; the
    original parser runs first and a tolerant fallback takes over if it fails.
  * entities are resolved to track ids ("mark 7" -> 7); triplets whose subject
    or object is not a mark of that window are dropped and logged (LLM4SGG
    grounds free nouns with GLIP instead; here every object is already a mask).

Output: OUT/<vid>/triplets_raw_<path>.jsonl
  {window_id, span, subject, object, predicate_phrase, source, sentence}

    python extract_relations.py --data-root $DATA --videos fam_1_mcs_2 --path som --backend qwen
"""
import argparse
import hashlib
import json
import os
import re

import sgcore as S

number_dict = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}

MARK_NOTE = ("Here the objects in each sentence are referred to by numbered marks "
             "(e.g. 'mark 7'); keep every mark exactly as written, with its number, "
             "as the subject or object. \n")

# --- ORIGINAL prompt: verbatim from extract_triplet_with_original_caption.py
ORIGINAL_HEAD = (
    "From the given sentence, the task is to extract meaningful triplets formed as <subject, predicate, object>. Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). \n"
    + MARK_NOTE +
    "Let's take a few examples to understand how to extract meaningful triplets. "
    "Question: Given the sentence 'a slice of bread is covered with a sour cream and quacamole,' extract meaningful triplets. Answer: "
    "Meaningful triplets are <bread, covered with, sour cream>, and <bread, covered with, guacamole>. \n"
    "Question: Given the sentence 'A beautiful woman walking a dog on top of a beach,' extract meaningful triplets. Answer: "
    "Meaningful triplets are <woman, walking with, dog>, <woman, on, beach>, and <dog, on, beach>. \n"
    "Question: Given the sentence 'Four clock sitting on a floor next to a woman's feet,' extract meaningful triplets. Answer: "
    "Meaningful triplets are <clock, sitting on, floor> and <clock, next to, feet>. \n"
    "Question: Given the sentence 'One person sits in a chair looking at her phone while another rests on the couch,' extract meaningful triplets. Answer: "
    "Meaningful triplets are <person, sits in, chair>, <person, looking at, phone>, and <person, rests on, couch>. \n"
    "Question: Given the sentence 'A lady and a child near a park bench with kites and ducks flying in the sky and on the ground,' extract meaningful triplets. Answer: "
    "Meaningful triplets are <lady, near, park bench>, <child, near, park bench>, <kites, flying in sky>, and <ducks, on, ground>. \n"
    "Question: Given the sentence 'Two men sit on a bench near the sidewalk and one of them talks on a cell phone,' extract meaningful triplets. Answer: "
    "Meaningful triplets are <men, sit on, bench>, <bench, near, sidewalk>, and <man, talks on, phone>. \n"
    "Question: Given the sentence 'Mark 12 hands a cup to mark 4 while mark 4 sits on mark 9,' extract meaningful triplets. Answer: "
    "Meaningful triplets are <mark 12, hands cup to, mark 4> and <mark 4, sits on, mark 9>. \n"
    "Please answer the following {n} questions. ")

# --- PARAPHRASED prompt: verbatim from extract_triplet_with_paraphrased_caption.py
PARA_HEAD = (
    "From the given sentence, the task is to extract meaningful triplets formed as <subject, predicate, object>. To extract meaningful triplets from the sentence, please follow the following two steps. "
    "Step 1: Paraphrase the sentence. \n"
    "Step 2: From the paraphrased sentence obtained in the Step 1, extract meaningful triplets formed as <subject, predicate, object>. Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). \n"
    + MARK_NOTE +
    "Let's take a few examples to understand how to extract meaningful triplets. "
    "Question: Given the sentence 'a slice of bread is covered with a sour cream and quacamole,' extract meaningful triplets. Answer: "
    "Step 1: The sentence can be paraphrased as: 'A piece of bread is topped with both sour cream and guacamole'. "
    "Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence are: <bread, topped with, sour cream>, <bread, topped with, guacamole>. "
    "The meaningful triplets are <bread, topped with, sour cream>, and <bread, topped with, guacamole>. \n"
    "Question: Given the sentence 'A beautiful woman walking a dog on top of a beach,' extract meaningful triplets. Answer: "
    "Step 1: The sentence can be paraphrased as: 'A lovely woman strolling with a dog on the beach'. "
    "Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence are: <woman, strolling with, dog>, <woman, on, beach>, <dog, on, beach>. "
    "The meaningful triplets are <woman, strolling with, dog>, <woman, on, beach>, and <dog, on, beach>. \n"
    "Question: Given the sentence 'Four clock sitting on a floor next to a woman's feet,' extract meaningful triplets. Answer: "
    "Step 1: The sentence can be paraphrased as: 'Four clocks are placed on the floor beside a woman's feet'. "
    "Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <clocks, placed on, floor>, <clocks, beside, feet>. "
    "The meaningful triplets are <clocks, placed on, floor> and <clocks, beside, feet>. \n"
    "Question: Given the sentence 'One person sits in a chair looking at her phone while another rests on the couch,' extract meaningful triplets. Answer: "
    "Step 1: The sentence can be paraphrased as: 'A person is seated in a chair, using their phone, while someone else is relaxing on the couch'. "
    "Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <person, seated in, chair>, <person, using, phone>, <person, relaxing on, couch>. "
    "The meaningful triplets are <person, seated in, chair>, <person, using, phone>, and <person, relaxing on, couch>. \n"
    "Question: Given the sentence 'A lady and a child near a park bench with kites and ducks flying in the sky and on the ground,' extract meaningful triplets. Answer: "
    "Step 1: The sentence can be paraphrased as: 'A woman and a child are close to a park bench, while kites soar through the sky and ducks move around on the ground'. "
    "Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <woman, close to, park bench>, <child, close to, park bench>, <kites, soar through, sky>, <ducks, move around, ground>. "
    "The meaningful triplets are <woman, close to, park bench>, <child, close to, park bench>, <kites, soar through, sky>, and <ducks, move around, ground>. \n"
    "Question: Given the sentence 'Two men sit on a bench near the sidewalk and one of them talks on a cell phone,' extract meaningful triplets. Answer: "
    "Step 1: The sentence can be paraphrased as: 'Two guys are seated on a bench near the road, and one of them talks on a mobile phone'. "
    "Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <guys, seated on, bench>, <bench, near, road>, <guy, talks on, phone>. "
    "The meaningful triplets are <guys, seated on, bench>, <bench, near, road>, and <guy, talks on, phone>. \n"
    "Question: Given the sentence 'Mark 12 hands a cup to mark 4 while mark 4 sits on mark 9,' extract meaningful triplets. Answer: "
    "Step 1: The sentence can be paraphrased as: 'Mark 12 gives a cup to mark 4, who is seated on mark 9'. "
    "Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <mark 12, gives cup to, mark 4>, <mark 4, seated on, mark 9>. "
    "The meaningful triplets are <mark 12, gives cup to, mark 4> and <mark 4, seated on, mark 9>. \n"
    "Please answer the following {n} questions. ")


def clean(c):
    # LLM4SGG: c.strip('\n').strip("").strip("'").strip(".")
    return c.strip('\n').strip("").strip("'").strip(".").replace("'", "")


def build(head, sentences):
    q = head.format(n=number_dict[len(sentences)])
    for c in sentences:
        q += f"Question: Given the sentence '{c}', extract meaningful triplets. Answer: "
    return q


def triplets_in(text):
    """LLM4SGG's inner loop: split on '<', take 3 comma fields."""
    out = []
    for a in text[text.find("triplet"):].split("<")[1:]:
        a = a.split(",")
        if len(a) < 3:
            continue
        out.append([a[0].strip(), a[1].strip(), a[2].split(">")[0].strip()])
    return out


def refine_output_original(response):
    """extract_triplet_with_original_caption.refine_output, with a fallback for replies
    that do not follow the "Question ... Answer:" layout."""
    output_dict = {}
    for i, res in enumerate(response.split("Question")):
        if i == 0:
            continue
        parts = res.split("Answer:")
        if len(parts) > 1:
            output_dict[i] = {'before_matched_triplets': triplets_in(parts[1])}
    if not any(v['before_matched_triplets'] for v in output_dict.values()):
        # Qwen often drops the "Question ... Answer:" layout ("1. **Given the sentence...**")
        # or writes "<mark 1, sitting in, mark 7>" with no "triplet" before it. Sentence
        # attribution is lost here, but it is only logged; the triplet itself is kept.
        output_dict = {1: {'before_matched_triplets': triplets_in('triplet' + response)}}
    return output_dict


def refine_output_paraphrased(response):
    """extract_triplet_with_paraphrased_caption.refine_output, with a fallback
    for when the fixed colon positions don't hold."""
    output_dict = {}
    for i, res in enumerate(response.split("Question")):
        if i == 0:
            continue
        try:
            p_sentence, triplets = res.split(":")[4], res.split(":")[6]   # original
            p_sentence = p_sentence[p_sentence.find("'") + 1: p_sentence.rfind("'")].strip().strip('').strip(".")
            trip = triplets_in(triplets)
        except IndexError:
            p_sentence, trip = '', []
        if not trip:                                                      # fallback
            tail = re.split(r'(?i)the meaningful triplets are', res)
            trip = triplets_in('triplet' + tail[-1]) if len(tail) > 1 else \
                triplets_in(res[res.find('Step 2'):] if 'Step 2' in res else res)
        output_dict[i] = {'p_sentence': p_sentence, 'before_matched_triplets': trip}
    return output_dict


def sentences_of(caption):
    parts = re.split(r'(?<=[.!?])\s+|\n+', caption.strip())
    return [clean(p) for p in parts if len(clean(p).split()) >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--path', required=True, choices=S.PATHS)
    ap.add_argument('--max-tokens', type=int, default=900)
    S.add_llm_args(ap)
    args = ap.parse_args()
    llm = None                          # loaded on the first cache miss
    # temperature 0 -> same prompt, same reply: cache replies so rerunning the
    # graph step on unchanged captions costs no LLM calls
    cpath = os.path.join(S.OUT, f'extract_cache_{args.backend}.json')
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    n_cached = 0

    for vid in args.videos:
        caps = S.io_jsonl(S.vdir(vid, f'captions_{args.path}.jsonl'))
        if not caps:
            print(f'[{vid}] no captions_{args.path}.jsonl — run the captioner first'); continue
        rows, log = [], []
        for c in caps:
            valid = set(c['objects'])
            sents = sentences_of(c['caption'])
            for source, head, parse in (('original', ORIGINAL_HEAD, refine_output_original),
                                        ('paraphrased', PARA_HEAD, refine_output_paraphrased)):
                for b in range(0, len(sents), 7):
                    batch = sents[b:b + 7]
                    q = build(head, batch)
                    key = hashlib.sha1(f'{getattr(args, "model_id", "")}|{args.max_tokens}|{q}'.encode()).hexdigest()
                    try:
                        if key in cache:
                            resp = cache[key]; n_cached += 1
                        else:
                            llm = llm or S.make_llm(args)
                            resp = cache[key] = llm.chat(q, max_tokens=args.max_tokens,
                                                         temperature=0)
                        parsed = parse(resp)
                    except Exception as e:
                        log.append({'window': c['window_id'], 'source': source,
                                    'error': repr(e)}); continue
                    for qi, v in parsed.items():
                        sent = batch[qi - 1] if qi - 1 < len(batch) else ''
                        for sub, pred, obj in v['before_matched_triplets']:
                            s, o = S.mark_id(sub), S.mark_id(obj)
                            if s is None or o is None or s not in valid or o not in valid or s == o:
                                log.append({'window': c['window_id'], 'dropped': [sub, pred, obj],
                                            'why': 'entity is not a mark of this window'})
                                continue
                            rows.append({'window_id': c['window_id'], 'span': c['span'],
                                         'subject': s, 'object': o, 'predicate_phrase': pred,
                                         'source': source, 'sentence': sent})
        json.dump(cache, open(cpath + '.tmp', 'w')); os.replace(cpath + '.tmp', cpath)
        S.io_jsonl(S.vdir(vid, f'triplets_raw_{args.path}.jsonl'), rows)
        json.dump(log, open(S.vdir(vid, f'extract_log_{args.path}.json'), 'w'), indent=1)
        print(f'[{vid}] {args.path}: {len(rows)} raw triplets from {len(caps)} captions, '
              f'{sum("dropped" in l for l in log)} unmarked/invalid dropped, '
              f'{sum("error" in l for l in log)} LLM errors, {n_cached} cached replies')


if __name__ == '__main__':
    main()
