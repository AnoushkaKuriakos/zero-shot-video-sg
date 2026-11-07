#!/usr/bin/env python3
"""Open questions about one video, answered by the LLM from the whole graph (for the report's showcase figure).

The prompt holds the whole graph of the video (stsg.json): every object with its class and the first sentence of
its DAM description, and every relation with its time intervals and its total time. Answers are saved as given.
    SG_OUT=$PWD/exp/v3final python tools/showcase_qa.py --video fam_1_ext_alt2 --backend qwen
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcore as S

QUESTIONS = [
    'Which relations occur in this video? Reply with only the list of relation words, e.g. driving, holding.',
    'Describe what is going on in this video in two or three sentences.',
    'How long, in total, is the driver driving the car?',
    'How long, in total, are the two people talking to each other?',
    'Who is inside the car, and when?',
    'What is the person in the passenger seat doing between 15 s and 20 s?',
]


def graph_text(G):
    cls = {o['id']: o['class'] for o in G['objects']}
    L = [f"Video length: {G['duration_s']:.1f} s.", 'Objects:']
    for o in G['objects']:
        d = (o.get('description') or '').split('. ')[0].strip().rstrip('.')
        L.append(f"  {o['class']} {o['id']}" + (f": {d}." if d else ''))
    L.append('Relations (subject, predicate, object: when it holds; total time):')
    for r in sorted(G['relations'], key=lambda r: r['intervals_s'][0][0]):
        iv = ', '.join(f'{a:.1f}-{b:.1f} s' for a, b in r['intervals_s'])
        tot = sum(b - a for a, b in r['intervals_s'])
        L.append(f"  {cls[r['subject']]} {r['subject']} {r['predicate']} {cls[r['object']]} {r['object']}: {iv}; total {tot:.1f} s")
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--video', required=True); S.add_llm_args(ap)
    a = ap.parse_args()
    G = json.load(open(S.vdir(a.video, 'stsg.json', make=False)))
    ctx = graph_text(G); llm = S.make_llm(a)
    head = ('You answer questions about a video using only its scene graph below, which was extracted automatically '
            'and may be incomplete. Times are in seconds. Answer briefly and concretely; say so if the graph does '
            'not contain the answer.\n\n' + ctx + '\n\nQuestion: ')
    out = []
    for q in QUESTIONS:
        r = llm.chat(head + q + '\nAnswer:', max_tokens=220, temperature=0).strip()
        out.append({'question': q, 'answer': r}); print('Q:', q, '\nA:', r, '\n')
    json.dump({'video': a.video, 'context': ctx, 'qa': out}, open(S.vdir(a.video, 'showcase_qa.json'), 'w'), indent=1)
