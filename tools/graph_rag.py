#!/usr/bin/env python3
"""Graph RAG over a video scene graph (stsg.json): the LLM answers from a retrieved subgraph, not the whole graph.

  1. plan     the LLM reads the question and writes JSON: scope (global: about the whole video, e.g. "list all
              activities", "describe the video"; local: about particular objects/actions), the object mentions,
              the action mentions, and a time range if one is named.
  2. retrieve global -> every relation (sorted by start).
              local  -> nodes: each mention is matched to object nodes by embedding similarity over
                        "class id: DAM description"; predicates: each action mention matched to the predicate list;
                        edges: every relation touching a matched node, scored by similarity to the question
                        (+ a bonus for a matched predicate, + one for both ends matched), cut to the asked time
                        range, top 12. One hop is kept implicitly (edges touching a matched node).
  3. compute  for the retrieved edges, code adds facts the LLM is bad at: per relation the total time, first
              start and last end, and for a pair of objects with the same predicate in both directions (e.g.
              talking to each other) the merged total.
  4. answer   the LLM answers from the retrieved nodes, edges and computed facts only. Questions that ask to
              list the relations or objects of the whole video are answered by code from the graph (a 7B LLM
              falls into a repetition loop on long lists and invents items); other answers have repeated
              items and sentences removed.
    SG_OUT=$PWD/exp/v3final python tools/graph_rag.py --video fam_1_ext_alt2 --backend qwen
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcore as S
from showcase_qa import QUESTIONS

_ENC = None
MAX_REL = 150      # max relations in one prompt (GPU memory; mic_4_int's hybrid graph has 454)
def embed(xs):
    global _ENC
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        _ENC = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    return _ENC.encode(xs, normalize_embeddings=True, show_progress_bar=False)


def load_captions(path, cls, fps=5):
    """Window captions of the caption branch, with 'mark N' written as '<class> N' like the graph's nodes."""
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        c = json.loads(l)
        txt = re.sub(r'mark (\d+)', lambda m: f"{cls.get(int(m.group(1)), 'object')} {m.group(1)}", c['caption'])
        out.append({'span_s': [c['span'][0] / fps, (c['span'][1] + 1) / fps], 'text': txt})
    return out


def union(ivs):
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class Graph:
    def __init__(self, G, captions=None):
        """captions: optional window captions [{'span_s': [a, b], 'text': ...}] attached to the graph as text
        evidence (each linked to its time span); retrieved next to the relations, for detail only."""
        self.caps = captions or []
        self.G = G; self.cls = {o['id']: o['class'] for o in G['objects']}
        self.desc = {o['id']: (o.get('description') or '').split('. ')[0].strip().rstrip('.') for o in G['objects']}
        self.R = sorted(G['relations'], key=lambda r: r['intervals_s'][0][0])
        self.node_txt = {i: f"{self.cls[i]} {i}: {self.desc[i]}" for i in self.cls}
        self.preds = sorted({r['predicate'] for r in self.R})

    def n(self, i):
        return f'{self.cls[i]} {i}'

    def edge_txt(self, r):
        iv = ', '.join(f'{a:.1f}-{b:.1f} s' for a, b in r['intervals_s'])
        tot = sum(b - a for a, b in r['intervals_s'])
        return (f"{self.n(r['subject'])} {r['predicate']} {self.n(r['object'])}: {iv} "
                f"(total {tot:.1f} s, first starts {r['intervals_s'][0][0]:.1f} s, last ends {r['intervals_s'][-1][1]:.1f} s)")

    def rich(self, r):
        d = lambda i: f"{self.n(i)} ({self.desc[i][:90]})" if self.desc[i] else self.n(i)
        return f"{d(r['subject'])} {r['predicate']} {d(r['object'])}"

    def ask_time(self, q, plan):
        t = plan.get('time')
        if not (isinstance(t, list) and len(t) == 2 and all(isinstance(x, (int, float)) for x in t)):
            m = re.findall(r'(\d+(?:\.\d+)?)\s*s\b', q)
            t = [float(m[0]), float(m[1])] if len(m) >= 2 else ([float(m[0]) - 1, float(m[0]) + 1] if m else None)
        return t

    def retrieve_captions(self, q, plan, k=3):
        if not self.caps or plan.get('scope') == 'global':
            return []
        t = self.ask_time(q, plan)
        cand = [c for c in self.caps if not t or (c['span_s'][0] < t[1] and c['span_s'][1] > t[0])] or self.caps
        sc = embed([c['text'] for c in cand]) @ embed([q])[0]
        return sorted([cand[i] for i in np.argsort(-sc)[:k]], key=lambda c: c['span_s'][0])

    def retrieve(self, q, plan):
        if plan.get('scope') == 'global':      # whole graph, or its CAP most-voted relations if larger
            top = sorted(sorted(self.R, key=lambda r: -r['votes'])[:MAX_REL], key=lambda r: r['intervals_s'][0][0])
            used = sorted({r['subject'] for r in top} | {r['object'] for r in top})
            return used, top, ('whole graph' if len(top) == len(self.R) else f'{len(top)} most-voted relations') + ' (global question)'
        nodes = set()
        for x in plan.get('nodes') or []:
            m = re.search(r'\d+', str(x))
            if m and int(m.group(0)) in self.cls:
                nodes.add(int(m.group(0)))
        preds = {p for p in plan.get('predicates') or [] if p in self.preds}
        t = plan.get('time')
        if not (isinstance(t, list) and len(t) == 2 and all(isinstance(x, (int, float)) for x in t)):
            m = re.findall(r'(\d+(?:\.\d+)?)\s*s\b', q)
            t = [float(m[0]), float(m[1])] if len(m) >= 2 else ([float(m[0]) - 1, float(m[0]) + 1] if m else None)
        cand = self.R
        if t:
            cand = [r for r in self.R if any(a < t[1] and b > t[0] for a, b in r['intervals_s'])] or self.R
        qv = embed([q])[0]
        sc = embed([self.rich(r) for r in cand]) @ qv
        pv = embed(self.preds) @ qv                      # predicates the question itself is close to
        close = {self.preds[k] for k in np.argsort(-pv)[:2] if pv[k] >= 0.35}
        close |= self.word_preds(q)
        sc = sc + 0.25 * np.array([r['predicate'] in (preds | close) for r in cand]) + \
             0.15 * np.array([r['subject'] in nodes or r['object'] in nodes for r in cand])
        top = sorted([cand[k] for k in np.argsort(-sc)[:8]], key=lambda r: r['intervals_s'][0][0])
        used = sorted({r['subject'] for r in top} | {r['object'] for r in top})
        how = (f"top {len(top)} of {len(cand)} edges" + (f" active in {t[0]:g}-{t[1]:g} s" if t else '')
               + f"; predicates boosted: {', '.join(sorted(preds | close)) or '-'}")
        return used, top, how

    def word_preds(self, q):
        """Predicates whose main word appears in the question ('drives' -> driving, 'loads' -> loading;
        'inside'/'into' -> in). Embedding similarity alone misses short predicates such as 'in'."""
        words = re.findall(r'[a-z]+', q.lower())
        out = set()
        for p in self.preds:
            head = p.split()[0]
            if len(head) >= 4 and any(w.startswith(head[:4]) for w in words):
                out.add(p)
        if 'in' in self.preds and any(w in ('inside', 'into') for w in words):
            out.add('in')
        return out

    def facts(self, edges):
        F = []
        seen = set()
        for r in edges:
            k = (frozenset((r['subject'], r['object'])), r['predicate'])
            both = [x for x in self.R if (frozenset((x['subject'], x['object'])), x['predicate']) == k]
            if k in seen or len({(x['subject'], x['object']) for x in both}) < 2:
                continue
            seen.add(k); u = union([iv for x in both for iv in x['intervals_s']])
            a, b = sorted(k[0])
            F.append(f"{self.n(a)} and {self.n(b)} {r['predicate']} each other (either direction): "
                     + ', '.join(f'{x:.1f}-{y:.1f} s' for x, y in u) + f" (total {sum(y - x for x, y in u):.1f} s)")
        return F


PLAN = ('You plan a lookup in a video scene graph. The graph has these objects (class id: description):\n{nodes}\n'
        'and these predicates: {preds}.\n'
        'Reply with JSON only: {{"scope": "global" or "local", "nodes": [ids of the objects the question is about], '
        '"predicates": [predicates from the list that fit the question], "time": [start_s, end_s] or null}}.\n'
        'scope is "global" only when the question asks about the whole video (list all activities, describe the '
        'video, summarise); otherwise "local". time only if the question names a time.\n'
        'Examples:\n'
        'Q: Describe the video. -> {{"scope": "global", "nodes": [], "predicates": [], "time": null}}\n'
        'Q: How long is the woman holding the cup? -> {{"scope": "local", "nodes": [<id of the woman>, <id of the cup>], '
        '"predicates": ["holding"], "time": null}}\n'
        'Q: What is the man doing at 10 s? -> {{"scope": "local", "nodes": [<id of the man>], "predicates": [], "time": [9, 11]}}\n'
        'Question: {q}\nJSON:')
CAP = ('\nWindow captions (free text from the captioner, for detail only; take times and totals from the relations '
       'and computed facts):\n{caps}\n')
ANS = ('You answer a question about a video from part of its scene graph, retrieved for this question. The graph '
       'was extracted automatically and may be incomplete; times are in seconds. Use the computed totals rather '
       'than adding times yourself. Answer briefly and concretely; if the retrieved part does not contain the '
       'answer, say so.\n\nObjects:\n{nodes}\n\nRelations:\n{edges}\n{facts}\nQuestion: {q}\n{tail}Answer:')


LIST_Q = re.compile(r'^\s*(list|name)\b|\b(which|what)\s+(relations?|relationships|predicates|interactions|objects)\b', re.I)


def list_answer(g, q, edges):
    """Distinct predicates (or objects) of the graph, in order of first appearance, with how often they occur."""
    if re.search(r'\bobjects?\b', q, re.I) and not re.search(r'\brelation', q, re.I):
        ids = sorted({r['subject'] for r in edges} | {r['object'] for r in edges})
        return ', '.join(g.n(i) for i in ids)
    count = {}
    for r in edges:
        count[r['predicate']] = count.get(r['predicate'], 0) + 1
    return ', '.join(count) + f' ({len(edges)} relations in the graph)'


def dedupe(a):
    """Remove the repetition loops of small LLMs: repeated sentences and repeated comma-separated items."""
    norm = lambda x: x.strip().strip('.').strip().lower()
    out, seen_s, seen_i = [], set(), set()
    for sent in re.split(r'(?<=[.!?])\s+', a.strip()):
        if not sent.strip() or norm(sent) in seen_s:
            continue
        seen_s.add(norm(sent))
        items = sent.split(', ')
        if len(items) > 2:                                   # a list: keep each item once
            kept = []
            for x in items:
                if norm(x) not in seen_i:
                    seen_i.add(norm(x)); kept.append(x)
            end = sent.rstrip()[-1] if sent.rstrip()[-1:] in '.!?' else ''
            sent = ', '.join(kept).rstrip('.!?') + end
        out.append(sent)
    return ' '.join(out)


def answer(g, q, llm, tail='', max_tokens=None, plan_q=None):
    """tail: extra answer-format line (benchmark scoring); plan_q: question text used for planning/retrieval."""
    pq = plan_q or q
    rep = llm.chat(PLAN.format(q=pq, nodes='\n'.join(g.node_txt[i] for i in sorted(g.cls)), preds=', '.join(g.preds)),
                   max_tokens=120, temperature=0)
    try:
        plan = json.loads(re.search(r'\{.*\}', rep, re.S).group(0))
    except Exception:
        plan = {'scope': 'local', 'nodes': [], 'predicates': [], 'time': None}
    nodes, edges, how = g.retrieve(pq, plan); F = g.facts(edges)
    C = g.retrieve_captions(pq, plan)
    capt = CAP.format(caps='\n'.join(f"{c['span_s'][0]:.1f}-{c['span_s'][1]:.1f} s: {c['text']}" for c in C)) if C else ''
    prompt = ANS.format(nodes='\n'.join(g.node_txt[i] for i in nodes), edges='\n'.join(g.edge_txt(r) for r in edges),
                        facts=(('\nComputed facts:\n' + '\n'.join(F) + '\n') if F else '') + capt, q=q, tail=tail)
    if plan.get('scope') == 'global' and LIST_Q.search(pq) and not tail:
        a = list_answer(g, pq, edges); how += '; list answered by code'
    else:
        mt = max_tokens or (250 if plan.get('scope') == 'global' else 160)
        a = dedupe(llm.chat(prompt, max_tokens=mt, temperature=0).strip())
    return {'question': q, 'plan': plan, 'retrieved': how, 'n_edges': len(edges), 'n_total_edges': len(g.R),
            'facts': F, 'n_captions': len(C), 'answer': a}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--video', required=True)
    ap.add_argument('--questions', default='', help='text file, one question per line (default: showcase_qa.QUESTIONS)')
    ap.add_argument('--captions', default='', help='exp dir whose <vid>/captions_ref.jsonl is attached (final setting)')
    ap.add_argument('--out', default='graph_rag_qa.json')
    S.add_llm_args(ap)
    a = ap.parse_args()
    G0 = json.load(open(S.vdir(a.video, 'stsg.json', make=False))); caps = None
    if a.captions:
        cls = {o['id']: o['class'] for o in G0['objects']}
        vd = os.path.join(a.captions, a.video); voc = json.load(open(os.path.join(vd, 'vocab.json')))['vocab']
        for w in json.load(open(os.path.join(vd, 'windows.json'))):
            for t_, c_ in w['classes'].items():
                cls.setdefault(int(t_), voc[c_] if 0 <= c_ < len(voc) else 'object')
        caps = load_captions(os.path.join(vd, 'captions_ref.jsonl'), cls)
    g = Graph(G0, caps); llm = S.make_llm(a)
    qs = [l.strip() for l in open(a.questions) if l.strip()] if a.questions else QUESTIONS
    out = [answer(g, q, llm) for q in qs]
    for x in out:
        print('Q:', x['question']); print('  plan:', x['plan']); print('  retrieved:', x['retrieved'], f"({x['n_edges']}/{x['n_total_edges']} edges)")
        print('  facts:', x['facts']); print('A:', x['answer']); print()
    json.dump({'video': a.video, 'captions': bool(a.captions), 'qa': out}, open(S.vdir(a.video, a.out), 'w'), indent=1)
