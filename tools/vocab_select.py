#!/usr/bin/env python3
"""Pick the pruned draft vocabulary from the model's own outputs.

Ranking: tokens seen in vocab/gen_*.jsonl by count (desc), then the remaining ids in increasing id order (byte-level BPE:
lower id = earlier merge = more frequent in the tokenizer corpus), special/added tokens always in.
Report: held-out coverage (every 5th prompt held out, frequencies from the rest) for several N, for this ranking and for
the plain id prefix [0, N). Writes vocab/draft_vocab_<N>.pt (sorted int64, frequencies from ALL prompts) for --write N.
  vocab_select.py vocab/gen_w5_vocab.jsonl [--write 32768]
"""
import argparse, collections, json, torch

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--vocab-size", type=int, default=248320)
ap.add_argument("--tokenizer", required=True, help="tokenizer.json of nvidia/Qwen3.8-27B-NVFP4")
ap.add_argument("--write", type=int, default=0)
ap.add_argument("--corpus", default="vocab/corpus_counts.npz")
ap.add_argument("--w", default="zh=0.3,en=0.2,code=0.15,gen=0.35", help="mixture weights of normalized frequencies")
a = ap.parse_args()

recs = [json.loads(l) for f in a.files for l in open(f)]
special = sorted(t["id"] for t in json.load(open(a.tokenizer))["added_tokens"])


import numpy as np
CORP = dict(np.load(a.corpus)) if a.corpus else {}
W = {k: float(v) for k, v in (x.split("=") for x in a.w.split(","))}
SRC = {"zh": "magpie_zh", "en": "magpie_en", "code": "code"}


def ranking(seqs):
    """Mixture of normalized frequencies (corpora + the model's own outputs), ties broken by id; specials first."""
    g = np.zeros(a.vocab_size)
    for s in seqs:
        np.add.at(g, np.asarray(s, dtype=np.int64), 1)
    p = W.get("gen", 0) * g / max(g.sum(), 1)
    for k, src in SRC.items():
        if W.get(k, 0) and src in CORP:
            c = CORP[src].astype(np.float64); p += W[k] * c / c.sum()
    order = np.lexsort((np.arange(a.vocab_size), -p))
    sp = set(special)
    return special + [int(t) for t in order if int(t) not in sp]


def subset(rank, n):
    out, seen = [], set()
    for t in rank:
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) == n:
            break
    return set(out)


train = [r["ids"] for r in recs if r["i"] % 5 != 0]
test = [t for r in recs if r["i"] % 5 == 0 for t in r["ids"]]
cats = collections.defaultdict(list)
for r in recs:
    if r["i"] % 5 == 0:
        cats[r["cat"]] += r["ids"]
rank = ranking(train)
allc = collections.Counter(t for r in recs for t in r["ids"])
print(f"{len(recs)} prompts, {sum(len(r['ids']) for r in recs)} tokens, {len(allc)} distinct; held-out {len(test)} tokens "
      f"from {sum(1 for r in recs if r['i'] % 5 == 0)} prompts; {len(special)} special tokens; max id seen {max(allc)}")
print("N      | freq+id-fill coverage | id-prefix coverage | worst held-out category (freq+id)")
for n in (8192, 16384, 24576, 32768, 49152, 65536):
    S = subset(rank, n)
    cov = sum(t in S for t in test) / len(test)
    cid = sum(t < n for t in test) / len(test)
    worst = min(((sum(t in S for t in v) / len(v)), k) for k, v in cats.items())
    print(f"{n:6d} | {100*cov:8.3f} %            | {100*cid:8.3f} %         | {worst[1]} {100*worst[0]:.2f} %")
if a.write:
    S = subset(ranking([r["ids"] for r in recs]), a.write)
    ids = torch.tensor(sorted(S), dtype=torch.long)
    p = f"vocab/draft_vocab_{a.write}.pt"
    torch.save(ids, p)
    print(f"wrote {p}: {ids.numel()} ids, max {int(ids[-1])}")
