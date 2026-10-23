#!/usr/bin/env python3
"""Token frequencies of reference corpora with the Qwen3.8 tokenizer (input of vocab_select.py --corpus).
Sources: <corpus dir>/magpie_zh/*.parquet (Magpie-Qwen2-Pro-200K-Chinese) and <corpus dir>/magpie_en/*.parquet
(Magpie-Qwen2.5-Pro-1M, shard 0), "response" column; Python sources under <code dir> (capped at 60M chars).
One int64 count array per source. Needs pyarrow + tokenizers.
  python tools/corpus_freq.py <corpus dir> <tokenizer.json> <code dir> vocab/corpus_counts.npz
"""
import glob, os, sys
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

corpus, tokjson, codedir, out = sys.argv[1:5]
tok = Tokenizer.from_file(tokjson)
V = 248320


def count(texts, c):
    for i in range(0, len(texts), 2048):
        for e in tok.encode_batch(texts[i:i + 2048], add_special_tokens=False):
            np.add.at(c, np.asarray(e.ids, dtype=np.int64), 1)


res = {}
for name in ("magpie_zh", "magpie_en"):
    c = np.zeros(V, dtype=np.int64)
    for f in sorted(glob.glob(f"{corpus}/{name}/*.parquet")):
        texts = [t for t in pq.read_table(f, columns=["response"]).column("response").to_pylist() if t]
        count(texts, c)
    res[name] = c
    print(name, int(c.sum()), "tokens", int((c > 0).sum()), "distinct", flush=True)
code, n = [], 0
for f in sorted(glob.glob(os.path.join(codedir, "**/*.py"), recursive=True)):
    try:
        t = open(f, encoding="utf-8").read()
    except Exception:
        continue
    code.append(t); n += len(t)
    if n > 60_000_000:
        break
c = np.zeros(V, dtype=np.int64); count(code, c); res["code"] = c
print("code", int(c.sum()), "tokens", int((c > 0).sum()), "distinct", flush=True)
np.savez(out, **res)
