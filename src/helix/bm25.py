"""BM25 in thirty lines, implemented here rather than imported.

Keyword search is the right first retriever: fast,
interpretable, and a strong baseline, with the useful property that you can
see exactly why a document was retrieved. That transparency is the reason it
is written out here instead of pulled from a library. A participant who wants
to know why the wrong chunk ranked first can read this file, and the workshop
runs on numpy alone with nothing to install.

Robertson and Zaragoza (2009), with the usual k1=1.5, b=0.75.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def scores(
    corpus_tokens: list[list[str]],
    query_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    n = len(corpus_tokens)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    avgdl = sum(len(d) for d in corpus_tokens) / n

    df: Counter[str] = Counter()
    for doc in corpus_tokens:
        df.update(set(doc))

    out = np.zeros(n, dtype=np.float32)
    for i, doc in enumerate(corpus_tokens):
        tf = Counter(doc)
        dl = len(doc)
        total = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if not freq:
                continue
            # IDF is what makes a rare term like "amsterdam" outweigh a common
            # one like "leave". Without it, a long document that repeats the
            # common words wins, which is a failure mode worth seeing once.
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            total += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        out[i] = total
    return out
