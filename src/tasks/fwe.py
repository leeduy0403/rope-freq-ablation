"""Frequent Words Extraction: Zeta-distributed coded words, report the top 3."""
import re

import numpy as np

from src.tasks.base import Task

CONS = "bcdfghjklmnprstvz"
VOW = "aeiou"


def make_vocab(rng, size):
    vocab = set()
    while len(vocab) < size:
        n_syl = rng.integers(2, 4)
        vocab.add("".join(CONS[rng.integers(len(CONS))] + VOW[rng.integers(len(VOW))]
                          for _ in range(n_syl)))
    return sorted(vocab)


class FWE(Task):
    name = "fwe"
    max_new_tokens = 30

    def __init__(self, alpha=2.0, vocab_size=500, **params):
        super().__init__(alpha=alpha, vocab_size=vocab_size, **params)
        self.alpha = alpha
        self.vocab_size = vocab_size

    def _vocab(self, seed):
        rng = np.random.default_rng([seed, 3])
        vocab = make_vocab(rng, self.vocab_size)
        return list(rng.permutation(vocab))  # rank order is random w.r.t. spelling

    def words(self, n_units, seed):
        vocab = self._vocab(seed)
        p = np.arange(1, self.vocab_size + 1, dtype=float) ** -self.alpha
        rng = np.random.default_rng([seed, 4])
        idx = rng.choice(self.vocab_size, size=n_units, p=p / p.sum())  # prefix-stable
        return vocab, [vocab[i] for i in idx]

    def build(self, n_units, seed, depth_fraction):
        vocab, ws = self.words(n_units, seed)
        uniq, counts = np.unique(ws, return_counts=True) if ws else ([], [])
        order = np.argsort(-np.asarray(counts), kind="stable")
        top = [str(uniq[i]) for i in order[:3]]
        c = sorted(counts, reverse=True)
        tie = len(c) > 3 and c[2] == c[3]
        user = (
            "Read the following coded text and track the frequency of each coded word. "
            "Find the three most frequently appeared coded words.\n\n"
            f"{' '.join(ws)}\n\n"
            "Question: Do not provide any explanation. Please ignore the dots '....'. "
            "What are the three most frequently appeared words in the above coded text?"
        )
        prefix = "Answer: According to the coded text above, the three most frequently appeared words are:"
        return user, prefix, sorted(top), {"top_counts": [int(x) for x in c[:4]], "tie": bool(tie),
                                           "vocab": vocab}

    def generate(self, tok, length_tokens, seed, depth_fraction=0.5, tol=0.02):
        # re-seed on a 3rd/4th-place tie so the gold top-3 is unambiguous
        for attempt in range(20):
            s = seed if attempt == 0 else seed * 1000 + attempt
            ex = super().generate(tok, length_tokens, s, depth_fraction, tol)
            if not ex.meta["tie"]:
                ex.meta.update(effective_seed=s)
                ex.gold = {"top3": ex.gold, "vocab": ex.meta.pop("vocab")}
                return ex
        raise RuntimeError("fwe: could not draw a tie-free example")

    def score(self, output, gold):
        vocab = set(gold["vocab"])
        pred = {w for w in re.findall(r"[a-z]+", output.lower()) if w in vocab}
        return pred == set(gold["top3"])  # exact set: top 3, no extras
