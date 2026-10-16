"""Variable Tracking: a chain X1 = V, X2 = X1, ... scattered in the haystack."""
import re
import string

import numpy as np

from src.tasks.base import Task, filler_sentences


class VT(Task):
    name = "vt"
    max_new_tokens = 40

    def __init__(self, n_hops=4, ordered=False, **params):
        super().__init__(n_hops=n_hops, ordered=ordered, **params)
        self.n_hops = n_hops
        self.ordered = ordered

    def _chain(self, seed):
        rng = np.random.default_rng([seed, 2])
        names = []
        while len(names) < self.n_hops:
            n = "".join(rng.choice(list(string.ascii_uppercase), size=5))
            if n not in names:
                names.append(n)
        value = str(rng.integers(10_000, 100_000))
        stmts = [f"VAR {names[0]} = {value}."]
        stmts += [f"VAR {names[i]} = {names[i - 1]}." for i in range(1, self.n_hops)]
        pos = np.sort(rng.uniform(0.02, 0.98, size=self.n_hops))
        if not self.ordered:
            pos = rng.permutation(pos)
        return names, value, stmts, pos

    def build(self, n_units, seed, depth_fraction):
        names, value, stmts, pos = self._chain(seed)
        sents = filler_sentences(n_units, seed)
        for p, s in sorted(zip(pos, stmts), reverse=True):  # insert back-to-front
            sents.insert(int(round(p * len(sents))), s)
        context = " ".join(sents)
        user = (
            "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
            f"{context}\n\n"
            f"Question: Find all variables that are assigned the value {value} in the text above."
        )
        prefix = (f"Answer: According to the chain(s) of variable assignment in the text above, "
                  f"{self.n_hops} variables are assigned the value {value}, they are:")
        return user, prefix, sorted(names), {"value": value}

    def score(self, output, gold):
        pred = set(re.findall(r"\b[A-Z]{5}\b", output))
        return pred == set(gold)  # exact set: all names, no extras
