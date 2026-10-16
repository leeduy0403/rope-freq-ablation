"""S-NIAH and MK-NIAH (RULER-style 'special magic number' needles)."""
import re

import numpy as np

from src.tasks.base import Task, filler_sentences

ADJ = ["amber", "brisk", "calm", "dusty", "eager", "faint", "gentle", "hollow", "ivory",
       "jolly", "keen", "lofty", "mellow", "noble", "olive", "polite", "quiet", "rustic",
       "silver", "tidy", "upbeat", "vivid", "witty", "young", "zesty"]
NOUN = ["anchor", "badger", "candle", "dolphin", "engine", "falcon", "garden", "harbor",
        "island", "jacket", "kettle", "ladder", "meadow", "needle", "orchard", "pepper",
        "quarry", "rocket", "saddle", "tunnel", "violin", "walnut", "yacht", "zipper"]


def needle(key, value):
    return f"The special magic number for {key} is {value}."


class NIAH(Task):
    max_new_tokens = 16

    def __init__(self, n_distractors=0, **params):
        super().__init__(n_distractors=n_distractors, **params)
        self.n_distractors = n_distractors
        self.name = "mk_niah" if n_distractors else "s_niah"

    def _needles(self, seed):
        rng = np.random.default_rng([seed, 1])
        keys, vals = [], []
        while len(keys) < 1 + self.n_distractors:
            k = f"{ADJ[rng.integers(len(ADJ))]}-{NOUN[rng.integers(len(NOUN))]}"
            if k not in keys:
                keys.append(k)
                vals.append(str(rng.integers(1_000_000, 10_000_000)))
        depths = rng.uniform(0.05, 0.95, size=self.n_distractors)
        return keys, vals, depths

    def build(self, n_units, seed, depth_fraction):
        keys, vals, ddepths = self._needles(seed)
        sents = filler_sentences(n_units, seed)
        # insert distractors first, target last so its depth stays exactly depth_fraction
        inserts = [(d, needle(k, v)) for d, k, v in zip(ddepths, keys[1:], vals[1:])]
        inserts.append((depth_fraction, needle(keys[0], vals[0])))
        for d, s in inserts:
            sents.insert(int(round(d * len(sents))), s)
        context = " ".join(sents)
        user = (
            "Some special magic numbers are hidden within the following text. "
            "Make sure to memorize them. I will quiz you about the numbers afterwards.\n\n"
            f"{context}\n\n"
            f"What is the special magic number for {keys[0]} mentioned in the provided text?"
        )
        prefix = f"Answer: The special magic number for {keys[0]} mentioned in the provided text is"
        return user, prefix, vals[0], {"key": keys[0], "distractors": dict(zip(keys[1:], vals[1:]))}

    def score(self, output, gold):
        return gold in re.findall(r"\d+", output)
