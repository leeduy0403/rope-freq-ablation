"""Task interface, synthetic haystack filler and token-exact length fitting."""
from dataclasses import dataclass, field

import numpy as np

# Neutral filler: no digits, no all-caps words (they would collide with needles / VT names).
FILLER_SENTENCES = [
    "The morning light spread slowly across the quiet valley.",
    "A narrow path wound between the old stone walls.",
    "Several birds gathered on the branches of the tall oak tree.",
    "The river carried leaves downstream at a gentle pace.",
    "Farmers in the region usually plant their crops in early spring.",
    "The library kept its oldest books in a cool and dry room.",
    "Clouds drifted over the hills without bringing any rain.",
    "The market opened early and closed shortly after noon.",
    "Children walked to school along the edge of the meadow.",
    "A light breeze moved through the open windows of the house.",
    "The baker prepared fresh bread before the sun came up.",
    "Travelers often stopped at the small inn near the bridge.",
    "The museum displayed pottery collected from many villages.",
    "Snow covered the mountain peaks for most of the year.",
    "The harbor was busy with fishing boats returning at dusk.",
    "Gardeners trimmed the hedges around the central square.",
    "The train moved steadily through the long green fields.",
    "Musicians played softly in the corner of the cafe.",
    "The teacher wrote a short poem on the board.",
    "Wild flowers grew along both sides of the country road.",
    "The old clock in the hall chimed every hour.",
    "A family of deer crossed the clearing at sunset.",
    "The workshop smelled of fresh wood and warm glue.",
    "Lanterns were hung along the street for the evening festival.",
    "The lake reflected the colors of the autumn forest.",
    "Students gathered in the courtyard between their lessons.",
    "The painter mixed colors carefully on a wooden palette.",
    "Rain tapped lightly against the roof of the barn.",
    "The ferry crossed the bay twice each afternoon.",
    "A thin layer of frost covered the grass in the morning.",
    "The village held a small fair at the end of every summer.",
    "Bees moved from flower to flower in the warm garden.",
    "The shop on the corner sold maps and travel guides.",
    "Waves rolled onto the sandy beach in a steady rhythm.",
    "The cook stirred a large pot of soup over the fire.",
    "An old bridge connected the two halves of the town.",
    "The orchard produced apples, pears and plums each year.",
    "Visitors admired the carved doors of the ancient chapel.",
    "The night sky was clear and full of stars.",
    "A cat slept peacefully on the warm windowsill.",
    "The road climbed gradually toward the mountain pass.",
    "Neighbors shared vegetables from their gardens with each other.",
    "The tailor measured the fabric twice before cutting it.",
    "Fog settled over the fields late in the evening.",
    "The choir practiced in the hall every week.",
    "Sheep grazed quietly on the slopes above the farm.",
    "The postman delivered letters along the winding lanes.",
    "A wooden boat rested on the shore near the reeds.",
    "The forest path was soft with fallen pine needles.",
    "Lamps glowed in the windows as the town grew dark.",
]


@dataclass
class Example:
    prompt: str
    gold: object
    n_tokens: int
    meta: dict = field(default_factory=dict)


def count_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def apply_template(tok, user_msg, answer_prefix):
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": user_msg}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + answer_prefix
    return f"{user_msg}\n\n{answer_prefix}"


def filler_sentences(n, seed):
    """Prefix-stable: the first k sentences are identical for every n >= k."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(FILLER_SENTENCES), size=n)
    return [FILLER_SENTENCES[i] for i in idx]


class Task:
    name = "base"
    max_new_tokens = 32

    def __init__(self, **params):
        self.params = params

    # --- to implement -------------------------------------------------------
    def build(self, n_units, seed, depth_fraction):
        """Return (user_msg, answer_prefix, gold, meta) for a haystack of n_units."""
        raise NotImplementedError

    def score(self, output, gold):
        raise NotImplementedError

    # --- shared -------------------------------------------------------------
    def render(self, tok, n_units, seed, depth_fraction):
        user, prefix, gold, meta = self.build(n_units, seed, depth_fraction)
        prompt = apply_template(tok, user, prefix)
        return prompt, gold, meta

    def generate(self, tok, length_tokens, seed, depth_fraction=0.5, tol=0.02):
        """Grow the haystack until the *tokenized, templated* prompt hits length_tokens ±tol."""
        def n_tok(n):
            return count_tokens(tok, self.render(tok, n, seed, depth_fraction)[0])

        lo, hi = 0, 16
        while n_tok(hi) <= length_tokens:
            lo, hi = hi, hi * 2
        while hi - lo > 1:  # largest n with n_tok(n) <= target
            mid = (lo + hi) // 2
            if n_tok(mid) <= length_tokens:
                lo = mid
            else:
                hi = mid
        best = min((lo, hi), key=lambda n: abs(n_tok(n) - length_tokens))
        prompt, gold, meta = self.render(tok, best, seed, depth_fraction)
        n = count_tokens(tok, prompt)
        if abs(n - length_tokens) / length_tokens > tol:
            raise ValueError(f"{self.name}: cannot fit {length_tokens} tokens (got {n})")
        meta.update(n_units=best)
        return Example(prompt=prompt, gold=gold, n_tokens=n, meta=meta)
