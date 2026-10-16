import pytest

from src.tasks import get_task


class WsTok:
    """Whitespace tokenizer stand-in (no download)."""
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


TOK = WsTok()


@pytest.mark.parametrize("name", ["s_niah", "mk_niah", "vt", "fwe"])
@pytest.mark.parametrize("length", [500, 2000])
def test_length_within_tolerance(name, length):
    task = get_task(name, {"mk_niah": {"n_distractors": 3}})
    ex = task.generate(TOK, length, seed=7)
    assert abs(ex.n_tokens - length) / length <= 0.02


def test_deterministic():
    t = get_task("vt")
    a, b = t.generate(TOK, 800, seed=3), t.generate(TOK, 800, seed=3)
    assert a.prompt == b.prompt and a.gold == b.gold


def test_niah_needle_present_and_scored():
    t = get_task("mk_niah", {"mk_niah": {"n_distractors": 3}})
    ex = t.generate(TOK, 800, seed=1)
    assert f"is {ex.gold}." in ex.prompt
    assert t.score(f" {ex.gold}.", ex.gold)
    wrong = next(iter(ex.meta["distractors"].values()))
    assert not t.score(f" {wrong}.", ex.gold)


def test_vt_exact_set():
    t = get_task("vt")
    ex = t.generate(TOK, 800, seed=1)
    assert t.score(" " + ", ".join(ex.gold), ex.gold)
    assert not t.score(" " + ", ".join(ex.gold[:-1]), ex.gold)
    assert not t.score(" " + ", ".join(ex.gold) + ", ZZZZZ", ex.gold)


def test_fwe_gold_is_true_top3():
    t = get_task("fwe")
    ex = t.generate(TOK, 1000, seed=1)
    c = ex.meta["top_counts"]
    assert c[2] > c[3]
    assert t.score(" " + ", ".join(ex.gold["top3"]), ex.gold)
    assert not t.score(" " + ex.gold["top3"][0], ex.gold)
