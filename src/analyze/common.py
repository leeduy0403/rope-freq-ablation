import json
from pathlib import Path

import pandas as pd

from src.config import model_slug
from src.eval.metrics import paired_bootstrap_ci


def sweep_dir(results_dir, model_id, tag):
    return Path(results_dir) / model_slug(model_id) / "sweep" / tag


def load_importance(results_dir, model_id, tag):
    """Long table: task, length, band, acc, drop (= acc(band=none) - acc(band)) with a paired
    bootstrap CI (same examples in every band), wavelength range."""
    cells = {}
    for p in sorted(sweep_dir(results_dir, model_id, tag).glob("*.json")):
        r = json.loads(p.read_text())
        cells[(r["task"], r["length"], r["band"])] = r
    if not cells:
        raise FileNotFoundError(f"no sweep cells in {sweep_dir(results_dir, model_id, tag)}")
    rows = []
    for (task, length, band), r in cells.items():
        base = cells.get((task, length, None))
        if band is None or base is None:
            continue
        b = [e["correct"] for e in base["examples"]]
        x = [e["correct"] for e in r["examples"]]
        lo, hi = paired_bootstrap_ci(b, x, seed=r["seed"])
        rows.append({"task": task, "length": length, "band": int(band),
                     "acc": r["accuracy"], "base_acc": base["accuracy"],
                     "drop": base["accuracy"] - r["accuracy"], "drop_lo": lo, "drop_hi": hi,
                     "significant": lo > 0,
                     "wavelength_range": r["wavelength_range"], "rotary_dim": r["rotary_dim"],
                     "n_bands": r["n_bands"], "n": r["n_examples"]})
    return pd.DataFrame(rows).sort_values(["task", "length", "band"]).reset_index(drop=True)
