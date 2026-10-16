import json
from pathlib import Path

import pandas as pd

from src.config import model_slug


def sweep_dir(results_dir, model_id, tag):
    return Path(results_dir) / model_slug(model_id) / "sweep" / tag


def load_importance(results_dir, model_id, tag):
    """Long table: task, length, band, acc, drop (= acc(band=none) - acc(band)), wavelength."""
    rows = []
    for p in sorted(sweep_dir(results_dir, model_id, tag).glob("*.json")):
        r = json.loads(p.read_text())
        rows.append({"task": r["task"], "length": r["length"], "band": r["band"],
                     "acc": r["accuracy"], "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1],
                     "wavelength_range": r["wavelength_range"], "rotary_dim": r["rotary_dim"],
                     "n_bands": r["n_bands"]})
    if not rows:
        raise FileNotFoundError(f"no sweep cells in {sweep_dir(results_dir, model_id, tag)}")
    df = pd.DataFrame(rows)
    base = df[df.band.isna()].set_index(["task", "length"]).acc.rename("base_acc")
    bands = df[df.band.notna()].copy()
    bands["band"] = bands.band.astype(int)
    bands = bands.join(base, on=["task", "length"])
    bands["drop"] = bands.base_acc - bands.acc
    return bands
