"""D2 table + D3 overlay + D4 correlations (importance vs method compression per band)."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analyze.common import load_importance
from src.config import load_config, model_slug
from src.rope.methods import compression_profile


def corr(x, y, method):
    s = pd.Series(x).corr(pd.Series(y), method=method)
    return float(s) if s == s else float("nan")  # NaN when one side is constant (e.g. PI)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model-id")
    ap.add_argument("--tag", default="compress_f8")
    ap.add_argument("--base", type=float, help="rope_theta (default: read from model config)")
    ap.add_argument("--l-orig", type=int, help="native context (default: model config)")
    ap.add_argument("--scale", type=float, default=4, help="extension scale shown in the overlay")
    a = ap.parse_args()
    cfg = load_config(a.config, {"model_id": a.model_id})
    imp = load_importance(cfg["results_dir"], cfg["model_id"], a.tag)
    rotary_dim, n_bands = int(imp.rotary_dim.iloc[0]), int(imp.n_bands.iloc[0])
    base, l_orig = a.base, a.l_orig
    if base is None or l_orig is None:
        from src.select_model import inspect
        info = inspect(cfg["model_id"])
        base = base or info["rope_theta"]
        l_orig = l_orig or info["max_pos"]
    comp = compression_profile(rotary_dim, base, l_orig, n_bands)
    slug = model_slug(cfg["model_id"])
    rdir = Path(cfg["results_dir"]) / slug
    comp.to_csv(rdir / "D2_compression_profile.csv", index=False)

    # D4
    rows = []
    for (task, L), g in imp.groupby(["task", "length"]):
        v = g.sort_values("band")["drop"].values
        for (meth, s), c in comp.groupby(["method", "scale"]):
            cb = c.sort_values("band").compression_bits.values
            rows.append({"task": task, "length": L, "method": meth, "scale": s,
                         "pearson": corr(v, cb, "pearson"), "spearman": corr(v, cb, "spearman")})
    d4 = pd.DataFrame(rows)
    d4.to_csv(rdir / f"D4_correlations_{a.tag}.csv", index=False)
    print(d4.pivot_table(index=["task", "length"], columns=["method", "scale"], values="spearman")
          .round(2).to_string())

    # D3: longest length, one panel per task
    L = imp.length.max()
    tasks = [t for t in cfg["tasks"] if t in set(imp.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), squeeze=False, sharey=True)
    cs = comp[comp.scale == a.scale]
    for ax, t in zip(axes[0], tasks):
        g = imp[(imp.task == t) & (imp.length == L)].sort_values("band")
        ax.bar(g.band, g["drop"], color="#888", label="importance (acc drop)")
        ax.set_title(f"{t} @ {L}")
        ax.set_xlabel("band (0 = high freq)")
        ax2 = ax.twinx()
        for meth, c in cs.groupby("method"):
            c = c.sort_values("band")
            ax2.plot(c.band, c.compression_bits, marker="o", label=f"{meth} ×{a.scale:g}")
        ax2.set_ylim(0, np.log2(a.scale) * 1.1)
        if ax is axes[0, -1]:
            ax2.set_ylabel("compression (bits)")
            ax2.legend(fontsize=7, loc="upper left")
    axes[0, 0].set_ylabel("accuracy drop")
    fig.tight_layout()
    out = Path(cfg["figures_dir"]) / f"{slug}_D3_overlay_{a.tag}_s{a.scale:g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
