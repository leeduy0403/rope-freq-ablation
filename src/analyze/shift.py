"""D7: how each band's importance changes with context length.

One panel per task; x = context length (log2), y = accuracy drop with paired 95% CI, one line per
band. Tests "a band starts to matter once its wavelength approaches the context length": the line
of the band whose wavelength range contains L should rise as L grows. Also writes the table with
drop relative to baseline (drop / baseline acc), since baselines fall with length.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analyze.common import load_importance
from src.config import load_config, model_slug
from src.rope.bands import band_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model-id")
    ap.add_argument("--tag", help="sweep subdir (default: from config)")
    ap.add_argument("--bands", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--tasks", nargs="+", default=["s_niah", "vt", "fwe"])
    a = ap.parse_args()
    cfg = load_config(a.config, {"model_id": a.model_id})
    a.tag = a.tag or f"compress_f{cfg['intervention']['compress_factor']:g}"
    imp = load_importance(cfg["results_dir"], cfg["model_id"], a.tag)
    imp = imp[imp.task.isin(a.tasks)].copy()
    imp["rel_drop"] = imp["drop"] / imp.base_acc
    slug = model_slug(cfg["model_id"])
    rdir = Path(cfg["results_dir"]) / slug
    tab = imp[imp.band > 0].pivot_table(index=["task", "length"], columns="band", values="rel_drop")
    tab.round(3).to_csv(rdir / f"D7_relative_drop_{a.tag}.csv")
    print("relative drop (drop / baseline):")
    print(tab.round(2).to_string())

    lam = imp.drop_duplicates("band").set_index("band").wavelength_range.to_dict()
    tasks = [t for t in a.tasks if t in set(imp.task)]
    colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(a.bands)))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.8), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], tasks):
        for col, b in zip(colors, a.bands):
            g = imp[(imp.task == t) & (imp.band == b)].sort_values("length")
            if g.empty:
                continue
            err = np.vstack([g["drop"] - g.drop_lo, g.drop_hi - g["drop"]]).clip(0)
            ax.errorbar(g.length, g["drop"], yerr=err, color=col, marker="o", ms=4, capsize=2,
                        lw=1.5, label=f"b{b} (λ {band_label(lam[b])})")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_xscale("log", base=2)
        lengths = sorted(imp.length.unique())
        ax.set_xticks(lengths, [f"{L // 1024}k" for L in lengths])
        ax.set_title(t)
        ax.set_xlabel("context length")
    axes[0, 0].set_ylabel("accuracy drop (paired 95% CI)")
    axes[0, -1].legend(fontsize=7, loc="upper left")
    fig.suptitle("importance of each band vs context length (compress ×"
                 f"{cfg['intervention']['compress_factor']:g})", fontsize=9)
    fig.tight_layout()
    out = Path(cfg["figures_dir"]) / f"{slug}_D7_shift_{a.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
