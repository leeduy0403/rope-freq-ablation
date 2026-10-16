"""D1: band x task accuracy-drop heatmap, one panel per length."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analyze.common import load_importance
from src.config import load_config, model_slug
from src.rope.bands import band_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model-id")
    ap.add_argument("--tag", default="compress_f8", help="sweep subdir, e.g. compress_f8 or zero")
    a = ap.parse_args()
    cfg = load_config(a.config, {"model_id": a.model_id})
    df = load_importance(cfg["results_dir"], cfg["model_id"], a.tag)
    slug = model_slug(cfg["model_id"])
    out_csv = Path(cfg["results_dir"]) / slug / f"importance_{a.tag}.csv"
    df.to_csv(out_csv, index=False)

    lengths = sorted(df.length.unique())
    tasks = [t for t in cfg["tasks"] if t in set(df.task)]
    labels = (df.drop_duplicates("band").sort_values("band")
              .apply(lambda r: f"b{r.band}\n{band_label(r.wavelength_range)}", axis=1).tolist())
    vmax = max(0.05, df["drop"].abs().max())
    fig, axes = plt.subplots(len(lengths), 1, figsize=(1.1 * len(labels) + 2, 1.0 * len(tasks) * len(lengths) + 1.5),
                             squeeze=False)
    for ax, L in zip(axes[:, 0], lengths):
        m = (df[df.length == L].pivot(index="task", columns="band", values="drop").reindex(tasks))
        im = ax.imshow(m.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for (i, j), v in __import__("numpy").ndenumerate(m.values):
            if v == v:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_yticks(range(len(tasks)), tasks)
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"accuracy drop when band is compressed — length {L}")
        fig.colorbar(im, ax=ax, fraction=0.03)
    axes[-1, 0].set_xlabel("band (wavelength range, tokens) — band 0 = highest frequency")
    fig.tight_layout()
    out = Path(cfg["figures_dir"]) / f"{slug}_D1_heatmap_{a.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}\nsaved -> {out_csv}")


if __name__ == "__main__":
    main()
