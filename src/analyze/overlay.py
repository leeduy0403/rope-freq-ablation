"""D2 table + D3 overlay + D4 correlations (importance vs method compression per band).

D4 is reported over all bands and excluding band 0: compressing band 0 breaks the model on every
task (output degenerates), so it is a model-wide failure, not task-specific importance, and it
would dominate any correlation. Spearman is identical across scales s for NTK and YaRN (their
per-band compression keeps the same rank order for every s), so one scale is enough to read.
PI compresses every band equally -> correlation undefined (NaN) by construction.
"""
import argparse
import math
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
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")  # undefined when one side is constant (e.g. PI)
    return float(pd.Series(x).corr(pd.Series(y), method=method))


def band_of_wavelength(wavelength_ranges, lam):
    """Band whose wavelength range contains lam (or the nearest one, on a log scale)."""
    def dist(r):
        lo, hi = r
        return 0.0 if lo <= lam <= hi else min(abs(math.log(lam / lo)), abs(math.log(lam / hi)))
    return int(np.argmin([dist(r) for r in wavelength_ranges]))


def d4_table(imp, comp):
    rows = []
    for (task, L), g in imp.groupby(["task", "length"]):
        g = g.sort_values("band")
        for (meth, s), c in comp.groupby(["method", "scale"]):
            c = c.sort_values("band")
            for subset, keep in (("all", g.band >= 0), ("excl_b0", g.band > 0)):
                v = g[keep]["drop"].values
                cb = c[c.band.isin(g[keep].band)].compression_bits.values
                rows.append({"task": task, "length": L, "method": meth, "scale": s, "bands": subset,
                             "pearson": corr(v, cb, "pearson"), "spearman": corr(v, cb, "spearman")})
    return pd.DataFrame(rows)


def plot_d3(imp, comp, tasks, scale, out):
    lengths = sorted(imp.length.unique())
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(lengths)))
    lam = imp.drop_duplicates("band").sort_values("band").wavelength_range.tolist()
    cs = comp[comp.scale == scale]
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.2 * len(tasks), 5.6), squeeze=False,
                             sharex=True, sharey="row", gridspec_kw={"height_ratios": [2.2, 1]})
    for k, t in enumerate(tasks):
        ax, axc = axes[0, k], axes[1, k]
        for col, L in zip(colors, lengths):
            g = imp[(imp.task == t) & (imp.length == L)].sort_values("band")
            if g.empty:
                continue
            err = np.vstack([g["drop"] - g.drop_lo, g.drop_hi - g["drop"]]).clip(0)
            ax.errorbar(g.band, g["drop"], yerr=err, color=col, marker="o", ms=3, capsize=2,
                        lw=1.4, label=f"L={L}")
            b = band_of_wavelength(lam, L)
            for a in (ax, axc):
                a.axvline(b, color=col, ls=":", lw=1)  # band whose wavelength ~ L
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(t)
        for meth, c in cs.groupby("method"):
            c = c.sort_values("band")
            axc.plot(c.band, c.compression_bits, marker=".", lw=1.2, label=f"{meth} ×{scale:g}")
        axc.set_ylim(-0.1, np.log2(scale) * 1.15)
        axc.set_xlabel("band (0 = high freq)")
        axc.set_xticks(range(len(lam)))
    axes[0, 0].set_ylabel("accuracy drop\n(paired 95% CI)")
    axes[1, 0].set_ylabel("compression\n(bits)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    axes[1, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("band importance per length (top) vs extension-method compression (bottom); "
                 "dotted = band whose wavelength ≈ L", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model-id")
    ap.add_argument("--tag", help="sweep subdir (default: from config)")
    ap.add_argument("--base", type=float, help="rope_theta (default: read from model config)")
    ap.add_argument("--l-orig", type=int, help="native context (default: model config)")
    ap.add_argument("--scale", type=float, default=4, help="extension scale shown in the overlay")
    a = ap.parse_args()
    cfg = load_config(a.config, {"model_id": a.model_id})
    a.tag = a.tag or f"compress_f{cfg['intervention']['compress_factor']:g}"
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
    imp.to_csv(rdir / f"importance_{a.tag}.csv", index=False)

    d4 = d4_table(imp, comp)
    d4.to_csv(rdir / f"D4_correlations_{a.tag}.csv", index=False)
    view = d4[(d4.scale == a.scale) & (d4.method != "PI")]
    print(f"D4 Spearman (scale {a.scale:g}; identical for every scale):")
    print(view.pivot_table(index=["task", "length"], columns=["bands", "method"], values="spearman")
          .round(2).to_string())

    tasks = [t for t in cfg["tasks"] if t in set(imp.task)]
    out = Path(cfg["figures_dir"]) / f"{slug}_D3_overlay_{a.tag}_s{a.scale:g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_d3(imp, comp, tasks, a.scale, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
