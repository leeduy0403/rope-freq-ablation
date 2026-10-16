"""Partition the d/2 rotary frequencies into contiguous bands (band 0 = highest frequency)."""
import math

import numpy as np


def band_slices(head_dim, n_bands):
    n_freq = head_dim // 2
    edges = [round(k * n_freq / n_bands) for k in range(n_bands + 1)]
    return [slice(edges[k], edges[k + 1]) for k in range(n_bands)]


def band_wavelengths(inv_freq, slices):
    """Wavelength range lambda = 2*pi/theta (tokens) per band, as (min, max)."""
    inv = np.asarray(inv_freq, dtype=np.float64)
    out = []
    for s in slices:
        lam = 2 * math.pi / inv[s]
        out.append((float(lam.min()), float(lam.max())))
    return out


def fmt_tokens(x):
    for unit, div in (("M", 1e6), ("k", 1e3)):
        if x >= div:
            return f"{x / div:.3g}{unit}"
    return f"{x:.3g}"


def band_label(lam_range):
    lo, hi = lam_range
    return f"{fmt_tokens(lo)}–{fmt_tokens(hi)}"
