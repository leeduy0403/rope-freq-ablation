"""Phase 4: per-band compression profiles of PI / NTK-aware / YaRN. CPU only.

c_i = theta_i^method / theta_i^original  (1 = untouched, 1/s = fully interpolated).
We report compression in bits: log2(1 / c_i)  (0 = untouched, log2(s) = full PI).
YaRN's attention-temperature term is ignored: it does not change theta.
"""
import argparse

import numpy as np
import pandas as pd

from src.rope.bands import band_label, band_slices, band_wavelengths


def theta_original(head_dim, base):
    i = np.arange(0, head_dim, 2)[: head_dim // 2]
    return base ** (-i / head_dim)


def theta_pi(theta, s):
    return theta / s


def theta_ntk(head_dim, base, s):
    return theta_original(head_dim, base * s ** (head_dim / (head_dim - 2)))


def yarn_gamma(theta, L_orig, alpha=1.0, beta=32.0):
    """Interpolation weight per dim: 0 = keep (high freq), 1 = fully interpolate (low freq)."""
    wavelen = 2 * np.pi / theta
    r = L_orig / wavelen  # rotations completed within the original context
    return np.clip((beta - r) / (beta - alpha), 0.0, 1.0)


def theta_yarn(theta, s, L_orig, alpha=1.0, beta=32.0):
    g = yarn_gamma(theta, L_orig, alpha, beta)
    return (1 - g) * theta + g * (theta / s)


def compression_profile(head_dim, base, L_orig, n_bands, scales=(2, 4, 8)):
    th = theta_original(head_dim, base)
    slices = band_slices(head_dim, n_bands)
    lams = band_wavelengths(th, slices)
    rows = []
    for s in scales:
        per_method = {
            "PI": theta_pi(th, s),
            "NTK": theta_ntk(head_dim, base, s),
            "YaRN": theta_yarn(th, s, L_orig),
        }
        for method, tm in per_method.items():
            bits = np.log2(th / tm)
            for b, sl in enumerate(slices):
                rows.append({
                    "method": method, "scale": s, "band": b,
                    "wavelength": band_label(lams[b]),
                    "ratio_mean": float((tm / th)[sl].mean()),
                    "compression_bits": float(bits[sl].mean()),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--base", type=float, required=True)
    ap.add_argument("--l-orig", type=int, required=True, help="original context length")
    ap.add_argument("--n-bands", type=int, default=8)
    ap.add_argument("--out", default="results/compression_profile.csv")
    args = ap.parse_args()
    df = compression_profile(args.head_dim, args.base, args.l_orig, args.n_bands)
    df.to_csv(args.out, index=False)
    print(df.pivot_table(index=["band", "wavelength"], columns=["method", "scale"],
                         values="compression_bits").round(3).to_string())
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
