import numpy as np

from src.rope.bands import band_slices, band_wavelengths
from src.rope.methods import (compression_profile, theta_ntk, theta_original, theta_pi,
                              theta_yarn, yarn_gamma)

D, BASE, L = 64, 10000.0, 4096


def test_band_slices_cover_all_freqs():
    s = band_slices(D, 8)
    assert s[0].start == 0 and s[-1].stop == D // 2
    assert all(a.stop == b.start for a, b in zip(s, s[1:]))


def test_band0_is_highest_frequency():
    th = theta_original(D, BASE)
    lam = band_wavelengths(th, band_slices(D, 8))
    assert lam[0][1] < lam[-1][0]  # band 0 shortest wavelengths


def test_yarn_ramp_direction():
    th = theta_original(D, BASE)
    g = yarn_gamma(th, L)
    assert g[0] == 0.0   # highest freq: not interpolated
    assert g[-1] == 1.0  # lowest freq: fully interpolated
    ty = theta_yarn(th, 8, L)
    assert np.isclose(ty[0], th[0]) and np.isclose(ty[-1], th[-1] / 8)


def test_pi_uniform_ntk_shape():
    th = theta_original(D, BASE)
    assert np.allclose(theta_pi(th, 4) / th, 0.25)
    r = theta_ntk(D, BASE, 4) / th
    assert np.isclose(r[0], 1.0)            # high freq untouched
    assert np.isclose(r[-1], 0.25, rtol=1e-6)  # lowest freq compressed by exactly 1/s


def test_profile_table_shape():
    df = compression_profile(D, BASE, L, 8)
    assert len(df) == 8 * 3 * 3
    assert (df.compression_bits >= -1e-9).all()


def test_dotted_config_override():
    from src.config import load_config
    cfg = load_config("config/default.yaml", sets=["task_params.fwe.alpha=3.0", "n_bands=4"])
    assert cfg["task_params"]["fwe"]["alpha"] == 3.0 and cfg["n_bands"] == 4
