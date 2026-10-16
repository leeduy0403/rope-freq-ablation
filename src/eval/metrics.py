import numpy as np


def accuracy(correct):
    return float(np.mean(correct)) if len(correct) else float("nan")


def bootstrap_ci(correct, n_resamples=1000, ci=0.95, seed=0):
    x = np.asarray(correct, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_resamples, len(x)))].mean(axis=1)
    a = (1 - ci) / 2
    return (float(np.quantile(means, a)), float(np.quantile(means, 1 - a)))
