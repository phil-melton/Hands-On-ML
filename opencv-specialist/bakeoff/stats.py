"""The statistics of the bake-off, small enough to check by hand.

pass@k       Chen et al. 2021 estimator 1 - C(n-c, k) / C(n, k): the chance that at least one of k replies drawn
             without replacement from n replies, c of them correct, is correct. Unbiased for any n >= k.
SE           pass@1 at T=0: binomial sqrt(p (1-p) / n). pass@k: bootstrap over tasks (resample tasks with
             replacement, recompute the mean of the per-task estimates).
clustered    Tasks that share a prompt (same family and phrasing, different scene seed) are not independent
             draws: at T=0 they get the same reply. The cluster-robust SE and the cluster bootstrap resample or
             sum whole prompts instead, and are the honest ones when many tasks share a prompt.
McNemar      Paired pass/fail on the same tasks. Only discordant pairs carry information; under H0 each is a
             coin flip, so the exact two-sided p-value is a binomial tail. The paired difference of pass rates
             comes with its own SE (sample SD of the per-task differences / sqrt(n)), which uses the pairing.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n replies of which c are correct (Chen et al. 2021, estimator 1)."""
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        return 1.0  # fewer than k wrong replies: every draw of k contains a correct one
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def binomial_se(p: float, n: int) -> float:
    return math.sqrt(p * (1.0 - p) / n) if n > 0 else float("nan")


def bootstrap_se(values: Sequence[float], clusters: Sequence | None = None, *, B: int = 2000, seed: int = 0) -> float:
    """SE of the mean of ``values`` by bootstrap: resample tasks, or whole clusters when ``clusters`` is given."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = v[rng.integers(0, v.size, size=(B, v.size))].mean(axis=1)
    else:
        labels, inv = np.unique(np.asarray(clusters), return_inverse=True)
        if labels.size < 2:
            return float("nan")
        sums = np.bincount(inv, weights=v)
        sizes = np.bincount(inv).astype(np.float64)
        pick = rng.integers(0, labels.size, size=(B, labels.size))
        means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    return float(means.std(ddof=1))


def cluster_se(values: Sequence[float], clusters: Sequence) -> float:
    """Cluster-robust SE of a mean: sqrt(G/(G-1) * sum_g (sum_{i in g} (v_i - mean))^2) / n."""
    v = np.asarray(values, dtype=np.float64)
    labels, inv = np.unique(np.asarray(clusters), return_inverse=True)
    g = labels.size
    if v.size < 2 or g < 2:
        return float("nan")
    resid_sums = np.bincount(inv, weights=v - v.mean())
    return float(math.sqrt(g / (g - 1) * float((resid_sums ** 2).sum())) / v.size)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (A pass, B fail) and c (A fail, B pass)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def paired_compare(a: Sequence[int], b: Sequence[int], clusters: Sequence | None = None) -> dict:
    """Paired pass/fail vectors on the same tasks: counts, A - B difference, its SE, and the exact McNemar p.

    ``se`` is the plain paired SE; ``se_cluster`` and ``p_sign_cluster`` (an exact sign test over prompts whose
    mean difference is not zero) are given when clusters are supplied."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have the same length")
    n = a.size
    a_only, b_only = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    d = (a - b).astype(np.float64)
    out = {
        "n": n, "a_only": a_only, "b_only": b_only,
        "both": int(((a == 1) & (b == 1)).sum()), "neither": int(((a == 0) & (b == 0)).sum()),
        "diff": float(d.mean()) if n else float("nan"),
        "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "p": mcnemar_exact(a_only, b_only),
    }
    if clusters is not None and n > 1:
        out["se_cluster"] = cluster_se(d, clusters)
        labels, inv = np.unique(np.asarray(clusters), return_inverse=True)
        mean_d = np.bincount(inv, weights=d) / np.bincount(inv)
        out["p_sign_cluster"] = mcnemar_exact(int((mean_d > 0).sum()), int((mean_d < 0).sum()))
    return out


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if len(values) else float("nan")
