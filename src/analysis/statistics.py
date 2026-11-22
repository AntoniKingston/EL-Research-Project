"""Statistical tests following Demsar (2006) and Benavoli et al. (2016)."""

import math
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.aggregation import HIGHER_IS_BETTER

# Critical values q_alpha of the Nemenyi test (studentized range / sqrt(2)), alpha = 0.05,
# for K = 2..10 compared methods (Demsar, 2006, Table 5a).
NEMENYI_Q_005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


def friedman(ranks: pd.DataFrame) -> dict[str, float]:
    """Friedman chi^2_F and Iman-Davenport F_F from a datasets x methods rank matrix."""
    n, k = ranks.shape
    mean_ranks = ranks.mean(axis=0).to_numpy()
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(mean_ranks**2) - k * (k + 1) ** 2 / 4)
    df1, df2 = k - 1, (k - 1) * (n - 1)
    denominator = n * (k - 1) - chi2
    f_stat = (n - 1) * chi2 / denominator if denominator > 0 else math.inf
    return {
        "N": n,
        "K": k,
        "chi2_F": float(chi2),
        "p_chi2": float(stats.chi2.sf(chi2, k - 1)),
        "F_F": float(f_stat),
        "df1": df1,
        "df2": df2,
        "p_F": float(stats.f.sf(f_stat, df1, df2)) if math.isfinite(f_stat) else 0.0,
        "CD": nemenyi_cd(k, n),
    }


def nemenyi_cd(k: int, n: int) -> float:
    return NEMENYI_Q_005[k] * math.sqrt(k * (k + 1) / (6 * n))


def nemenyi_pairs(ranks: pd.DataFrame) -> pd.DataFrame:
    """All pairwise mean-rank differences with the Nemenyi decision at alpha = 0.05."""
    n, k = ranks.shape
    cd = nemenyi_cd(k, n)
    mean_ranks = ranks.mean(axis=0)
    rows = []
    methods = list(ranks.columns)
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            diff = float(mean_ranks[a] - mean_ranks[b])
            rows.append(
                {"method_a": a, "method_b": b, "rank_diff": diff, "significant": abs(diff) > cd}
            )
    return pd.DataFrame(rows)


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired Wilcoxon signed-rank test (exact when possible); (W, p)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    result = stats.wilcoxon(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def wilcoxon_holm(
    scores: pd.DataFrame,
    pairs: list[tuple[str, str]],
    metric: str,
    scope: str,
) -> pd.DataFrame:
    """Paired Wilcoxon tests over datasets (rows of `scores`) for each (proposed, baseline)
    pair, Holm-adjusted within this family. `metric` decides which direction is better."""
    sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
    rows = []
    for proposed, baseline in pairs:
        a, b = scores[proposed].to_numpy(), scores[baseline].to_numpy()
        w, p = wilcoxon(a, b)
        improvement = sign * (a - b)
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "proposed": proposed,
                "baseline": baseline,
                "N": len(a),
                "median_diff": float(np.median(a - b)),
                "n_better": int((improvement > 0).sum()),
                "W": w,
                "p": p,
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out


def fold_level_tests(folds: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Per (dataset, metric, method): paired two-sided t-test of the F fold-level means
    against `baseline`. With F = 5, a two-sided Wilcoxon test cannot reach p < 0.05."""
    rows = []
    for key, group in folds.groupby(["task", "dataset", "metric"]):
        task, dataset, metric = cast("tuple[str, str, str]", key)
        table = group.pivot(index="fold", columns="method", values="value")
        if baseline not in table:
            continue
        for method in table.columns:
            if method == baseline:
                continue
            a, b = table[method].to_numpy(), table[baseline].to_numpy()
            diff = a - b
            if np.allclose(diff, diff[0]):
                t_stat, p = (0.0, 1.0) if np.allclose(diff, 0.0) else (math.inf, 0.0)
            else:
                result = stats.ttest_rel(a, b)
                t_stat, p = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "metric": metric,
                    "method": method,
                    "baseline": baseline,
                    "mean_diff": float(diff.mean()),
                    "t": t_stat,
                    "p": p,
                }
            )
    return pd.DataFrame(rows)
