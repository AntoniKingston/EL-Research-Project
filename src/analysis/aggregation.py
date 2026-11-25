"""Within-dataset and cross-dataset aggregation of outer-test metrics (`metrics.csv`)."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.run_tools.resume import load_results

METHOD_ORDER = (
    "single_resnet",
    "resnet",
    "tangos",
    "batchensemble",
    "ensemble_none",
    "sym_cka",
    "sym_cos",
    "sfo_cka",
    "sfo_cos",
)
METHOD_LABELS = {
    "single_resnet": "ResNet (Stage 1)",
    "resnet": "ResNet (tuned)",
    "tangos": "TANGOS",
    "batchensemble": "BatchEnsemble",
    "ensemble_none": "Ens. (unreg.)",
    "sym_cka": "Sym. CKA",
    "sym_cos": "Sym. Cos",
    "sfo_cka": "SFO CKA",
    "sfo_cos": "SFO Cos",
}
DATASET_LABELS = {
    "california": "California Housing",
    "house_16H": "House 16H",
    "diamonds": "Diamonds",
    "elevators": "Elevators",
    "Ailerons": "Ailerons",
    "Bike_Sharing_Demand": "Bike Sharing Demand",
    "electricity": "Electricity",
    "credit": "Credit",
    "default-of-credit-card-clients": "Default Credit Card",
    "covertype": "Covertype",
    "pol": "Pol",
    "MiniBooNE": "MiniBooNE",
}
TASK_METRICS = {
    "classification": ("accuracy", "roc_auc", "nll"),
    "regression": ("rmse", "mae", "r2"),
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
    "nll": "NLL",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "$R^2$",
}
HIGHER_IS_BETTER = frozenset({"accuracy", "roc_auc", "r2"})
# Metric used for the combined (classification + regression) analysis of each task.
PRIMARY_METRIC = {"classification": "nll", "regression": "rmse"}
LOSS_METRICS = ("nll", "rmse", "mae")


def load_metrics(run_dir: Path) -> pd.DataFrame:
    df = load_results(run_dir / "metrics.csv")
    return df[df["method"].isin(METHOD_ORDER)]


def ordered_methods(methods) -> list[str]:
    return [m for m in METHOD_ORDER if m in set(methods)]


def task_datasets(df: pd.DataFrame, task: str) -> list[str]:
    names = set(df.loc[df["task"] == task, "dataset"])
    return [d for d in DATASET_LABELS if d in names] + sorted(names - set(DATASET_LABELS))


def fold_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per (task, dataset, method, metric, fold): \\bar m^{(f)}_{d,c}."""
    keys = ["task", "dataset", "method", "metric", "fold"]
    return df.groupby(keys, as_index=False)["value"].mean()


def within_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Per (task, dataset, method, metric): mean over all fold-seed results, the standard
    deviation over all F x S results (`std`), and for reference sigma_fold / sigma_seed."""
    keys = ["task", "dataset", "method", "metric"]
    pooled = df.groupby(keys)["value"].agg(mean="mean", std="std", n="count")
    per_fold = df.groupby([*keys, "fold"])["value"].agg(fold_mean="mean", seed_std="std")
    by_fold = per_fold.groupby(keys).agg(
        std_fold=("fold_mean", "std"), std_seed=("seed_std", "mean")
    )
    return pooled.join(by_fold).reset_index()


def means_table(summary: pd.DataFrame, metric: str, datasets: list[str]) -> pd.DataFrame:
    """datasets x methods matrix of mean scores for one metric."""
    sub = summary[summary["metric"] == metric]
    table = sub.pivot(index="dataset", columns="method", values="mean")
    return table.loc[datasets, ordered_methods(table.columns)]


def rank_table(means: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Ranks per dataset (1 = best, average ranks for ties)."""
    values = means.to_numpy()
    if metric in HIGHER_IS_BETTER:
        values = -values
    ranks = np.vstack([rankdata(row) for row in values])
    return pd.DataFrame(ranks, index=means.index, columns=means.columns)


def combined_rank_table(summary: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Ranks on every dataset by its task's primary metric (NLL / RMSE)."""
    parts = [
        rank_table(means_table(summary, metric, task_datasets(df, task)), metric)
        for task, metric in PRIMARY_METRIC.items()
    ]
    return pd.concat(parts)


def loss_ratios(means: pd.DataFrame, reference: str) -> pd.DataFrame:
    """r_{d,c} = mu_{d,c} / mu_{d,reference} for a strictly positive loss metric."""
    return means.div(means[reference], axis=0)


def geometric_mean(ratios: pd.DataFrame) -> pd.Series:
    return pd.Series(np.exp(np.log(ratios.to_numpy()).mean(axis=0)), index=ratios.columns)


def combined_loss_ratios(summary: pd.DataFrame, df: pd.DataFrame, reference: str) -> pd.DataFrame:
    parts = [
        loss_ratios(means_table(summary, metric, task_datasets(df, task)), reference)
        for task, metric in PRIMARY_METRIC.items()
    ]
    return pd.concat(parts)


def win_tie_loss(
    summary: pd.DataFrame,
    fold_tests: pd.DataFrame,
    metric: str,
    datasets: list[str],
    baseline: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per method: datasets where it is significantly better / not significantly different /
    significantly worse than `baseline` (paired fold-level test), and raw better / worse
    counts of the mean."""
    means = means_table(summary, metric, datasets)
    sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
    tests = fold_tests[(fold_tests["metric"] == metric) & (fold_tests["baseline"] == baseline)]
    rows = []
    for method in means.columns:
        if method == baseline:
            continue
        diff = sign * (means[method] - means[baseline])
        p = tests[tests["method"] == method].set_index("dataset").reindex(datasets)["p"]
        significant = p < alpha
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "method": method,
                "wins": int(((diff > 0) & significant).sum()),
                "ties": int((~significant).sum()),
                "losses": int(((diff < 0) & significant).sum()),
                "better_mean": int((diff > 0).sum()),
                "worse_mean": int((diff < 0).sum()),
            }
        )
    return pd.DataFrame(rows)
