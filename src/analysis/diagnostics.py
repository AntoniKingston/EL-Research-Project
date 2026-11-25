"""Ensemble diagnostics, stability, computational cost and selected hyperparameters of a run."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.aggregation import PRIMARY_METRIC, ordered_methods
from src.run_tools.resume import load_results

RUN_KEYS = ["dataset", "fold", "method", "seed"]
COST_METRICS = ("n_params", "epochs", "train_seconds")


def _gm(values) -> float:
    return float(np.exp(np.log(np.asarray(values, dtype=float)).mean()))


def load_diagnostics(run_dir: Path, tasks: dict[str, str]) -> pd.DataFrame:
    """diagnostics.csv with a `task` column (`tasks`: dataset -> task)."""
    df = load_results(run_dir / "diagnostics.csv")
    df["task"] = df["dataset"].map(tasks)
    return df


def staged_loss_ratios(diag: pd.DataFrame) -> pd.DataFrame:
    """Staged primary loss L(m) relative to the full ensemble L(M), per (task, method, member):
    geometric mean over the runs of each dataset, then over the datasets of the task."""
    primary = diag["task"].map(PRIMARY_METRIC)
    df = diag[diag["metric"] == primary].copy()
    full = df[df["member"] == df["member"].max()].set_index(RUN_KEYS)["value"]
    df["ratio"] = df["value"].to_numpy() / full.reindex(df.set_index(RUN_KEYS).index).to_numpy()
    per_dataset = df.groupby(["task", "dataset", "method", "member"])["ratio"].agg(_gm)
    return per_dataset.groupby(["task", "method", "member"]).agg(_gm).rename("ratio").reset_index()


def member_norm_ratios(diag: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """||f_m|| / ||f_1|| and the weighted w_m ||f_m|| / ||f_1|| (w_1 = 1, w_m = nu), geometric
    mean over all runs, per (method, member). `weights`: dataset, fold, method -> nu."""
    df = diag[diag["metric"] == "member_output_norm"].copy()
    first = df[df["member"] == 1].set_index(RUN_KEYS)["value"]
    df["ratio"] = df["value"].to_numpy() / first.reindex(df.set_index(RUN_KEYS).index).to_numpy()
    nu = weights.set_index(["dataset", "fold", "method"])["additive_coef"]
    df["nu"] = nu.reindex(df.set_index(["dataset", "fold", "method"]).index).to_numpy()
    df["weighted"] = df["ratio"] * np.where(df["member"] == 1, 1.0, df["nu"])
    return df.groupby(["method", "member"])[["ratio", "weighted"]].agg(_gm).reset_index()


def load_similarities(run_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Mean off-diagonal similarity per run and measure, and the mean matrix per
    (method, measure) over all runs."""
    rows, matrices = [], {}
    for path in sorted(run_dir.glob("*/similarity/*.npz")):
        match = re.fullmatch(r"(.+)_f(\d+)_s(\d+)", path.stem)
        if match is None:
            continue
        method, fold, seed = match[1], int(match[2]), int(match[3])
        with np.load(path) as data:
            for measure in ("cka", "cosine"):
                matrix = data[measure]
                off = matrix[~np.eye(len(matrix), dtype=bool)]
                rows.append(
                    {
                        "dataset": path.parent.parent.name,
                        "fold": fold,
                        "method": method,
                        "seed": seed,
                        "measure": measure,
                        "mean_offdiag": float(off.mean()),
                    }
                )
                matrices.setdefault((method, measure), []).append(matrix)
    mean_matrices = {k: np.mean(v, axis=0) for k, v in matrices.items()}
    return pd.DataFrame(rows), mean_matrices


def stability_ratios(summary: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Geometric mean over the datasets of each task of std / std_fold / std_seed of the
    primary metric relative to `reference`."""
    primary = summary["task"].map(PRIMARY_METRIC)
    sub = summary[summary["metric"] == primary]
    rows = []
    for task, group in sub.groupby("task"):
        for column in ("std", "std_fold", "std_seed"):
            table = group.pivot(index="dataset", columns="method", values=column)
            ratios = table.div(table[reference], axis=0)
            for method in ordered_methods(table.columns):
                rows.append(
                    {
                        "task": task,
                        "quantity": column,
                        "method": method,
                        "ratio": _gm(ratios[method]),
                    }
                )
    return pd.DataFrame(rows)


def cost_summary(run_dir: Path, reference: str) -> pd.DataFrame:
    """Per method: median over all refits of parameters, refit epochs and training time, and
    the geometric mean over datasets of the per-dataset mean relative to `reference`."""
    df = load_results(run_dir / "metrics.csv")
    df = df[df["metric"].isin(COST_METRICS)]
    medians = df.groupby(["method", "metric"])["value"].median().unstack()
    per_dataset = df.groupby(["metric", "dataset", "method"])["value"].mean().unstack()
    rows = []
    for method in ordered_methods(medians.index):
        row: dict = {"method": method}
        for metric in COST_METRICS:
            table = per_dataset.loc[metric]
            row[f"{metric}_median"] = float(medians.loc[method, metric])
            row[f"{metric}_ratio"] = _gm(table[method] / table[reference])
        rows.append(row)
    return pd.DataFrame(rows)


def load_best_params(run_dir: Path) -> pd.DataFrame:
    """Selected hyperparameters of every (dataset, method, fold), one row each."""
    rows = []
    for path in sorted(run_dir.glob("*/best_params/*.json")):
        method, _, fold = path.stem.rpartition("_f")
        data = json.loads(path.read_text())
        rows.append(
            {
                "dataset": path.parent.parent.name,
                "method": method,
                "fold": int(fold),
                "best_epoch": data["best_epoch"],
                **data["params"],
            }
        )
    return pd.DataFrame(rows)
