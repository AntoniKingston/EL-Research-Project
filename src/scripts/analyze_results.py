"""Aggregates the outer-test results of a run and computes the statistical tests.

    uv run python -m src.scripts.analyze_results --run results/main

Writes CSV files to `<run>/analysis/`, LaTeX tables to `documents/tables/` and critical
difference diagrams to `documents/figures/`.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.aggregation import (
    DATASET_LABELS,
    HIGHER_IS_BETTER,
    LOSS_METRICS,
    METHOD_LABELS,
    METRIC_LABELS,
    TASK_METRICS,
    combined_loss_ratios,
    combined_rank_table,
    fold_means,
    geometric_mean,
    load_metrics,
    loss_ratios,
    means_table,
    ordered_methods,
    rank_table,
    task_datasets,
    win_tie_loss,
    within_dataset,
)
from src.analysis.diagnostics import (
    cost_summary,
    load_best_params,
    load_diagnostics,
    load_similarities,
    member_norm_ratios,
    stability_ratios,
    staged_loss_ratios,
)
from src.analysis.statistics import (
    fold_level_tests,
    friedman,
    nemenyi_cd,
    nemenyi_pairs,
    wilcoxon_holm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROPOSED = ("sfo_cka", "sfo_cos")
WILCOXON_BASELINES = ("ensemble_none", "sym_cka", "sym_cos", "resnet", "single_resnet")
WTL_BASELINES = ("ensemble_none", "single_resnet")
ALPHA = 0.05
TASK_LABELS = {"classification": "Classification", "regression": "Regression"}
ENSEMBLES = ("ensemble_none", "sym_cka", "sym_cos", "sfo_cka", "sfo_cos")
METHOD_COLORS = {m: f"C{i}" for i, m in enumerate(ENSEMBLES)}
MEASURE_LABELS = {"cka": r"$\Phi_{\mathrm{CKA}}$", "cosine": r"$\Phi_{\mathrm{Cos}}$"}
SEARCH_BOUNDS = {"additive_coef": (0.05, 1.0), "orthogonality_weight": (1e-3, 10.0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default="results/main", help="run directory")
    parser.add_argument("--tables", default="documents/tables")
    parser.add_argument("--figures", default="documents/figures")
    parser.add_argument(
        "--reference", default="single_resnet", help="reference method of the loss ratios"
    )
    parser.add_argument(
        "--baseline",
        default="ensemble_none",
        help="baseline of the within-dataset tests marked in the per-dataset tables",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------- LaTeX helpers ----------
def fmt_p(p: float) -> str:
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def row_format(values: np.ndarray) -> tuple[int, int]:
    """(power-of-ten scale exponent, decimals) for one table row of comparable values."""
    magnitude = float(np.median(np.abs(values)))
    exponent = 0
    if magnitude > 0 and (magnitude < 0.01 or magnitude >= 1000):
        exponent = math.floor(math.log10(magnitude))
    scaled = magnitude / 10**exponent if magnitude > 0 else 1.0
    decimals = min(max(3 - math.floor(math.log10(scaled)), 0), 4)
    return exponent, decimals


def write_table(
    path: Path,
    caption: str,
    label: str,
    colspec: str,
    header: str,
    body: list[str],
    font: str = "",
) -> None:
    """booktabs table, shrunk to the text width only if it is wider."""
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        *([f"    {font}", r"    \setlength{\tabcolsep}{4pt}"] if font else []),
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        r"    \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        rf"    \begin{{tabular}}{{{colspec}}}",
        r"        \toprule",
        f"        {header} \\\\",
        r"        \midrule",
        *[f"        {row}" for row in body],
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def stacked(top: str, bottom: str) -> str:
    return rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{top}\\{bottom}\end{{tabular}}"


def method_header(methods: list[str]) -> str:
    """Method names split over two lines to keep wide tables narrow."""
    cells = []
    for m in methods:
        first, _, rest = METHOD_LABELS[m].partition(" ")
        cells.append(rf"\textbf{{{stacked(first, rest)}}}" if rest else rf"\textbf{{{first}}}")
    return " & ".join(cells)


def per_dataset_table(
    path: Path,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    task: str,
    metric: str,
    datasets: list[str],
    baseline: str,
) -> None:
    means = means_table(summary, metric, datasets)
    stds = (
        summary[summary["metric"] == metric]
        .pivot(index="dataset", columns="method", values="std")
        .loc[means.index, means.columns]
    )
    methods = list(means.columns)
    higher = metric in HIGHER_IS_BETTER
    tests = tests[(tests["metric"] == metric) & (tests["baseline"] == baseline)]
    p_values = tests.pivot(index="dataset", columns="method", values="p")
    diffs = tests.pivot(index="dataset", columns="method", values="mean_diff")

    body = []
    for dataset in datasets:
        mu, sd = means.loc[dataset].to_numpy(), stds.loc[dataset].to_numpy()
        exponent, decimals = row_format(mu)
        scale = 10.0**exponent
        best = np.nanmax(mu) if higher else np.nanmin(mu)
        label = DATASET_LABELS.get(dataset, dataset)
        if exponent:
            label += rf" ($\times 10^{{{exponent}}}$)"
        cells = [label]
        for method, m, s in zip(methods, mu, sd, strict=True):
            mean = f"{m / scale:.{decimals}f}"
            if np.isclose(m, best):
                mean = rf"\textbf{{{mean}}}"
            if method != baseline and method in p_values.columns:
                p, diff = p_values.loc[dataset, method], diffs.loc[dataset, method]
                if p < ALPHA:
                    improved = diff > 0 if higher else diff < 0
                    mean += r"$^{\uparrow}$" if improved else r"$^{\downarrow}$"
            cells.append(stacked(mean, rf"{{\scriptsize $\pm${s / scale:.{decimals}f}}}"))
        body.append(" & ".join(cells) + r" \\")

    direction = "higher" if higher else "lower"
    caption = (
        f"{TASK_LABELS[task]}: outer-test {METRIC_LABELS[metric]} ({direction} is better), "
        r"mean $\pm$ standard deviation over 5 outer folds $\times$ 5 seeds. "
        r"Best mean in bold; $^{\uparrow}$/$^{\downarrow}$: significantly better/worse than "
        f"{METHOD_LABELS[baseline]} (paired $t$-test over fold means, $p < {ALPHA}$)."
    )
    write_table(
        path,
        caption,
        f"tab:results_{metric}",
        "l" + "c" * len(methods),
        r"\textbf{Dataset} & " + method_header(methods),
        body,
        font=r"\small",
    )


def bold_extreme(values: list[float], fmt: str, lowest: bool) -> list[str]:
    target = min(values) if lowest else max(values)
    return [rf"\textbf{{{v:{fmt}}}}" if np.isclose(v, target) else f"{v:{fmt}}" for v in values]


# ---------- critical difference diagram ----------
def cd_diagram(path: Path, ranks: pd.DataFrame, title: str) -> None:
    mean_ranks = ranks.mean(axis=0).sort_values()
    n, k = ranks.shape
    cd = nemenyi_cd(k, n)
    names = [METHOD_LABELS[m] for m in mean_ranks.index]
    values = mean_ranks.to_numpy()

    cliques, last_end = [], -1
    for i in range(k):
        j = max(j for j in range(i, k) if values[j] - values[i] < cd)
        if j > i and j > last_end:
            cliques.append((i, j))
            last_end = j

    half = (k + 1) // 2
    clique_gap, label_gap = 0.22, 0.45
    label_top = -0.25 - clique_gap * len(cliques) - 0.2
    bottom = label_top - label_gap * (half - 1) - 0.35
    fig, ax = plt.subplots(figsize=(7, 0.55 * (1.8 - bottom)))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(bottom, 1.8)
    ax.axis("off")
    ax.hlines(0, 1, k, color="black", lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, 0, 0.12, color="black", lw=1)
        ax.text(r, 0.22, str(r), ha="center", va="bottom", fontsize=9)
    ax.hlines(1.1, 1, 1 + cd, color="black", lw=1.5)
    ax.vlines([1, 1 + cd], 1.0, 1.2, color="black", lw=1.5)
    ax.text(1 + cd / 2, 1.25, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)

    for i, (name, r) in enumerate(zip(names, values, strict=True)):
        if i < half:
            y = label_top - i * label_gap
            ax.plot([r, r, 0.6], [0, y, y], color="black", lw=0.8)
            ax.text(0.55, y, f"{name} ({r:.2f})", ha="right", va="center", fontsize=9)
        else:
            y = label_top - (k - 1 - i) * label_gap
            ax.plot([r, r, k + 0.4], [0, y, y], color="black", lw=0.8)
            ax.text(k + 0.45, y, f"({r:.2f}) {name}", ha="left", va="center", fontsize=9)

    for c, (i, j) in enumerate(cliques):
        y = -0.25 - clique_gap * c
        ax.hlines(y, values[i] - 0.05, values[j] + 0.05, color="black", lw=2.5)

    ax.set_title(title, fontsize=10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------- ensemble diagnostics, stability, cost, hyperparameters ----------
def sci(value: float, digits: int = 1) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}{{\times}}10^{{{int(exponent)}}}$"


def quartiles(values: pd.Series, fmt) -> str:
    q1, q2, q3 = values.quantile([0.25, 0.5, 0.75])
    return stacked(fmt(q2), rf"{{\scriptsize [{fmt(q1)}, {fmt(q3)}]}}")


def line_panels(path: Path, data: pd.DataFrame, panels: list[tuple[str, str, str]]) -> None:
    """One panel per (query column filter, value column, title); one line per ensemble."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.8))
    for ax, (query, column, title) in zip(np.atleast_1d(axes), panels, strict=True):
        sub = data.query(query) if query else data
        for method in ordered_methods(sub["method"].unique()):
            line = sub[sub["method"] == method].sort_values("member")
            ax.plot(
                line["member"],
                line[column],
                marker="o",
                ms=3,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_xlabel("Ensemble member $m$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def similarity_figure(path: Path, matrices: dict[tuple[str, str], np.ndarray]) -> None:
    methods = [m for m in ENSEMBLES if (m, "cka") in matrices]
    fig, axes = plt.subplots(2, len(methods), figsize=(2.1 * len(methods), 3.9), squeeze=False)
    for row, measure in enumerate(("cka", "cosine")):
        off = [matrices[(m, measure)] for m in methods]
        size = len(off[0])
        vmax = max(float(a[~np.eye(size, dtype=bool)].max()) for a in off)
        for col, (method, matrix) in enumerate(zip(methods, off, strict=True)):
            ax = axes[row, col]
            shown = np.where(np.eye(size, dtype=bool), np.nan, matrix)
            image = ax.imshow(shown, vmin=0, vmax=vmax, cmap="viridis")
            ticks = range(0, size, max(size // 4, 1))
            ax.set_xticks(list(ticks), [str(t + 1) for t in ticks], fontsize=7)
            ax.set_yticks(list(ticks), [str(t + 1) for t in ticks], fontsize=7)
            if row == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=9)
            if col == 0:
                ax.set_ylabel("CKA" if measure == "cka" else "Cosine", fontsize=9)
        fig.colorbar(image, ax=list(axes[row]), shrink=0.85, pad=0.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def hyperparameter_figure(path: Path, params: pd.DataFrame) -> None:
    columns = [("additive_coef", r"Additive coefficient $\nu$")]
    columns.append(("orthogonality_weight", r"Penalty weight $\lambda$"))
    fig, axes = plt.subplots(1, len(columns), figsize=(3.6 * len(columns), 2.8))
    rng = np.random.default_rng(0)
    for ax, (column, title) in zip(axes, columns, strict=True):
        methods = [m for m in ENSEMBLES if params.loc[params["method"] == m, column].notna().any()]
        for i, method in enumerate(methods):
            values = params.loc[params["method"] == method, column].dropna()
            jitter = rng.uniform(-0.18, 0.18, len(values))
            ax.scatter(i + jitter, values, s=6, alpha=0.6, color=METHOD_COLORS[method])
            ax.hlines(values.median(), i - 0.3, i + 0.3, color="black", lw=1.5)
        for bound in SEARCH_BOUNDS[column]:
            ax.axhline(bound, color="grey", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], fontsize=7)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def diagnostic_outputs(
    run_dir: Path,
    out_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    summary: pd.DataFrame,
    df: pd.DataFrame,
    reference: str,
) -> None:
    tasks = df.groupby("dataset")["task"].first().to_dict()
    params = load_best_params(run_dir)
    params.to_csv(out_dir / "best_params.csv", index=False)

    # ----- staged loss and member norms -----
    diag = load_diagnostics(run_dir, tasks)
    staged = staged_loss_ratios(diag)
    staged.to_csv(out_dir / "staged_loss.csv", index=False)
    norms = member_norm_ratios(diag, params)
    norms.to_csv(out_dir / "member_norms.csv", index=False)
    line_panels(
        figures_dir / "staged_loss.pdf",
        staged,
        [
            ("task == 'classification'", "ratio", "Classification: NLL$(m)$ / NLL$(M)$"),
            ("task == 'regression'", "ratio", "Regression: RMSE$(m)$ / RMSE$(M)$"),
        ],
    )
    line_panels(
        figures_dir / "member_norms.pdf",
        norms,
        [
            ("", "ratio", r"$\|f_m(X)\| \,/\, \|f_1(X)\|$"),
            ("", "weighted", r"$w_m \|f_m(X)\| \,/\, \|f_1(X)\|$"),
        ],
    )

    sims, matrices = load_similarities(run_dir)
    sims.to_csv(out_dir / "similarity.csv", index=False)
    similarity_figure(figures_dir / "similarity.pdf", matrices)
    size = int(diag["member"].max())
    sim_means = sims.groupby(["method", "measure"])["mean_offdiag"].mean()
    staged_at = staged.set_index(["task", "method", "member"])["ratio"]
    columns = [(t, m) for t in TASK_METRICS for m in (1, size // 2)]
    body = []
    for method in ENSEMBLES:
        cells = [METHOD_LABELS[method]]
        cells += [f"{staged_at[(t, method, m)]:.3f}" for t, m in columns]
        cells += [f"{sim_means[(method, measure)]:.3f}" for measure in ("cka", "cosine")]
        body.append(" & ".join(cells) + r" \\")
    write_table(
        tables_dir / "ensemble_diagnostics.tex",
        rf"Ensemble diagnostics. Staged loss of the first $m$ members relative to the full "
        rf"ensemble of $M = {size}$ (NLL for classification, RMSE for regression; geometric "
        r"mean over runs and datasets), and the mean off-diagonal pairwise CKA and squared "
        r"cosine similarity of the members' penultimate representations on the outer test "
        r"fold (mean over all runs).",
        "tab:ensemble_diagnostics",
        "lcccccc",
        r"\textbf{Method} & \multicolumn{2}{c}{\textbf{Classification}} & "
        r"\multicolumn{2}{c}{\textbf{Regression}} & \multicolumn{2}{c}{\textbf{Similarity}} \\ "
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} & "
        + " & ".join([f"$L({m})/L({size})$" for _, m in columns] + ["CKA", "Cosine"]),
        body,
    )

    # ----- stability -----
    stability = stability_ratios(summary, reference)
    stability.to_csv(out_dir / "stability.csv", index=False)
    lookup = stability.set_index(["task", "quantity", "method"])["ratio"]
    quantities = [(t, q) for t in TASK_METRICS for q in ("std", "std_fold", "std_seed")]
    methods = ordered_methods(stability["method"].unique())
    cells = {
        c: bold_extreme([float(lookup[(c[0], c[1], m)]) for m in methods], ".3f", lowest=True)
        for c in quantities
    }
    body = [
        " & ".join([METHOD_LABELS[m], *[cells[c][i] for c in quantities]]) + r" \\"
        for i, m in enumerate(methods)
    ]
    quantity_labels = [r"$\sigma$", r"$\sigma_{\text{fold}}$", r"$\bar{\sigma}_{\text{seed}}$"]
    write_table(
        tables_dir / "stability.tex",
        "Stability: geometric mean over datasets of the standard deviation of the primary metric "
        f"(NLL / RMSE) relative to {METHOD_LABELS[reference]} (values below 1 are more stable). "
        r"$\sigma$: over all $25$ fold--seed results; $\sigma_{\text{fold}}$: over the $5$ "
        r"fold means; $\bar{\sigma}_{\text{seed}}$: mean over folds of the standard deviation "
        r"over the $5$ seeds.",
        "tab:stability",
        "lcccccc",
        r"\textbf{Method} & \multicolumn{3}{c}{\textbf{Classification (NLL)}} & "
        r"\multicolumn{3}{c}{\textbf{Regression (RMSE)}} \\ "
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} & " + " & ".join(quantity_labels * 2),
        body,
    )

    # ----- computational cost -----
    cost = cost_summary(run_dir, reference)
    cost.to_csv(out_dir / "cost.csv", index=False)
    body = []
    for _, r in cost.iterrows():
        body.append(
            f"{METHOD_LABELS[r['method']]} & {r['n_params_median']:,.0f} & "
            f"{r['n_params_ratio']:.2f} & {r['epochs_median']:.1f} & {r['epochs_ratio']:.2f} & "
            f"{r['train_seconds_median']:.1f} & {r['train_seconds_ratio']:.2f} \\\\"
        )
    write_table(
        tables_dir / "cost.tex",
        "Computational cost of the final refits: median over all $300$ refits per method, and "
        "geometric mean over datasets of the per-dataset mean relative to "
        f"{METHOD_LABELS[reference]} ($\\times$ref). Training time is the wall-clock time of "
        "the refit on a GPU.",
        "tab:cost",
        "lcccccc",
        r"\textbf{Method} & \multicolumn{2}{c}{\textbf{Trainable parameters}} & "
        r"\multicolumn{2}{c}{\textbf{Refit epochs}} & "
        r"\multicolumn{2}{c}{\textbf{Training time [s]}} \\ "
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} & "
        + " & ".join(["median", r"$\times$ref"] * 3),
        body,
    )

    # ----- selected hyperparameters -----
    hyperparameter_figure(figures_dir / "hyperparameters.pdf", params)
    columns = [
        ("lr", r"$\eta$", sci),
        ("wd", r"$\gamma$", sci),
        ("additive_coef", r"$\nu$", lambda v: f"{v:.3f}"),
        ("orthogonality_weight", r"$\lambda$", sci),
        ("best_epoch", r"$e^\star$", lambda v: f"{v:.0f}"),
    ]
    body = []
    for method in ordered_methods(params["method"].unique()):
        sub = params[params["method"] == method]
        cells = [METHOD_LABELS[method]]
        for column, _, fmt in columns:
            values = sub[column].dropna() if column in sub else pd.Series(dtype=float)
            cells.append(quartiles(values, fmt) if len(values) else "--")
        body.append(" & ".join(cells) + r" \\")
    write_table(
        tables_dir / "hyperparameters.tex",
        r"Selected hyperparameters: median and [first, third] quartile over the $60$ "
        r"dataset--fold pairs. $\eta$: learning rate, $\gamma$: weight decay, $\nu$: additive "
        r"coefficient, $\lambda$: penalty weight, $e^\star$: best inner-validation epoch (the "
        r"refit length). The Stage~1 ResNet uses the fixed $\eta = 10^{-3}$, "
        r"$\gamma = 10^{-4}$.",
        "tab:hyperparameters_selected",
        "lccccc",
        r"\textbf{Method} & " + " & ".join(label for _, label, _ in columns),
        body,
        font=r"\small",
    )

    print("\nStaged loss L(m)/L(M):")
    print(staged.pivot_table(index=["task", "method"], columns="member").round(3).to_string())
    print("\nMean off-diagonal similarity:")
    print(sim_means.unstack().round(3).to_string())
    print("\nStability ratios:")
    print(stability.pivot_table(index="method", columns=["task", "quantity"]).round(3))
    print("\nCost:")
    print(cost.round(2).to_string(index=False))


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run)
    out_dir, tables_dir, figures_dir = (
        run_dir / "analysis",
        resolve(args.tables),
        resolve(args.figures),
    )
    for d in (out_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_metrics(run_dir)
    datasets = {task: task_datasets(df, task) for task in TASK_METRICS}
    methods = ordered_methods(df["method"].unique())
    print(f"{len(df)} rows | methods: {', '.join(methods)}")
    for task, names in datasets.items():
        print(f"{task}: {', '.join(names)}")

    # ----- within-dataset aggregation and fold-level tests -----
    summary = within_dataset(df)
    summary.to_csv(out_dir / "within_dataset.csv", index=False)
    folds = fold_means(df)
    fold_tests = pd.concat(
        [fold_level_tests(folds, b) for b in {args.baseline, *WTL_BASELINES}], ignore_index=True
    )
    fold_tests.to_csv(out_dir / "fold_tests.csv", index=False)
    for task, metrics in TASK_METRICS.items():
        for metric in metrics:
            per_dataset_table(
                tables_dir / f"results_{metric}.tex",
                summary,
                fold_tests,
                task,
                metric,
                datasets[task],
                args.baseline,
            )

    # ----- ranks, Friedman / Iman-Davenport, Nemenyi -----
    rank_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for task, metrics in TASK_METRICS.items():
        for metric in metrics:
            rank_tables[(task, metric)] = rank_table(
                means_table(summary, metric, datasets[task]), metric
            )
    rank_tables[("combined", "primary")] = combined_rank_table(summary, df)

    mean_rank_rows, friedman_rows, nemenyi_rows = [], [], []
    for (scope, metric), ranks in rank_tables.items():
        mean_rank_rows.append({"scope": scope, "metric": metric, **ranks.mean(axis=0).to_dict()})
        friedman_rows.append({"scope": scope, "metric": metric, **friedman(ranks)})
        pairs = nemenyi_pairs(ranks)
        pairs.insert(0, "metric", metric)
        pairs.insert(0, "scope", scope)
        nemenyi_rows.append(pairs)
    mean_ranks = pd.DataFrame(mean_rank_rows)
    friedman_df = pd.DataFrame(friedman_rows)
    mean_ranks.to_csv(out_dir / "mean_ranks.csv", index=False)
    friedman_df.to_csv(out_dir / "friedman.csv", index=False)
    pd.concat(nemenyi_rows).to_csv(out_dir / "nemenyi.csv", index=False)

    columns = [(t, m) for t, ms in TASK_METRICS.items() for m in ms] + [("combined", "primary")]
    col_labels = [METRIC_LABELS[m] for _, m in columns[:-1]] + ["NLL/RMSE"]
    rank_cells = {
        c: bold_extreme([rank_tables[c].mean(axis=0)[m] for m in methods], ".2f", lowest=True)
        for c in columns
    }
    body = [
        " & ".join([METHOD_LABELS[m], *[rank_cells[c][i] for c in columns]]) + r" \\"
        for i, m in enumerate(methods)
    ]
    write_table(
        tables_dir / "mean_ranks.tex",
        "Mean rank of each method (1 = best) over the 6 classification datasets, the 6 "
        "regression datasets, and all 12 datasets (ranked by NLL for classification and RMSE "
        "for regression).",
        "tab:mean_ranks",
        "l" + "c" * len(columns),
        r"\textbf{Method} & \multicolumn{3}{c}{\textbf{Classification}} & "
        r"\multicolumn{3}{c}{\textbf{Regression}} & \textbf{All} \\ "
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-8} & " + " & ".join(col_labels),
        body,
    )

    scope_labels = {
        ("combined", "primary"): "All (NLL/RMSE)",
        **{
            (t, m): f"{TASK_LABELS[t]} {METRIC_LABELS[m]}"
            for t, ms in TASK_METRICS.items()
            for m in ms
        },
    }
    body = []
    for row in friedman_rows:
        key = (row["scope"], row["metric"])
        body.append(
            f"{scope_labels[key]} & {row['N']} & {row['K']} & {row['chi2_F']:.2f} & "
            f"{fmt_p(row['p_chi2'])} & {row['F_F']:.2f} & {fmt_p(row['p_F'])} & "
            f"{row['CD']:.2f} \\\\"
        )
    write_table(
        tables_dir / "friedman.tex",
        r"Friedman test with the Iman--Davenport correction ($F_F$ with $(K-1)$ and "
        r"$(K-1)(N-1)$ degrees of freedom) and the Nemenyi critical difference (CD, "
        rf"$\alpha = {ALPHA}$) of the mean ranks.",
        "tab:friedman",
        "lccccccc",
        r"\textbf{Scope} & $N$ & $K$ & $\chi^2_F$ & $p$ & $F_F$ & $p$ & CD",
        body,
    )
    for scope, metric, title in (
        ("classification", "nll", "Classification (NLL)"),
        ("regression", "rmse", "Regression (RMSE)"),
        ("combined", "primary", "All datasets (NLL / RMSE)"),
    ):
        cd_diagram(figures_dir / f"cd_{scope}.pdf", rank_tables[(scope, metric)], title)

    # ----- geometric-mean loss ratios -----
    gm_rows = []
    ratio_columns: list[tuple[str, str, pd.DataFrame]] = []
    for task, metrics in TASK_METRICS.items():
        for metric in metrics:
            if metric in LOSS_METRICS:
                ratios = loss_ratios(means_table(summary, metric, datasets[task]), args.reference)
                ratio_columns.append((task, metric, ratios))
    ratio_columns.append(("combined", "primary", combined_loss_ratios(summary, df, args.reference)))
    for scope, metric, ratios in ratio_columns:
        gm_rows.append({"scope": scope, "metric": metric, **geometric_mean(ratios).to_dict()})
    gm_df = pd.DataFrame(gm_rows)
    gm_df.to_csv(out_dir / "gm_loss_ratios.csv", index=False)
    gm_cells = [
        bold_extreme([float(row[m]) for m in methods], ".4f", lowest=True)
        for _, row in gm_df.iterrows()
    ]
    body = [
        " & ".join([METHOD_LABELS[m], *[cells[i] for cells in gm_cells]]) + r" \\"
        for i, m in enumerate(methods)
    ]
    gm_labels = [scope_labels[(s, m)] for s, m, _ in ratio_columns]
    write_table(
        tables_dir / "gm_loss_ratios.tex",
        "Geometric mean over datasets of the loss ratio "
        r"$r_{d,c} = \hat{\mu}_{d,c} / \hat{\mu}_{d,\text{ref}}$ relative to "
        f"{METHOD_LABELS[args.reference]} (values below 1 are better).",
        "tab:gm_loss_ratios",
        "l" + "c" * len(gm_labels),
        r"\textbf{Method} & " + " & ".join(rf"\textbf{{{label}}}" for label in gm_labels),
        body,
    )

    # ----- win / tie / loss -----
    wtl = pd.concat(
        [
            win_tie_loss(summary, fold_tests, metric, datasets[task], baseline, ALPHA)
            for baseline in WTL_BASELINES
            for task, metrics in TASK_METRICS.items()
            for metric in metrics
        ],
        ignore_index=True,
    )
    wtl.to_csv(out_dir / "win_tie_loss.csv", index=False)
    metric_order = [m for ms in TASK_METRICS.values() for m in ms]
    body = []
    for baseline in WTL_BASELINES:
        sub = wtl[wtl["baseline"] == baseline]
        body.append(
            rf"\multicolumn{{{len(metric_order) + 1}}}{{l}}{{\textit{{vs.\ "
            rf"{METHOD_LABELS[baseline]}}}}} \\"
        )
        for method in methods:
            if method == baseline:
                continue
            cells = [METHOD_LABELS[method]]
            for metric in metric_order:
                r = sub[(sub["method"] == method) & (sub["metric"] == metric)].iloc[0]
                cells.append(f"{r['wins']}/{r['ties']}/{r['losses']}")
            body.append(" & ".join(cells) + r" \\")
        if baseline != WTL_BASELINES[-1]:
            body.append(r"\midrule")
    write_table(
        tables_dir / "win_tie_loss.tex",
        "Win/tie/loss counts over datasets against the baselines: a win (loss) is a "
        "significantly better (worse) result in the paired $t$-test over the 5 fold means "
        rf"($p < {ALPHA}$), otherwise a tie. 6 datasets per metric.",
        "tab:win_tie_loss",
        "l" + "c" * len(metric_order),
        r"\textbf{Method} & " + " & ".join(rf"\textbf{{{METRIC_LABELS[m]}}}" for m in metric_order),
        body,
    )

    # ----- Wilcoxon signed-rank tests with Holm correction -----
    pairs = [(p, b) for p in PROPOSED for b in WILCOXON_BASELINES if p in methods and b in methods]
    wilcoxon_parts = []
    for task, metrics in TASK_METRICS.items():
        for metric in metrics:
            scores = means_table(summary, metric, datasets[task])
            wilcoxon_parts.append(wilcoxon_holm(scores, pairs, metric, task))
    combined = combined_loss_ratios(summary, df, args.reference)
    wilcoxon_parts.append(wilcoxon_holm(combined, pairs, "rmse", "combined"))
    wilcoxon_df = pd.concat(wilcoxon_parts, ignore_index=True)
    wilcoxon_df.loc[wilcoxon_df["scope"] == "combined", "metric"] = "primary_ratio"
    wilcoxon_df.to_csv(out_dir / "wilcoxon.csv", index=False)

    scopes = [("classification", "nll"), ("regression", "rmse"), ("combined", "primary_ratio")]
    body = []
    for proposed, baseline in pairs:
        cells = [f"{METHOD_LABELS[proposed]} vs.\\ {METHOD_LABELS[baseline]}"]
        for scope, metric in scopes:
            r = wilcoxon_df[
                (wilcoxon_df["scope"] == scope)
                & (wilcoxon_df["metric"] == metric)
                & (wilcoxon_df["proposed"] == proposed)
                & (wilcoxon_df["baseline"] == baseline)
            ].iloc[0]
            cells += [f"{r['n_better']}/{r['N']}", fmt_p(r["p"]), fmt_p(r["p_holm"])]
        body.append(" & ".join(cells) + r" \\")
    write_table(
        tables_dir / "wilcoxon.tex",
        "Two-sided Wilcoxon signed-rank tests of the SFOvTSGM ensembles against the baselines "
        "over datasets: number of datasets on which the SFOvTSGM ensemble is better, raw "
        "$p$-value, and Holm-adjusted $p$-value (family: the 10 comparisons of one column "
        "group). Classification uses NLL and regression RMSE ($N = 6$); the combined test uses "
        f"the loss ratios relative to {METHOD_LABELS[args.reference]} ($N = 12$).",
        "tab:wilcoxon",
        "l" + "ccc" * len(scopes),
        r"\textbf{Comparison} & \multicolumn{3}{c}{\textbf{Classification (NLL)}} & "
        r"\multicolumn{3}{c}{\textbf{Regression (RMSE)}} & "
        r"\multicolumn{3}{c}{\textbf{All (loss ratio)}} \\ "
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} & "
        + " & ".join(["better", "$p$", "$p_{\\text{Holm}}$"] * len(scopes)),
        body,
    )

    # ----- console summary -----
    pd.set_option("display.width", 200)
    print("\nMean ranks:")
    print(mean_ranks.set_index(["scope", "metric"])[methods].round(2).to_string())
    print("\nFriedman / Iman-Davenport:")
    print(
        friedman_df.set_index(["scope", "metric"])[
            ["N", "K", "chi2_F", "p_chi2", "F_F", "p_F", "CD"]
        ]
        .round(4)
        .to_string()
    )
    print(f"\nGeometric-mean loss ratios vs {args.reference}:")
    print(gm_df.set_index(["scope", "metric"])[methods].round(4).to_string())
    print("\nWilcoxon (primary metrics):")
    primary = wilcoxon_df[wilcoxon_df["metric"].isin(["nll", "rmse", "primary_ratio"])]
    primary = primary[~((primary["scope"] == "classification") & (primary["metric"] == "rmse"))]
    print(
        primary[["scope", "proposed", "baseline", "n_better", "N", "p", "p_holm"]]
        .round(4)
        .to_string(index=False)
    )
    diagnostic_outputs(run_dir, out_dir, tables_dir, figures_dir, summary, df, args.reference)
    print(f"\nWritten: {out_dir}, {tables_dir}, {figures_dir}")


if __name__ == "__main__":
    main()
