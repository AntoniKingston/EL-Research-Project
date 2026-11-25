"""Removes all HPO studies and results of the given methods from a run, so they are re-run on the
next resume while the other methods' results are kept.

    uv run python -m src.scripts.purge_methods --run results/main \\
        --methods ensemble_none sym_cka sym_cos sfo_cka sfo_cos

The run directory is first copied to `<run>.bak-<timestamp>`. Also marks the run as using the
current ensemble training objective, so the config guard accepts it again.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd

from src.model_tools.model_manager import STAGE1_MODEL
from src.run_tools.resume import ENSEMBLE_TRAINING

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_CSVS = ("metrics.csv", "diagnostics.csv", "progress.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="run directory, e.g. results/main")
    parser.add_argument("--methods", nargs="+", required=True, help="method names to purge")
    parser.add_argument("--no-backup", action="store_true", help="skip the backup copy")
    return parser.parse_args()


def purge_studies(run_dir: Path, methods: set[str]) -> int:
    db = run_dir / "optuna.db"
    if not db.exists():
        return 0
    storage = f"sqlite:///{db}"
    names = [
        s.study_name
        for s in optuna.get_all_study_summaries(storage, include_best_trial=False)
        if s.study_name.rsplit("/", 1)[-1] in methods
    ]
    for name in names:
        optuna.delete_study(study_name=name, storage=storage)
    return len(names)


def purge_csvs(run_dir: Path, methods: set[str]) -> dict[str, int]:
    removed = {}
    for name in RESULT_CSVS:
        path = run_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        keep = ~df["method"].isin(methods)
        removed[name] = int((~keep).sum())
        tmp = path.with_name(path.name + ".tmp")
        df[keep].to_csv(tmp, index=False)
        os.replace(tmp, path)
    return removed


def purge_files(run_dir: Path, methods: set[str]) -> int:
    patterns = []
    for method in methods:
        patterns += [
            f"*/best_params/{method}_f*.json",
            f"*/similarity/{method}_f*_s*.npz",
            f"*/models/{method}_f*_s*.pt",
            f"checkpoints/*/{method}_f*_s*.pt",
        ]
    paths = {p for pattern in patterns for p in run_dir.glob(pattern)}
    for path in paths:
        path.unlink()
    return len(paths)


def mark_training(run_dir: Path) -> None:
    path = run_dir / "config_fingerprint.json"
    if not path.exists():
        return
    fingerprint = json.loads(path.read_text())
    fingerprint["ensemble_training"] = ENSEMBLE_TRAINING
    path.write_text(json.dumps(fingerprint, indent=2))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    run_dir = run_dir if run_dir.is_absolute() else PROJECT_ROOT / run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    methods = set(args.methods)
    if STAGE1_MODEL in methods:
        raise SystemExit(f"Refusing to purge {STAGE1_MODEL}: every Stage 2 method depends on it.")

    if not args.no_backup:
        backup = run_dir.with_name(f"{run_dir.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copytree(run_dir, backup)
        print(f"Backup: {backup}")

    n_studies = purge_studies(run_dir, methods)
    rows = purge_csvs(run_dir, methods)
    n_files = purge_files(run_dir, methods)
    if (run_dir / "analysis").exists():
        shutil.rmtree(run_dir / "analysis")
    mark_training(run_dir)

    print(f"Purged {', '.join(sorted(methods))} from {run_dir}:")
    print(f"  {n_studies} Optuna studies")
    for name, n in rows.items():
        print(f"  {n} rows from {name}")
    print(f"  {n_files} files (best params, similarity matrices, models, checkpoints)")
    print(f"  marked ensemble_training = {ENSEMBLE_TRAINING!r}")


if __name__ == "__main__":
    main()
