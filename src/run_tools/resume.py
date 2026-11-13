"""Crash/time-limit tolerant bookkeeping for an experiment run.

Resumable units: Optuna trial -> study (dataset, fold, method) -> refit (dataset, fold, method,
seed). Studies live in SQLite, finished refits in an append-only ledger, and refit training
state in per-unit checkpoints. Everything assumes a single worker process per run directory.
"""

import csv
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import yaml
from optuna.storages import RDBStorage
from optuna.trial import TrialState

from src.models.base import TrainingInterrupted

EXIT_CODE_INTERRUPTED = 3
# Config sections whose change would make previously stored results inconsistent.
GUARDED_KEYS = ("master_seed", "n_outer_folds", "training", "hpo")
DATASET_GUARDED_KEYS = (
    "source",
    "link",
    "dataset_path",
    "label_column",
    "task_type",
    "categorical_columns",
    "max_samples",
)
DONE_STATES = (TrialState.COMPLETE, TrialState.PRUNED)


class ConfigMismatchError(RuntimeError):
    pass


def _fingerprint(raw_config: dict[str, Any], dataset_specs: dict[str, dict]) -> dict[str, Any]:
    datasets = {
        name: {k: spec.get(k) for k in DATASET_GUARDED_KEYS if spec.get(k) is not None}
        for name, spec in dataset_specs.items()
    }
    return {
        "guarded": {k: raw_config.get(k) for k in GUARDED_KEYS},
        "datasets": datasets,
    }


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _append_csv(path: Path, rows: Iterable[dict], columns: tuple[str, ...]) -> None:
    rows = list(rows)
    if not rows:
        return
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


class RunDirectory:
    """`results/<run_name>/`: config snapshot + guard, Optuna DB, result CSVs and ledger."""

    METRIC_COLUMNS = ("dataset", "task", "fold", "method", "seed", "metric", "value")
    DIAGNOSTIC_COLUMNS = ("dataset", "fold", "method", "seed", "member", "metric", "value")
    PROGRESS_COLUMNS = ("dataset", "fold", "method", "seed", "timestamp")

    def __init__(
        self,
        path: Path,
        raw_config: dict[str, Any],
        dataset_specs: dict[str, dict],
        fresh: bool = False,
    ):
        self.path = path
        if fresh and path.exists():
            backup = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
            shutil.move(path, backup)
            print(f"--fresh: previous run moved to {backup}")
        path.mkdir(parents=True, exist_ok=True)
        self._guard(raw_config, dataset_specs)
        self._done = self._load_done()

    # ---------- config guard ----------
    def _guard(self, raw_config: dict[str, Any], dataset_specs: dict[str, dict]) -> None:
        current = _fingerprint(raw_config, dataset_specs)
        fp_path = self.path / "config_fingerprint.json"
        if not fp_path.exists():
            (self.path / "config.yaml").write_text(yaml.safe_dump(raw_config, sort_keys=False))
            fp_path.write_text(json.dumps({**current, "hash": _hash(current)}, indent=2))
            return

        saved = json.loads(fp_path.read_text())
        problems = [
            f"{k}: {saved['guarded'].get(k)!r} -> {current['guarded'].get(k)!r}"
            for k in GUARDED_KEYS
            if saved["guarded"].get(k) != current["guarded"].get(k)
        ]
        for name in set(saved["datasets"]) & set(current["datasets"]):
            if saved["datasets"][name] != current["datasets"][name]:
                old, new = saved["datasets"][name], current["datasets"][name]
                problems.append(f"datasets.{name}: {old} -> {new}")
        if problems:
            raise ConfigMismatchError(
                f"Config differs from the one stored in {self.path} in result-affecting "
                "settings:\n  " + "\n  ".join(problems) + "\nUse a new run_name or --fresh."
            )
        # Scope changes (new datasets/models/seeds) are fine; remember new dataset specs.
        merged = {**saved["datasets"], **current["datasets"]}
        saved["datasets"] = merged
        fp_path.write_text(json.dumps(saved, indent=2))

    # ---------- paths ----------
    @property
    def optuna_url(self) -> str:
        return f"sqlite:///{self.path / 'optuna.db'}"

    def unit_dir(self, dataset: str) -> Path:
        return self.path / dataset

    def checkpoint_path(self, dataset: str, fold: int, method: str, seed: int) -> Path:
        return self.path / "checkpoints" / dataset / f"{method}_f{fold}_s{seed}.pt"

    def splits_path(self, dataset: str) -> Path:
        return self.unit_dir(dataset) / "splits.npz"

    def similarity_path(self, dataset: str, fold: int, method: str, seed: int) -> Path:
        return self.unit_dir(dataset) / "similarity" / f"{method}_f{fold}_s{seed}.npz"

    def model_path(self, dataset: str, fold: int, method: str, seed: int) -> Path:
        return self.unit_dir(dataset) / "models" / f"{method}_f{fold}_s{seed}.pt"

    def best_params_path(self, dataset: str, fold: int, method: str) -> Path:
        return self.unit_dir(dataset) / "best_params" / f"{method}_f{fold}.json"

    # ---------- ledger ----------
    def _load_done(self) -> set[tuple[str, int, str, int]]:
        path = self.path / "progress.csv"
        if not path.exists():
            return set()
        df = pd.read_csv(path)
        rows = zip(df["dataset"], df["fold"], df["method"], df["seed"], strict=True)
        return {(str(d), int(f), str(m), int(s)) for d, f, m, s in rows}

    def is_done(self, dataset: str, fold: int, method: str, seed: int) -> bool:
        return (dataset, fold, method, seed) in self._done

    def record_refit(
        self,
        dataset: str,
        fold: int,
        method: str,
        seed: int,
        metric_rows: list[dict],
        diagnostic_rows: list[dict],
    ) -> None:
        """Appends results first, then marks the unit done (both fsync'ed)."""
        key = {"dataset": dataset, "fold": fold, "method": method, "seed": seed}
        _append_csv(
            self.path / "metrics.csv", [{**key, **r} for r in metric_rows], self.METRIC_COLUMNS
        )
        _append_csv(
            self.path / "diagnostics.csv",
            [{**key, **r} for r in diagnostic_rows],
            self.DIAGNOSTIC_COLUMNS,
        )
        _append_csv(
            self.path / "progress.csv",
            [{**key, "timestamp": datetime.now().isoformat(timespec="seconds")}],
            self.PROGRESS_COLUMNS,
        )
        self._done.add((dataset, fold, method, seed))

    def write_json(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, default=str))
        os.replace(tmp, path)


def load_results(path: str | Path) -> pd.DataFrame:
    """Reads metrics.csv / diagnostics.csv, dropping rows duplicated by a crash between the
    results write and the ledger write (the re-run's rows are kept)."""
    df = pd.read_csv(path)
    key = [c for c in df.columns if c not in ("value", "task")]
    return df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


# ---------- graceful stop ----------
class StopController:
    """Sets a stop flag on SIGTERM / SIGUSR1 (e.g. SLURM pre-timeout) / first Ctrl+C, or when
    the time budget is exceeded. A second Ctrl+C aborts immediately."""

    def __init__(self, time_budget_hours: float | None = None):
        self.deadline = (
            time.monotonic() + time_budget_hours * 3600 if time_budget_hours is not None else None
        )
        self.reason: str | None = None
        signals = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, "SIGUSR1"):
            signals.append(signal.SIGUSR1)
        for sig in signals:
            signal.signal(sig, self._handle)

    def _handle(self, signum, frame) -> None:
        if self.reason is not None and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.reason = f"signal {signal.Signals(signum).name}"
        print(f"\n[stop] {self.reason} received; finishing the current epoch and checkpointing")

    def should_stop(self) -> bool:
        if self.reason is None and self.deadline is not None and time.monotonic() > self.deadline:
            self.reason = "time budget exhausted"
        return self.reason is not None

    def exit(self) -> None:
        print(f"[stop] Stopped ({self.reason}). Re-run the same command to resume.")
        sys.exit(EXIT_CODE_INTERRUPTED)


# ---------- Optuna ----------
def load_study(storage_url: str, name: str, warmup: int) -> optuna.Study:
    storage = RDBStorage(storage_url, engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=warmup),
        load_if_exists=True,
    )
    # Single worker: anything still RUNNING belongs to a crashed process. Fail it and re-queue
    # its parameters so the same configuration is evaluated again.
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
        study.enqueue_trial(trial.params, skip_if_exists=False)
    return study


def n_done_trials(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False, states=DONE_STATES))


def run_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial, int], float],
    n_trials: int,
    base_seed: int,
    should_stop: Callable[[], bool],
) -> bool:
    """Runs trials until `n_trials` are COMPLETE/PRUNED. Returns False if stopped early.

    Trial k (k = number of finished trials before it) uses sampler seed `base_seed + k` and
    model seed `base_seed + k`, so an interrupted-and-resumed study reproduces the trials of
    an uninterrupted one. `objective(trial, seed)` must return the validation loss.
    """
    while (done := n_done_trials(study)) < n_trials:
        if should_stop():
            return False
        study.sampler = optuna.samplers.TPESampler(seed=base_seed + done)
        try:
            study.optimize(lambda t, s=base_seed + done: objective(t, s), n_trials=1)
        except TrainingInterrupted:
            failed = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
            if failed:
                study.enqueue_trial(failed[-1].params, skip_if_exists=False)
            return False
    return True
