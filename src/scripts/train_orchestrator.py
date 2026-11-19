"""Runs the experiment suite: per outer fold, two-stage Optuna HPO on the inner split, then
refits of every method on the outer training partition with each seed, evaluated on the outer
test fold. Re-running the same command resumes an interrupted run.

    uv run python -m src.scripts.train_orchestrator --config configs/experiment_config.yaml
"""

import argparse
import logging
import os
import platform
import socket
import time
from typing import Any

import numpy as np
import optuna
import torch

from src.config_readers import ConfigReader, ExperimentConfig
from src.dataset_tools.dataset_service import DatasetService
from src.dataset_tools.domain import Dataset, DatasetSplits
from src.dataset_tools.preprocessing import FoldData, prepare_fold
from src.model_tools.evaluation import (
    compute_metrics,
    count_parameters,
    ensemble_diagnostics,
    similarity_matrices,
)
from src.model_tools.model_manager import IModelManager
from src.model_tools.utils import OPTIMIZERS, RunContext, resolve_device
from src.models.base import TrainingInterrupted
from src.run_tools.logging_setup import (
    VERBOSE,
    add_log_file,
    dump_stacks_periodically,
    set_stage,
    setup_logging,
    start_heartbeat,
    timed,
)
from src.run_tools.resume import (
    RunDirectory,
    StopController,
    load_study,
    n_done_trials,
    run_study,
)

STAGES = ("data", "hpo", "refit")

# Named explicitly: under `python -m` this module's __name__ is "__main__".
log = logging.getLogger("src.orchestrator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--datasets", nargs="*", help="subset of configured datasets")
    parser.add_argument("--models", nargs="*", help="subset of configured Stage 2 models")
    parser.add_argument(
        "--stages",
        default="hpo,refit",
        help="comma-separated subset of data,hpo,refit; `data` alone only downloads/caches the "
        "datasets and writes the splits (run it on a node with internet access)",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="move the existing run directory aside and restart"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v: per-trial / per-refit progress; -vv: per-epoch losses and in-epoch heartbeat",
    )
    parser.add_argument(
        "--heartbeat-minutes",
        type=float,
        default=10.0,
        help="log the current stage, memory use and time since the last update every N "
        "minutes from a background thread (0 disables)",
    )
    parser.add_argument(
        "--dump-stacks-minutes",
        type=float,
        default=0.0,
        help="dump the Python stacks of all threads to stderr every N minutes (0 disables); "
        "SIGUSR2 dumps them on demand",
    )
    args = parser.parse_args()
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if unknown := set(args.stages) - set(STAGES):
        parser.error(f"unknown stages {unknown}; choose from {STAGES}")
    return args


def make_context(
    cfg: ExperimentConfig, dataset: Dataset, fold_data: FoldData, device: str
) -> RunContext:
    return RunContext(
        input_dim=fold_data.input_dim,
        output_dim=dataset.n_classes if dataset.is_classification else 1,
        is_classification=dataset.is_classification,
        device=device,
        optimizer=OPTIMIZERS[cfg.training.optimizer],
        batch_size=cfg.training.resolve_batch_size(len(dataset.y)),
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
        stage1_lr=cfg.hpo.stage1_lr,
        stage1_wd=cfg.hpo.stage1_wd,
        hidden_dims=cfg.hpo.hidden_dims,
        ensemble_size=cfg.hpo.ensemble_size,
    )


def study_name(dataset: str, fold: int, method: str) -> str:
    return f"{dataset}/fold{fold}/{method}"


class FoldRunner:
    """HPO + refits for one (dataset, outer fold)."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        run: RunDirectory,
        stop: StopController,
        device: str,
        dataset: Dataset,
        split: DatasetSplits,
        models: list[IModelManager],
    ):
        self.cfg, self.run, self.stop, self.device = cfg, run, stop, device
        self.dataset, self.split, self.models = dataset, split, models
        self.fold = split.fold

    def _n_trials(self, manager: IModelManager) -> int:
        return self.cfg.hpo.stage1_trials if manager.is_stage1 else self.cfg.hpo.stage2_trials

    def _study(self, manager: IModelManager) -> optuna.Study:
        start = time.perf_counter()
        study = load_study(
            self.run.optuna_url,
            study_name(self.dataset.name, self.fold, manager.name),
            self.cfg.hpo.pruner_warmup,
        )
        log.log(
            VERBOSE, "  study %s opened in %.1f s", study.study_name, time.perf_counter() - start
        )
        return study

    def _selection(self, study: optuna.Study) -> tuple[dict[str, Any], int]:
        best = study.best_trial
        return best.params, int(best.user_attrs["best_epoch"])

    def _prepare(self, kind: str, train_idx: np.ndarray, eval_idx: np.ndarray) -> FoldData:
        with timed(log, f"  preprocessing {kind} split", VERBOSE):
            data = prepare_fold(self.dataset, train_idx, eval_idx)
        log.log(
            VERBOSE,
            "  %s split: train %s, eval %s, input_dim %d",
            kind,
            data.X_train.shape,
            data.X_eval.shape,
            data.input_dim,
        )
        return data

    def run_hpo(self) -> None:
        set_stage(dataset=self.dataset.name, fold=self.fold, phase="hpo")
        inner = self._prepare("inner", self.split.inner_train_idx, self.split.inner_val_idx)
        ctx = make_context(self.cfg, self.dataset, inner, self.device)
        backbone: dict[str, Any] | None = None
        for manager in self.models:
            set_stage(method=manager.name)
            start = time.perf_counter()
            study = self._study(manager)
            n_trials = self._n_trials(manager)
            done = n_done_trials(study)
            log.info("  [hpo] %s: trials %d/%d", manager.name, done, n_trials)

            def objective(trial: optuna.Trial, seed: int, m=manager, bb=backbone) -> float:
                return m.hyperparameters_study(trial, ctx, inner, bb, seed, self.stop.should_stop)

            base_seed = self.cfg.master_seed + 1000 * self.fold
            if not run_study(study, objective, n_trials, base_seed, self.stop.should_stop):
                self.stop.exit()
            params, best_epoch = self._selection(study)
            if done < n_trials:
                log.info(
                    "  [hpo] %s: done in %.1f s, best value %.4g (trial %d, best_epoch %d)",
                    manager.name,
                    time.perf_counter() - start,
                    study.best_value,
                    study.best_trial.number,
                    best_epoch,
                )
            if manager.is_stage1:
                backbone = params
            self.run.write_json(
                self.run.best_params_path(self.dataset.name, self.fold, manager.name),
                {
                    "params": params,
                    "backbone": None if manager.is_stage1 else backbone,
                    "best_epoch": best_epoch,
                    "best_value": study.best_value,
                },
            )

    def run_refits(self) -> None:
        pending = [
            (m, seed)
            for m in self.models
            for seed in self.cfg.seeds
            if not self.run.is_done(self.dataset.name, self.fold, m.name, seed)
        ]
        set_stage(dataset=self.dataset.name, fold=self.fold, phase="refit")
        n_units = len(self.models) * len(self.cfg.seeds)
        log.info("  [refit] %d/%d units pending", len(pending), n_units)
        if not pending:
            return
        selections: dict[str, tuple[dict[str, Any], int]] = {}
        for manager in self.models:
            study = self._study(manager)
            if n_done_trials(study) < self._n_trials(manager):
                log.warning(
                    "  [refit] skipping fold %d: HPO of %s unfinished", self.fold, manager.name
                )
                return
            selections[manager.name] = self._selection(study)
        backbone = next(selections[m.name][0] for m in self.models if m.is_stage1)

        outer = self._prepare("outer", self.split.outer_train_idx, self.split.outer_test_idx)
        ctx = make_context(self.cfg, self.dataset, outer, self.device)
        for manager, seed in pending:
            params, best_epoch = selections[manager.name]
            self._refit_one(manager, seed, params, best_epoch, backbone, outer, ctx)
            if self.stop.should_stop():
                self.stop.exit()

    def _refit_one(
        self,
        manager: IModelManager,
        seed: int,
        params: dict[str, Any],
        epochs: int,
        backbone: dict[str, Any],
        outer: FoldData,
        ctx: RunContext,
    ) -> None:
        name, fold = self.dataset.name, self.fold
        set_stage(method=manager.name, seed=seed)
        log.log(
            VERBOSE,
            "  [refit] %s seed=%d: training %d epochs, params %s",
            manager.name,
            seed,
            epochs,
            params,
        )
        start = time.perf_counter()
        try:
            model = manager.train_optimal_model(
                params,
                None if manager.is_stage1 else backbone,
                ctx,
                outer,
                seed,
                epochs,
                checkpoint_path=self.run.checkpoint_path(name, fold, manager.name, seed),
                checkpoint_every=self.cfg.checkpoint.every_n_epochs,
                should_stop=self.stop.should_stop,
            )
        except TrainingInterrupted:
            self.stop.exit()
            raise
        train_seconds = time.perf_counter() - start

        metrics = compute_metrics(model, outer)
        metrics |= {
            "n_params": count_parameters(model),
            "epochs": epochs,
            "train_seconds": train_seconds,
        }
        task = "classification" if self.dataset.is_classification else "regression"
        metric_rows = [{"task": task, "metric": k, "value": v} for k, v in metrics.items()]
        diagnostic_rows = ensemble_diagnostics(model, outer)

        if sims := similarity_matrices(model, outer.X_eval):
            path = self.run.similarity_path(name, fold, manager.name, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, cka=sims["cka"], cosine=sims["cosine"])
        if self.cfg.save_models:
            path = self.run.model_path(name, fold, manager.name, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)

        self.run.record_refit(name, fold, manager.name, seed, metric_rows, diagnostic_rows)
        summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "n_params")
        log.info("  [refit] %s seed=%d: %s", manager.name, seed, summary)


def log_environment(args: argparse.Namespace, device: str) -> None:
    log.info(
        "host %s | SLURM job %s | pid %d | cwd %s",
        socket.gethostname(),
        os.environ.get("SLURM_JOB_ID", "-"),
        os.getpid(),
        os.getcwd(),
    )
    log.info(
        "python %s | torch %s | optuna %s | numpy %s",
        platform.python_version(),
        torch.__version__,
        optuna.__version__,
        np.__version__,
    )
    gpu = torch.cuda.get_device_name(device) if device.startswith("cuda") else "-"
    log.info(
        "device %s (%s) | torch threads %d | OMP_NUM_THREADS %s | SLURM_CPUS_PER_TASK %s",
        device,
        gpu,
        torch.get_num_threads(),
        os.environ.get("OMP_NUM_THREADS", "-"),
        os.environ.get("SLURM_CPUS_PER_TASK", "-"),
    )
    log.info(
        "config %s | stages %s | verbosity %d | heartbeat every %g min",
        args.config,
        ",".join(args.stages),
        args.verbose,
        args.heartbeat_minutes,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    start_heartbeat(args.heartbeat_minutes)
    dump_stacks_periodically(args.dump_stacks_minutes)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    reader = ConfigReader()
    with timed(log, f"reading config {args.config}", VERBOSE):
        cfg = reader.read_experiment_config(args.config)
    unknown = set(args.datasets or ()) - {d.name for d in cfg.datasets}
    if unknown:
        raise SystemExit(f"--datasets {sorted(unknown)} not selected in {args.config}")
    datasets = [d for d in cfg.datasets if not args.datasets or d.name in args.datasets]
    models = [m for m in cfg.models if m.is_stage1 or not args.models or m.name in args.models]

    with timed(log, f"opening run directory {cfg.run_dir}", VERBOSE):
        run = RunDirectory(
            cfg.run_dir, cfg.raw_config, reader.get_dataset_specs(cfg.raw_config), fresh=args.fresh
        )
    log_path = add_log_file(run.path / "logs")
    stop = StopController(cfg.checkpoint.time_budget_hours)
    device = resolve_device(cfg.device)
    log.info("Run directory: %s | log file: %s", run.path, log_path)
    log_environment(args, device)
    log.info("datasets: %s", ", ".join(d.name for d in datasets))
    log.info("models: %s", ", ".join(m.name for m in models))

    for dataset_manager in datasets:
        set_stage(dataset=dataset_manager.name, phase="data")
        with timed(log, f"loading dataset {dataset_manager.name}"):
            dataset = dataset_manager.load(seed=cfg.master_seed)
        with timed(log, "building splits", VERBOSE):
            splits = DatasetService.get_nested_splits(
                dataset.X, dataset.y, cfg.n_outer_folds, dataset.is_classification, cfg.master_seed
            )
            DatasetService.save_splits(run.splits_path(dataset.name), splits)
        log.info(
            "=== %s (%s, n=%d, %d numerical + %d categorical features) ===",
            dataset.name,
            dataset.task_type,
            len(dataset.y),
            len(dataset.numerical),
            len(dataset.categorical),
        )
        if not {"hpo", "refit"} & set(args.stages):
            continue
        for split in splits:
            log.info(" fold %d", split.fold)
            runner = FoldRunner(cfg, run, stop, device, dataset, split, models)
            if "hpo" in args.stages:
                runner.run_hpo()
            if "refit" in args.stages:
                runner.run_refits()
    log.info("All requested units finished.")


if __name__ == "__main__":
    main()
