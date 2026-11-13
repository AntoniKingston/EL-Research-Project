"""Runs the experiment suite: per outer fold, two-stage Optuna HPO on the inner split, then
refits of every method on the outer training partition with each seed, evaluated on the outer
test fold. Re-running the same command resumes an interrupted run.

    uv run python -m src.scripts.train_orchestrator --config configs/experiment_config.yaml
"""

import argparse
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
from src.run_tools.resume import (
    RunDirectory,
    StopController,
    load_study,
    n_done_trials,
    run_study,
)

STAGES = ("hpo", "refit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--datasets", nargs="*", help="subset of configured datasets")
    parser.add_argument("--models", nargs="*", help="subset of configured Stage 2 models")
    parser.add_argument("--stages", default="hpo,refit", help="comma-separated: hpo,refit")
    parser.add_argument(
        "--fresh", action="store_true", help="move the existing run directory aside and restart"
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
        return load_study(
            self.run.optuna_url,
            study_name(self.dataset.name, self.fold, manager.name),
            self.cfg.hpo.pruner_warmup,
        )

    def _selection(self, study: optuna.Study) -> tuple[dict[str, Any], int]:
        best = study.best_trial
        return best.params, int(best.user_attrs["best_epoch"])

    def run_hpo(self) -> None:
        inner = prepare_fold(self.dataset, self.split.inner_train_idx, self.split.inner_val_idx)
        ctx = make_context(self.cfg, self.dataset, inner, self.device)
        backbone: dict[str, Any] | None = None
        for manager in self.models:
            study = self._study(manager)
            n_trials = self._n_trials(manager)
            done = n_done_trials(study)
            if done < n_trials:
                print(f"  [hpo] {manager.name}: trials {done}/{n_trials}")

            def objective(trial: optuna.Trial, seed: int, m=manager, bb=backbone) -> float:
                return m.hyperparameters_study(trial, ctx, inner, bb, seed, self.stop.should_stop)

            base_seed = self.cfg.master_seed + 1000 * self.fold
            if not run_study(study, objective, n_trials, base_seed, self.stop.should_stop):
                self.stop.exit()
            params, best_epoch = self._selection(study)
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
        if not pending:
            return
        selections: dict[str, tuple[dict[str, Any], int]] = {}
        for manager in self.models:
            study = self._study(manager)
            if n_done_trials(study) < self._n_trials(manager):
                print(f"  [refit] skipping fold {self.fold}: HPO of {manager.name} unfinished")
                return
            selections[manager.name] = self._selection(study)
        backbone = next(selections[m.name][0] for m in self.models if m.is_stage1)

        outer = prepare_fold(self.dataset, self.split.outer_train_idx, self.split.outer_test_idx)
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
        print(f"  [refit] {manager.name} seed={seed}: {summary}")


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    reader = ConfigReader()
    cfg = reader.read_experiment_config(args.config)
    unknown = set(args.datasets or ()) - {d.name for d in cfg.datasets}
    if unknown:
        raise SystemExit(f"--datasets {sorted(unknown)} not selected in {args.config}")
    datasets = [d for d in cfg.datasets if not args.datasets or d.name in args.datasets]
    models = [m for m in cfg.models if m.is_stage1 or not args.models or m.name in args.models]

    run = RunDirectory(
        cfg.run_dir, cfg.raw_config, reader.get_dataset_specs(cfg.raw_config), fresh=args.fresh
    )
    stop = StopController(cfg.checkpoint.time_budget_hours)
    device = resolve_device(cfg.device)
    print(f"Run directory: {run.path} | device: {device} | stages: {args.stages}")

    for dataset_manager in datasets:
        dataset = dataset_manager.load(seed=cfg.master_seed)
        splits = DatasetService.get_nested_splits(
            dataset.X, dataset.y, cfg.n_outer_folds, dataset.is_classification, cfg.master_seed
        )
        DatasetService.save_splits(run.splits_path(dataset.name), splits)
        print(f"\n=== {dataset.name} ({dataset.task_type}, n={len(dataset.y)}) ===")
        for split in splits:
            print(f" fold {split.fold}")
            runner = FoldRunner(cfg, run, stop, device, dataset, split, models)
            if "hpo" in args.stages:
                runner.run_hpo()
            if "refit" in args.stages:
                runner.run_refits()
    print("\nAll requested units finished.")


if __name__ == "__main__":
    main()
