from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.dataset_tools.dataset_manager import (
    CommonDatasetManager,
    IDatasetManager,
    OpenMLDatasetManager,
)
from src.dataset_tools.openml import read_links
from src.model_tools.model_manager import MODEL_REGISTRY, STAGE1_MODEL, IModelManager

# Specialized DatasetManager classes by dataset name; otherwise chosen by the `source` field.
DATASET_REGISTRY: dict[str, type[IDatasetManager]] = {}

PLACEHOLDERS = ("{project_dir}", "{main_proj_dir}", "{main_porj_dir}")


@dataclass
class TrainingConfig:
    optimizer: str = "AdamW"
    max_epochs: int = 200
    patience: int = 16
    batch_size: int | str = "auto"

    def resolve_batch_size(self, n_samples: int) -> int:
        if self.batch_size == "auto":
            return 512 if n_samples > 50_000 else 256
        return int(self.batch_size)


@dataclass
class HPOConfig:
    stage1_trials: int = 40
    stage2_trials: int = 30
    stage1_lr: float = 1e-3
    stage1_wd: float = 1e-4
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 256])
    ensemble_size: int = 8
    pruner_warmup: int = 10


@dataclass
class CheckpointConfig:
    every_n_epochs: int = 5
    time_budget_hours: float | None = None


@dataclass
class ExperimentConfig:
    run_name: str
    results_dir: Path
    data_dir: Path
    device: str
    save_models: bool
    master_seed: int
    n_outer_folds: int
    seeds: list[int]
    training: TrainingConfig
    hpo: HPOConfig
    checkpoint: CheckpointConfig
    datasets: list[IDatasetManager]
    models: list[IModelManager]
    raw_config: dict[str, Any]

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_name


class ConfigReader:
    def __init__(self, project_root: str | Path | None = None):
        if project_root is None:
            # Resolves repository root (../../ from src/config_readers/)
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

    def resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    def _substitute_placeholders(self, text: str) -> str:
        root_str = str(self.project_root).replace("\\", "/")
        for placeholder in PLACEHOLDERS:
            text = text.replace(placeholder, root_str)
        return text

    def load_raw_config(self, config_path: str | Path) -> dict[str, Any]:
        """Loads and parses a YAML configuration file with path substitution."""
        content = self.resolve_path(config_path).read_text(encoding="utf-8")
        return yaml.safe_load(self._substitute_placeholders(content)) or {}

    def get_dataset_specs(self, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Merges datasets from `openml_links` with the inline `datasets` section and keeps
        those named in `run_datasets` (all of them if it is omitted), in that order."""
        specs: dict[str, dict[str, Any]] = {}
        if config.get("openml_links"):
            for name, link in read_links(self.resolve_path(config["openml_links"])).items():
                specs[name] = {"source": "openml", "link": link}
        for name, ds_cfg in (config.get("datasets") or {}).items():
            specs[name] = {**specs.get(name, {}), **(ds_cfg or {})}

        selected = config.get("run_datasets")
        if selected is None:
            return specs
        unknown = [name for name in selected if name not in specs]
        if unknown:
            raise ValueError(
                f"Unknown dataset(s) in run_datasets: {unknown}. "
                f"Add them to openml_links or `datasets`. Available: {list(specs)}"
            )
        return {name: specs[name] for name in dict.fromkeys(selected)}

    def get_dataset_managers(self, config: dict[str, Any]) -> list[IDatasetManager]:
        data_dir = self.resolve_path(config.get("data_dir", "data"))
        managers: list[IDatasetManager] = []
        for name, spec in self.get_dataset_specs(config).items():
            source = spec.get("source", "csv")
            max_samples = spec.get("max_samples")
            if name in DATASET_REGISTRY:
                managers.append(DATASET_REGISTRY[name](name=name, **spec))
            elif source == "openml":
                if "link" not in spec:
                    raise ValueError(f"OpenML dataset {name} needs a `link` (or openml_links)")
                managers.append(OpenMLDatasetManager(name, spec["link"], data_dir, max_samples))
            elif source == "csv":
                managers.append(
                    CommonDatasetManager(
                        name=name,
                        dataset_path=self.resolve_path(spec["dataset_path"]),
                        label_column=spec.get("label_column", "target"),
                        task_type=spec.get("task_type", "classification"),
                        categorical_columns=spec.get("categorical_columns"),
                        max_samples=max_samples,
                    )
                )
            else:
                raise ValueError(f"Unknown source {source!r} for dataset {name}")
        return managers

    def get_model_managers(self, config: dict[str, Any]) -> list[IModelManager]:
        """Stage 1 (`single_resnet`) first, then the configured Stage 2 methods."""
        names = [STAGE1_MODEL]
        for m_cfg in config.get("models", []):
            name = m_cfg if isinstance(m_cfg, str) else m_cfg.get("name", "")
            if name not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model name '{name}'. Available models: {list(MODEL_REGISTRY)}"
                )
            if name not in names:
                names.append(name)
        return [MODEL_REGISTRY[name](name) for name in names]

    def read_experiment_config(self, config_path: str | Path) -> ExperimentConfig:
        raw = self.load_raw_config(config_path)
        training = TrainingConfig(**(raw.get("training") or {}))
        hpo = HPOConfig(**(raw.get("hpo") or {}))
        checkpoint = CheckpointConfig(**(raw.get("checkpoint") or {}))

        if training.batch_size != "auto":
            if int(training.batch_size) < max(hpo.hidden_dims):
                raise ValueError(
                    f"batch_size={training.batch_size} must be >= max(hidden_dims)="
                    f"{max(hpo.hidden_dims)} so batch similarity estimates are not rank-deficient"
                )
        elif max(hpo.hidden_dims) > 256:
            raise ValueError("batch_size: auto gives 256 on small datasets; reduce hidden_dims")

        return ExperimentConfig(
            run_name=str(raw.get("run_name", "main")),
            results_dir=self.resolve_path(raw.get("results_dir", "results")),
            data_dir=self.resolve_path(raw.get("data_dir", "data")),
            device=str(raw.get("device", "auto")),
            save_models=bool(raw.get("save_models", False)),
            master_seed=int(raw.get("master_seed", 42)),
            n_outer_folds=int(raw.get("n_outer_folds", 5)),
            seeds=[int(s) for s in raw.get("refit_seeds", [42])],
            training=training,
            hpo=hpo,
            checkpoint=checkpoint,
            datasets=self.get_dataset_managers(raw),
            models=self.get_model_managers(raw),
            raw_config=raw,
        )
