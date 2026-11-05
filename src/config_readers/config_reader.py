from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.dataset_tools.dataset_manager import CommonDatasetManager, IDatasetManager
from src.dataset_tools.domain import TaskType
from src.model_tools.model_manager import (
    BatchEnsembleManager,
    EnsembleResnetManager,
    IModelManager,
    SingleResnetManager,
    TangosManager,
)

# Registry mapping model names directly to their manager classes
MODEL_REGISTRY: dict[str, type[IModelManager]] = {
    "single_resnet": SingleResnetManager,
    "tangos": TangosManager,
    "batchensemble": BatchEnsembleManager,
    "ensemble_resnet": EnsembleResnetManager,
}

# Registry mapping dataset names to specialized DatasetManager classes.
# If a dataset name is not registered, CommonDatasetManager is used by default.
DATASET_REGISTRY: dict[str, type[IDatasetManager]] = {}


@dataclass
class ExperimentConfig:
    datasets: list[IDatasetManager]
    models: list[IModelManager]
    seeds: list[int]
    raw_config: dict[str, Any]


class ConfigReader:
    def __init__(self, project_root: str | Path | None = None):
        if project_root is None:
            # Resolves repository root (../../ from src/config_readers/)
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

    def _substitute_placeholders(self, text: str) -> str:
        """Substitutes {main_porj_dir} and {main_proj_dir} with the actual project root path."""
        root_str = str(self.project_root).replace("\\", "/")
        return text.replace("{main_porj_dir}", root_str).replace("{main_proj_dir}", root_str)

    def load_raw_config(self, config_path: str | Path) -> dict[str, Any]:
        """Loads and parses a YAML configuration file with path substitution."""
        with open(config_path, encoding="utf-8") as f:
            content = f.read()
        substituted_content = self._substitute_placeholders(content)
        return yaml.safe_load(substituted_content) or {}

    def get_seeds(self, config_path: str | Path) -> list[int]:
        """Parses the 'seeds' section (defaults to [42] if not specified)."""
        config = self.load_raw_config(config_path)
        seeds = config.get("seeds", [42])
        return [int(s) for s in seeds]

    def get_dataset_managers(self, config_path: str | Path) -> list[IDatasetManager]:
        """Parses the 'datasets' section and constructs IDatasetManager instances by dataset name."""
        config = self.load_raw_config(config_path)
        datasets_dict = config.get("datasets", {})
        managers: list[IDatasetManager] = []

        for name, ds_cfg in datasets_dict.items():
            # Look up specialized DatasetManager by dataset name; fallback to CommonDatasetManager
            manager_cls = DATASET_REGISTRY.get(name, CommonDatasetManager)

            dataset_path = ds_cfg.get("dataset_path", "")
            split_path = ds_cfg.get("split_path", "")
            label_col = ds_cfg.get("label_column", "target")
            task_type_str = ds_cfg.get("task_type", "classification")
            task_type = TaskType(task_type_str)

            manager = manager_cls(
                name=name,
                label_column=label_col,
                task_type=task_type,
                dataset_path=dataset_path,
                split_path=split_path,
            )
            managers.append(manager)

        return managers

    def get_model_managers(self, config_path: str | Path) -> list[IModelManager]:
        """Parses the 'models' section and constructs IModelManager instances by model name."""
        config = self.load_raw_config(config_path)
        models_list = config.get("models", [])
        managers: list[IModelManager] = []

        for m_cfg in models_list:
            if isinstance(m_cfg, str):
                name = m_cfg
            elif isinstance(m_cfg, dict):
                name = m_cfg.get("name", "")
            else:
                continue

            if name not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model name '{name}'. Available models: {list(MODEL_REGISTRY.keys())}"
                )

            manager_cls = MODEL_REGISTRY[name]
            managers.append(manager_cls(name=name))

        return managers

    def read_experiment_config(self, config_path: str | Path) -> ExperimentConfig:
        """Reads complete experiment config containing dataset managers, model managers, and seeds."""
        raw_config = self.load_raw_config(config_path)
        datasets = self.get_dataset_managers(config_path)
        models = self.get_model_managers(config_path)
        seeds = self.get_seeds(config_path)
        return ExperimentConfig(
            datasets=datasets,
            models=models,
            seeds=seeds,
            raw_config=raw_config,
        )
