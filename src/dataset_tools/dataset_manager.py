from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .domain import Dataset, TaskType
from .openml import CATEGORICAL_DTYPES, load_openml_frame


def dataset_from_frame(
    name: str,
    frame: pd.DataFrame,
    label_column: str,
    task_type: TaskType,
    categorical: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Builds a raw `Dataset`; no statistics are fitted here (see `prepare_fold`).

    If `max_samples` is smaller than the dataset, a seeded subsample is drawn (stratified for
    classification). Class labels are mapped to 0..K-1 (a fixed relabeling, not a fitted
    statistic). Categorical columns default to the non-numeric dtypes.
    """
    is_classification = task_type == TaskType.classification
    if max_samples is not None and len(frame) > max_samples:
        stratify = frame[label_column] if is_classification else None
        frame, _ = train_test_split(
            frame, train_size=max_samples, random_state=seed, stratify=stratify
        )
        frame = frame.reset_index(drop=True)

    X = frame.drop(columns=label_column)
    if categorical is None:
        categorical = X.select_dtypes(include=CATEGORICAL_DTYPES).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]
    X[categorical] = X[categorical].astype(object)

    raw_y = frame[label_column]
    if is_classification:
        y = LabelEncoder().fit_transform(raw_y.astype(str)).astype(np.int64)
    else:
        y = raw_y.to_numpy(dtype=np.float32)
    return Dataset(name, X, y, task_type, categorical, numerical)


class IDatasetManager(ABC):
    """Loads one dataset as raw features + labels."""

    def __init__(self, name: str, max_samples: int | None = None):
        self.name = name
        self.max_samples = max_samples

    @abstractmethod
    def load(self, seed: int = 42) -> Dataset:
        """Returns the raw dataset; `seed` is used only for `max_samples` subsampling."""


class CommonDatasetManager(IDatasetManager):
    """Local CSV file with numeric and optional categorical features."""

    def __init__(
        self,
        name: str,
        dataset_path: str | Path,
        label_column: str,
        task_type: TaskType | str,
        categorical_columns: list[str] | None = None,
        max_samples: int | None = None,
    ):
        super().__init__(name, max_samples)
        self.dataset_path = Path(dataset_path)
        self.label_column = label_column
        try:
            self.task_type = TaskType(task_type)
        except ValueError as err:
            raise ValueError(f"Unknown task type {task_type!r} for dataset {name}") from err
        self.categorical_columns = categorical_columns

    def load(self, seed: int = 42) -> Dataset:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset {self.name} not found at {self.dataset_path}")
        frame = pd.read_csv(self.dataset_path)
        return dataset_from_frame(
            self.name,
            frame,
            self.label_column,
            self.task_type,
            self.categorical_columns,
            self.max_samples,
            seed,
        )


class OpenMLDatasetManager(IDatasetManager):
    """OpenML dataset, downloaded once into `data_dir/raw` and loaded from the cache after."""

    def __init__(self, name: str, link: str, data_dir: str | Path, max_samples: int | None = None):
        super().__init__(name, max_samples)
        self.link = link
        self.data_dir = Path(data_dir)

    def load(self, seed: int = 42) -> Dataset:
        frame, meta = load_openml_frame(self.name, self.link, self.data_dir)
        task_type = TaskType.classification if meta["is_classification"] else TaskType.regression
        return dataset_from_frame(
            self.name,
            frame,
            meta["target"],
            task_type,
            meta["categorical"],
            self.max_samples,
            seed,
        )
