from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class TaskType(StrEnum):
    classification = "classification"
    regression = "regression"


@dataclass
class DatasetSplits:
    fold: int
    outer_train_idx: np.ndarray
    outer_test_idx: np.ndarray
    inner_train_idx: np.ndarray
    inner_val_idx: np.ndarray


@dataclass
class Dataset:
    """Raw (unpreprocessed) features with integer-encoded labels for classification."""

    name: str
    X: pd.DataFrame
    y: np.ndarray
    task_type: TaskType
    categorical: list[str]
    numerical: list[str]

    @property
    def is_classification(self) -> bool:
        return self.task_type == TaskType.classification

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1 if self.is_classification else 0
