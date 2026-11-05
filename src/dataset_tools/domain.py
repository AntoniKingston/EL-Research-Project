from dataclasses import dataclass
from enum import Enum

import numpy as np


class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"


@dataclass
class DatasetSplits:
    fold: int
    outer_train_idx: np.ndarray
    outer_test_idx: np.ndarray
    inner_train_idx: np.ndarray
    inner_val_idx: np.ndarray
