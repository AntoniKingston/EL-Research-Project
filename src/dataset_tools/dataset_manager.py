from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .domain import TaskType


class IDatasetManager(ABC):
    """Serves as data loader and encoder for a dataset."""

    def __init__(
        self,
        name: str,
        label_column: str,
        task_type: TaskType | str,
        dataset_path: str = "",
        split_path: str = "",
    ):
        self.name = name
        self.label_column = label_column
        self.dataset_path = dataset_path
        self.split_path = split_path

        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                raise ValueError(f"Unknown task type provided: {task_type}, for dataset: {name}")
        elif task_type not in (TaskType.classification, TaskType.regression):
            raise ValueError(f"Unknown task type provided: {task_type}, for dataset: {name}")
        self.task_type = task_type

    @abstractmethod
    def get_labled_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Provide labeled data as numeric numpy arrays (features and encoded targets)."""


class CommonDatasetManager(IDatasetManager):
    """Data loader and encoder for datasets with numeric features and optional categorical columns."""

    def get_labled_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw_y = df[self.label_column].to_numpy()
        features_df = df.drop(columns=[self.label_column])

        # Impute missing values with column medians for numeric columns
        features_df = features_df.fillna(features_df.median(numeric_only=True))

        # Encode categorical / non-numeric columns if any exist
        for col in features_df.select_dtypes(include=["object", "category"]).columns:
            features_df[col] = LabelEncoder().fit_transform(features_df[col].astype(str))

        X = features_df.to_numpy().astype(np.float32)

        if self.task_type == TaskType.classification:
            encoder = LabelEncoder()
            Y = encoder.fit_transform(raw_y).astype(np.int64)
        else:
            Y = raw_y.astype(np.float32)

        return X, Y
