import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .domain import DatasetSplits

VALID_SPLIT_EXTENSIONS = (".pkl", ".pickle", ".plk")


class DatasetService:
    def __init__(self):
        pass

    def read_raw_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def save_dataset_splits(self, saving_path: str, splits: list[DatasetSplits]) -> None:
        """Saves computed split of the dataset as pickle file."""
        parent_dir = os.path.dirname(saving_path)

        ext = Path(saving_path).suffix.lower()
        if ext not in VALID_SPLIT_EXTENSIONS:
            raise ValueError(
                f"Split only could be saved as a pickle file ({VALID_SPLIT_EXTENSIONS}), tried to save as {ext} instead"
            )

        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(saving_path, "wb") as f:
            pickle.dump(splits, f)

    def load_dataset_and_splits(
        self, dataset_path: str, splits_path: str
    ) -> tuple[pd.DataFrame, list[DatasetSplits]]:
        """Reads dataset and previously computed splits."""
        dataset = pd.read_csv(dataset_path)

        ext = Path(splits_path).suffix.lower()
        if ext not in VALID_SPLIT_EXTENSIONS:
            raise ValueError(
                f"Split only could be read as a pickle file ({VALID_SPLIT_EXTENSIONS}), tried to read {ext} instead"
            )

        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        return dataset, splits

    def get_nested_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        is_classification: bool = True,
        seed: int = 42,
    ) -> list[DatasetSplits]:
        """Generates deterministic outer and inner splits to be reused across all models."""
        if is_classification:
            outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = []
        for fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
            y_outer_train = y[outer_train_idx] if is_classification else None
            # Split 80% outer train into 80% inner_train and 20% inner_val
            inner_train_rel, inner_val_rel = train_test_split(
                np.arange(len(outer_train_idx)),
                test_size=0.20,
                shuffle=True,
                random_state=seed,
                stratify=y_outer_train,
            )
            inner_train_idx = outer_train_idx[inner_train_rel]
            inner_val_idx = outer_train_idx[inner_val_rel]
            splits.append(
                DatasetSplits(
                    fold=fold,
                    outer_train_idx=outer_train_idx,
                    outer_test_idx=outer_test_idx,
                    inner_train_idx=inner_train_idx,
                    inner_val_idx=inner_val_idx,
                )
            )
        return splits
