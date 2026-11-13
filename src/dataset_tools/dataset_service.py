from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .domain import DatasetSplits


class DatasetService:
    @staticmethod
    def get_nested_splits(
        X,
        y: np.ndarray,
        n_splits: int = 5,
        is_classification: bool = True,
        seed: int = 42,
    ) -> list[DatasetSplits]:
        """Deterministic outer folds, each with a single 80/20 inner train/val split."""
        if is_classification:
            outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = []
        for fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
            y_outer_train = y[outer_train_idx] if is_classification else None
            inner_train_rel, inner_val_rel = train_test_split(
                np.arange(len(outer_train_idx)),
                test_size=0.20,
                shuffle=True,
                random_state=seed,
                stratify=y_outer_train,
            )
            splits.append(
                DatasetSplits(
                    fold=fold,
                    outer_train_idx=outer_train_idx,
                    outer_test_idx=outer_test_idx,
                    inner_train_idx=outer_train_idx[inner_train_rel],
                    inner_val_idx=outer_train_idx[inner_val_rel],
                )
            )
        return splits

    @staticmethod
    def save_splits(path: str | Path, splits: list[DatasetSplits]) -> None:
        """Stores the split indices for the record (they are recomputed, never read back)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for s in splits:
            for part in ("outer_train_idx", "outer_test_idx", "inner_train_idx", "inner_val_idx"):
                arrays[f"fold{s.fold}_{part}"] = getattr(s, part)
        np.savez_compressed(path, **arrays)
