import json
import os
from pathlib import Path

import numpy as np
import optuna
import torch

from src.config_readers import ConfigReader
from src.dataset_tools.dataset_manager import IDatasetManager
from src.dataset_tools.dataset_service import DatasetService
from src.dataset_tools.domain import TaskType

CONFIG_PATH = "configs/experiment_config.yaml"
N_TRIALS = 30

def get_dataset_and_split(
    dataset_manager: IDatasetManager, dataset_service: DatasetService, seed: int = 42
):
    if os.path.exists(dataset_manager.split_path):
        dataset_df, splits = dataset_service.load_dataset_and_splits(
            dataset_path=dataset_manager.dataset_path, splits_path=dataset_manager.split_path
        )
        X, Y = dataset_manager.get_labled_data(dataset_df)
    else:
        dataset_df = dataset_service.read_raw_csv(dataset_manager.dataset_path)
        X, Y = dataset_manager.get_labled_data(dataset_df)
        splits = dataset_service.get_nested_splits(
            X, Y, is_classification=(dataset_manager.task_type == TaskType.classification), seed=seed
        )
        dataset_service.save_dataset_splits(dataset_manager.split_path, splits)
    return X, Y, splits


def main():
    # Mute verbose trial-level logging from Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    project_dir = Path(__file__).resolve().parents[2]
    config = ConfigReader(project_root=project_dir).read_experiment_config(config_path=CONFIG_PATH)
    dataset_managers = config.datasets
    model_managers = config.models
    seeds = config.seeds

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting orchestrator on device: {device}")

    dataset_service = DatasetService()

    for dataset_manager in dataset_managers:
        if not os.path.exists(dataset_manager.dataset_path):
            print(f"Lack of dataset {dataset_manager.name} under the path {dataset_manager.dataset_path}")
            continue

        print(f"\n=================== Dataset: {dataset_manager.name} ===================")
        X, Y, folds = get_dataset_and_split(dataset_manager, dataset_service, 42)

        input_dim = X.shape[1]
        is_classification = dataset_manager.task_type == TaskType.classification
        output_dim = len(np.unique(Y)) if is_classification else 1

        for model_manager in model_managers:
            print(f"\n--- Model: {model_manager.name} ---")

            # Configure dataset-specific dimensions and loss function
            model_manager.input_dim = input_dim
            model_manager.output_dim = output_dim

            for fold in folds:
                print(f">> Tuning Fold {fold.fold} ({N_TRIALS} trials)...")
                
                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    study = optuna.create_study(direction="minimize")
                    study.optimize(
                        lambda trial: model_manager.hyperparameters_study(trial, device, X, Y, is_classification, fold),
                        n_trials=N_TRIALS,
                    )
                    best_params = study.best_params
                    print(f"Fold {fold.fold} Best Params: {best_params}")

                    model = model_manager.train_optimal_model(best_params, device, X, Y, is_classification, fold)

                    # Save model checkpoint
                    save_dir = os.path.join(
                        project_dir, "saved_models", dataset_manager.name, model_manager.name
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(
                        save_dir, f"{model_manager.name}_seed{seed}_fold{fold.fold}.pt"
                    )
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved model checkpoint to: {save_path}")

                    # Save loss history
                    history_path = os.path.join(
                        save_dir, f"{model_manager.name}_seed{seed}_fold{fold.fold}_history.json"
                    )
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(model.loss_history, f, indent=2)

                    # Placeholder for evaluation methods
                    X_train = X[fold.outer_train_idx]
                    X_test = X[fold.outer_test_idx]
                    Y_test = Y[fold.outer_test_idx]
                    _, X_test_norm = DatasetService.normalize_data(X_train, X_test)
                    y_pred = torch.as_tensor(model.predict(X_test_norm))
                    y_true = torch.as_tensor(Y_test)
                    if not np.issubdtype(Y_test.dtype, np.integer):
                        y_true = y_true.to(y_pred.dtype)
                    with torch.no_grad():
                        test_loss = model.task_loss(y_pred, y_true).item()

                    print(
                        f"[{dataset_manager.name} | {model_manager.name} | Fold {fold.fold} | Seed {seed}] "
                        f"Test Loss: {test_loss:.4f}"
                    )


if __name__ == "__main__":
    main()