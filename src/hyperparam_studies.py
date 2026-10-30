import numpy as np
from sklearn.metrics import accuracy_score
from src.dataset_splits import DatasetSplits

from src.models.resnet import Resnet


def objective_single_resnet(
    trial,
    input_dim,
    output_dim,
    epochs,
    patience,
    optimizer,
    loss_fn,
    batch_size,
    device,
    X,
    y,
    folds: list[DatasetSplits],
):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    n_blocks = trial.suggest_int("n_blocks", 2, 4)
    dr = trial.suggest_float("dr", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5 * 1e-2, log=True)
    wd = trial.suggest_float("wd", 2.5 * 1e-3, 1e-2, log=True)

    scores = np.array([])

    for fold in folds:
        X_train = X[fold.inner_train_idx]
        y_train = y[fold.inner_train_idx]
        X_val = X[fold.inner_val_idx]
        y_val = y[fold.inner_val_idx]

        model = Resnet(input_dim, output_dim, hidden_dim, n_blocks, dr)

        model.to(device)

        model.fit(X_train, y_train, epochs, patience, lr, wd, optimizer, loss_fn, batch_size)
        y_pred = model.predict(X_val)
        scores = np.append(scores, accuracy_score(y_val, y_pred))

    return np.mean(scores)
