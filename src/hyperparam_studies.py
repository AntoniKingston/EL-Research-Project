from src.models.batchensemble import BatchEnsemble
from src.models.ensemble import Ensemble
from src.models.resnet import Resnet
from src.models.tangos import Tangos
from src.utils import DatasetSplits, mean_score_by_folds


def objective_single_resnet(
    trial,
    input_dim,
    output_dim,
    epochs,
    patience,
    optimizer,
    lr,
    wd,
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

    def model_fun():
        return Resnet(input_dim, output_dim, hidden_dim, n_blocks, dr)

    return mean_score_by_folds(
        model_fun, X, y, folds, epochs, patience, lr, wd, optimizer, loss_fn, batch_size, device
    )


def objective_tangos_resnet(
    trial,
    input_dim,
    output_dim,
    hidden_dim,
    n_blocks,
    dr,
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
    lr = trial.suggest_float("lr", 1e-4, 5 * 1e-2, log=True)
    wd = trial.suggest_float("wd", 2.5 * 1e-3, 1e-2, log=True)
    sparsity = trial.suggest_float("sparsity", 1e-4, 1e-1, log=True)
    orthogonality = trial.suggest_float("orthogonality", 1e-3, 1.0, log=True)

    def model_fun():
        return Tangos(
            Resnet(input_dim, output_dim, hidden_dim, n_blocks, dr), sparsity, orthogonality
        )

    return mean_score_by_folds(
        model_fun, X, y, folds, epochs, patience, lr, wd, optimizer, loss_fn, batch_size, device
    )


def objective_batchensemble_resnet(
    trial,
    input_dim,
    output_dim,
    hidden_dim,
    n_blocks,
    dr,
    ensemble_size,
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
    lr = trial.suggest_float("lr", 1e-4, 5 * 1e-2, log=True)
    wd = trial.suggest_float("wd", 2.5 * 1e-3, 1e-2, log=True)
    adapter_initialization = trial.suggest_categorical(
        "adapter_initialization", ["random_sign", "normal"]
    )

    def model_fun():
        return BatchEnsemble(
            Resnet(input_dim, output_dim, hidden_dim, n_blocks, dr),
            ensemble_size,
            adapter_initialization,
        )

    return mean_score_by_folds(
        model_fun, X, y, folds, epochs, patience, lr, wd, optimizer, loss_fn, batch_size, device
    )


# For unregularized ensemble, symmetrically orthogonalized and SFOvTSGM
def objective_ensemble_resnet(
    trial,
    input_dim,
    output_dim,
    hidden_dim,
    n_blocks,
    dr,
    mask,
    orthogonality_weight,
    ensemble_size,
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
    lr = trial.suggest_float("lr", 1e-4, 5 * 1e-2, log=True)
    wd = trial.suggest_float("wd", 2.5 * 1e-3, 1e-2, log=True)
    additive_coef = trial.suggest_float("additive_coef", 0.0, 1.0, log=True)
    similarity_metric = trial.suggest_categorical("similarity_metric", ["cosine", "cka"])

    def model_fun():
        return Ensemble(
            Resnet(input_dim, output_dim, hidden_dim, n_blocks, dr),
            ensemble_size,
            mask,
            orthogonality_weight,
            additive_coef,
            similarity_metric,
        )

    return mean_score_by_folds(
        model_fun, X, y, folds, epochs, patience, lr, wd, optimizer, loss_fn, batch_size, device
    )
