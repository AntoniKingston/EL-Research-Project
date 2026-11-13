import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.dataset_tools.preprocessing import FoldData
from src.models.base import TabularModel
from src.models.batchensemble import BatchEnsemble
from src.models.ensemble import Ensemble, cka_matrix, cosine_similarity_matrix


def metrics_from_outputs(
    outputs: np.ndarray, fold_data: FoldData, is_classification: bool
) -> dict[str, float]:
    """Classification: accuracy, ROC-AUC, NLL from logits. Regression: RMSE, MAE, R^2 on the
    original target scale."""
    if is_classification:
        y = fold_data.y_eval
        n_classes = outputs.shape[-1]
        probs = softmax(outputs.astype(np.float64), axis=-1)
        if n_classes == 2:
            auc = roc_auc_score(y, probs[:, 1])
        else:
            auc = roc_auc_score(y, probs, multi_class="ovr", labels=list(range(n_classes)))
        return {
            "accuracy": float(accuracy_score(y, probs.argmax(axis=-1))),
            "roc_auc": float(auc),
            "nll": float(log_loss(y, probs, labels=list(range(n_classes)))),
        }
    inverse = fold_data.preprocessor.inverse_target
    y_true, y_pred = inverse(fold_data.y_eval), inverse(outputs.reshape(-1))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics(model: TabularModel, fold_data: FoldData) -> dict[str, float]:
    return metrics_from_outputs(model.predict(fold_data.X_eval), fold_data, model.is_classification)


@torch.no_grad()
def _member_outputs_and_latents(
    model: TabularModel, X: np.ndarray, batch_size: int = 1024
) -> tuple[np.ndarray, np.ndarray | None]:
    """(M, N, out) member outputs and, for Ensemble, (M, N, d) latents."""
    model.eval()
    device = next(model.parameters()).device
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    outs, lats = [], []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i : i + batch_size]
        if isinstance(model, Ensemble):
            latents, outputs = model._members_forward(xb)
            lats.append(latents.cpu())
        elif isinstance(model, BatchEnsemble):
            outputs = model.member_outputs(xb)
        else:
            raise TypeError(f"{type(model).__name__} has no ensemble members")
        outs.append(outputs.cpu())
    latents_all = torch.cat(lats, dim=1).numpy() if lats else None
    return torch.cat(outs, dim=1).numpy(), latents_all


def ensemble_diagnostics(model: TabularModel, fold_data: FoldData) -> list[dict]:
    """Staged test metrics for the first m members and the functional norm of each member.

    Ensemble: Y^(m) = f_1 + nu * sum_{j=2..m} f_j. BatchEnsemble: mean of the first m members.
    Returns [] for single models.
    """
    if not isinstance(model, Ensemble | BatchEnsemble):
        return []
    member_outputs, _ = _member_outputs_and_latents(model, fold_data.X_eval)
    if isinstance(model, Ensemble):
        weights = model.member_weights.cpu().numpy()[:, None, None]
        staged = np.cumsum(weights * member_outputs, axis=0)
    else:
        counts = np.arange(1, len(member_outputs) + 1)[:, None, None]
        staged = np.cumsum(member_outputs, axis=0) / counts

    rows = []
    for m in range(len(member_outputs)):
        metrics = metrics_from_outputs(staged[m], fold_data, model.is_classification)
        norm = float(np.linalg.norm(member_outputs[m]) / np.sqrt(len(member_outputs[m])))
        rows.extend({"member": m + 1, "metric": k, "value": v} for k, v in metrics.items())
        rows.append({"member": m + 1, "metric": "member_output_norm", "value": norm})
    return rows


@torch.no_grad()
def similarity_matrices(model: TabularModel, X: np.ndarray) -> dict[str, np.ndarray]:
    """Pairwise CKA and squared functional cosine similarity between Ensemble members' centered
    penultimate representations on X. Returns {} for other models."""
    if not isinstance(model, Ensemble):
        return {}
    _, latents = _member_outputs_and_latents(model, X)
    assert latents is not None
    Z = torch.as_tensor(latents, dtype=torch.float64)
    Z = Z - Z.mean(dim=1, keepdim=True)
    return {
        "cka": cka_matrix(Z, Z).numpy(),
        "cosine": cosine_similarity_matrix(Z, Z).numpy(),
    }


def count_parameters(model: TabularModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
