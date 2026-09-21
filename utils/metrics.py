import torch
import numpy as np
from typing import Dict
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)


def calculate_metrics(
    y_true: torch.Tensor,
    y_pred_logits: torch.Tensor,
    task_type: str,
) -> Dict[str, float]:
    try:
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred_logits.detach().cpu().numpy()
    except Exception:
        return (
            {"roc_auc": 0.0}
            if task_type == "classification"
            else {"rmse": float("inf"), "mae": float("inf")}
        )

    if y_true_np.ndim == 1:
        y_true_np = y_true_np.reshape(-1, 1)
    if y_pred_np.ndim == 1:
        y_pred_np = y_pred_np.reshape(-1, 1)
    if y_true_np.shape[0] == 0:
        return (
            {"roc_auc": 0.0}
            if task_type == "classification"
            else {"rmse": float("inf"), "mae": float("inf")}
        )

    num_tasks = y_true_np.shape[1]

    if task_type == "classification":
        prob_np = 1.0 / (1.0 + np.exp(-y_pred_np))
        auc_list = []
        acc_list = []
        f1_list = []

        for i in range(num_tasks):
            t = y_true_np[:, i]
            p = prob_np[:, i]
            mask = ~np.isnan(t)
            if not np.any(mask):
                continue
            tv = t[mask].astype(int)
            pv = p[mask]
            if len(np.unique(tv)) > 1:
                try:
                    auc_list.append(float(roc_auc_score(tv, pv)))
                except Exception:
                    pass
            try:
                pred = (pv >= 0.5).astype(int)
                acc_list.append(float(accuracy_score(tv, pred)))
                f1_list.append(
                    float(f1_score(tv, pred, average="binary", zero_division=0))
                )
            except Exception:
                pass

        return {
            "roc_auc": float(np.mean(auc_list)) if auc_list else 0.0,
            "accuracy": float(np.mean(acc_list)) if acc_list else 0.0,
            "f1": float(np.mean(f1_list)) if f1_list else 0.0,
        }

    if task_type == "regression":
        rmse_list = []
        mae_list = []
        for i in range(num_tasks):
            t = y_true_np[:, i]
            p = y_pred_np[:, i]
            mask = ~np.isnan(t)
            if not np.any(mask):
                continue
            tv = t[mask]
            pv = p[mask]
            try:
                rmse_list.append(float(np.sqrt(mean_squared_error(tv, pv))))
                mae_list.append(float(mean_absolute_error(tv, pv)))
            except Exception:
                pass
        return {
            "rmse": float(np.mean(rmse_list)) if rmse_list else float("inf"),
            "mae": float(np.mean(mae_list)) if mae_list else float("inf"),
        }

    return {}
