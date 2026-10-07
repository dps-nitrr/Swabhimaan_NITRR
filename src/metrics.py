"""Evaluation metrics: forecast error (mg/dL) and early-warning quality."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .data import HYPER, HYPO


def regression_metrics(Y_true, Y_pred, steps=(2, 4, 8)):
    """RMSE / MAE (mg/dL) at +30, +60, +120 min (steps are 15-min units)."""
    out = {}
    for h in steps:
        err = Y_true[:, h - 1] - Y_pred[:, h - 1]
        out[f"RMSE_{h * 15}"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"MAE_{h * 15}"] = float(np.mean(np.abs(err)))
    return out


def bootstrap_rmse_ci(Y_true, Y_pred, pids, h=4, n_boot=500, seed=0):
    """95% CI of the pooled RMSE at step h, resampling whole patients/records (not windows)."""
    err2 = (Y_true[:, h - 1] - Y_pred[:, h - 1]) ** 2
    ids = np.unique(pids)
    s = np.array([err2[pids == i].sum() for i in ids])
    c = np.array([(pids == i).sum() for i in ids])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ids), len(ids))
        vals.append(np.sqrt(s[k].sum() / c[k].sum()))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sens_at_spec(y, s, spec=0.9):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(1 - spec, fpr, tpr))


def event_metrics(Y_true, scores, X_raw):
    """Early-warning metrics.

    Only windows where glucose is currently IN RANGE (70-180) are used, so the model
    must anticipate a NEW excursion instead of just repeating the current value.
    scores: (N, 2) -> [hypo_score, hyper_score] (higher = more likely).
    """
    now = X_raw[:, -1, 0]
    inr = (now >= HYPO) & (now <= HYPER)
    labels = np.stack([(Y_true < HYPO).any(1), (Y_true > HYPER).any(1)], axis=1)
    out = {}
    for j, name in enumerate(["hypo", "hyper"]):
        y, s = labels[inr, j], scores[inr, j]
        out[f"{name}_prevalence"] = float(y.mean()) if len(y) else float("nan")
        out[f"{name}_positives"] = int(y.sum())
        if y.sum() < 5 or (1 - y).sum() < 5:
            out[f"AUROC_{name}"] = float("nan")
            out[f"Sens@90Spec_{name}"] = float("nan")
        else:
            out[f"AUROC_{name}"] = float(roc_auc_score(y, s))
            out[f"Sens@90Spec_{name}"] = sens_at_spec(y, s)
    return out


def scores_from_forecast(Y_pred):
    """Turn a glucose forecast into event scores (used for the baselines)."""
    return np.stack([-Y_pred.min(1), Y_pred.max(1)], axis=1)
