"""Scoring functions shared by every model in this project.

Deliberately one module: the baseline and the gradient-boosted model must be
scored by literally the same code, or the comparison between them is not
evidence of anything. Nothing here fits or learns.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "predicted_positive_rate": float(y_pred.mean()),
    }


def ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "prevalence": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweep a fixed grid on VAL only. Applied unchanged to test."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])
