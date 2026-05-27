from sklearn.metrics import roc_curve, roc_auc_score

import numpy as np

def netEval(s, p0, p1):
    """
    Compute weighted ROC curve metrics for network output.

    Args:
        s:  network output scores in [0, 1], shape (N,)
        p0: event weights under c0 (background), shape (N,)
        p1: event weights under c1 (signal), shape (N,)
    Returns:
        fpr: false positive rates
        tpr: true positive rates
        auc: area under ROC curve
        acc: weighted accuracy at threshold 0.5
    """
    scores  = np.concatenate([s,  s])
    labels  = np.concatenate([np.zeros(len(s)), np.ones(len(s))])
    weights = np.concatenate([p0, p1])

    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    auc         = roc_auc_score(labels, scores, sample_weight=weights)

    preds   = (scores >= 0.5).astype(int)
    correct = (preds == labels).astype(float)
    acc     = np.average(correct, weights=weights)

    return fpr, tpr, auc, acc
