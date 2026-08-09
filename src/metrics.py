"""
Canonical industrial sorting metrics for the optical sorting project.

Purity, Yield and Recovery are derived from a confusion matrix and follow the
sensor-based sorting literature (Küppers et al. 2021; Maier et al. 2024):

    Purity   = TP / (TP + FP)        precision: how clean the output stream is
    Yield    = TP / (TP + FN)        recall: how much of the material is captured
    Recovery = TP / (TP + FP + FN)   composite, penalises contamination and loss

`compute_industrial_metrics` is the function notebook 08 defines inline. It is
lifted here verbatim so that notebooks added after 08 can score against exactly
the same definition instead of copying it. Notebook 08 is left as it is, and
notebook 14 asserts that this module reproduces the committed
`results/rf_industrial_metrics.csv` from `results/rf_confusion_matrix.npy`, so
the two cannot drift apart unnoticed.

The confusion matrix must be built with `labels=CLASS_ORDER` (see
`src.features.CLASS_ORDER`); passing an unsorted class list silently mislabels
every row.
"""

import numpy as np
import pandas as pd


def compute_industrial_metrics(cm, classes):
    """Per-class purity, yield and recovery from a confusion matrix.

    Returns a DataFrame with one row per class and columns
    Class, TP, FP, FN, Purity, Yield, Recovery.
    """
    results = []

    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP   # everything predicted as class i, minus the correct ones
        FN = np.sum(cm[i, :]) - TP   # everything truly class i, minus the correct ones

        purity   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        yield_   = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        recovery = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        results.append({
            'Class':    cls,
            'TP':       int(TP),
            'FP':       int(FP),
            'FN':       int(FN),
            'Purity':   round(purity, 4),
            'Yield':    round(yield_, 4),
            'Recovery': round(recovery, 4)
        })

    return pd.DataFrame(results)
