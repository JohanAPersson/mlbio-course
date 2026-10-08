"""
bioml.plots — the three plots this course uses over and over.

Each returns the matplotlib Axes so you can add titles, series or annotations
without this module anticipating what you want. Nothing here computes a
statistic you are then asked to interpret.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .columns import resolve


def parity_plot(y_true, y_pred, ax=None, label: str | None = None, title: str = ""):
    """
    Predicted vs measured, with the 1:1 line.

    Read it for the SHAPE, not the R². Curvature means a mis-specified model;
    a fan means error grows with level; bands mean the target is discrete.
    None of that is visible in a single R² number.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.scatter(y_true, y_pred, s=18, alpha=0.6, label=label)

    # 1:1 line spanning both series, so the aspect is honest.
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")   # a squashed parity plot lies

    ax.set_xlabel("measured")
    ax.set_ylabel("predicted")
    if title:
        ax.set_title(title)
    if label:
        ax.legend(loc="upper left", fontsize=8)
    return ax


def residual_plot(y_true, y_pred, x=None, ax=None, xlabel: str = "predicted", title: str = ""):
    """
    Residuals (measured − predicted) against `x`, defaulting to the prediction.

    Pass x=time to see whether error drifts through a batch — for a soft sensor
    that is the difference between a usable instrument and one that is only
    right in mid-fermentation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    x = y_pred if x is None else np.asarray(x, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.axhline(0.0, color="k", lw=1, zorder=0)       # the "no error" line
    ax.scatter(x, resid, s=18, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("residual (measured − predicted)")
    if title:
        ax.set_title(title)
    return ax


def batch_trajectory_plot(df: pd.DataFrame, column: str, batches=None, ax=None,
                          time_col: str | None = None, batch_col: str | None = None,
                          alpha: float = 0.7, legend: bool = True):
    """
    One line per batch for a single column, against process time.

    df      : long frame with batch and time columns (what load_process_vars gives).
    column  : the actual column name to plot.
    batches : which batches to draw; None draws every batch in df.

    NaNs are dropped per batch, so a ~98%-sparse offline assay shows as a
    sequence of markers rather than empty axes.
    """
    time_col = time_col or resolve(df, "time")
    batch_col = batch_col or resolve(df, "batch")
    if column not in df.columns:
        raise KeyError(f"{column!r} is not a column of this frame")

    ids = sorted(df[batch_col].unique()) if batches is None else list(batches)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    sparse = df[column].notna().mean() < 0.5      # markers for sparse, lines for dense
    for b in ids:
        sub = df.loc[df[batch_col] == b, [time_col, column]].dropna()
        if sub.empty:
            continue
        ax.plot(sub[time_col], sub[column], marker="o" if sparse else None,
                ms=3, lw=1.2, alpha=alpha, label=f"batch {int(b)}")

    ax.set_xlabel("process time (h)")
    ax.set_ylabel(column)
    if legend and len(ids) <= 8:
        ax.legend(fontsize=8, ncol=2)
    return ax
