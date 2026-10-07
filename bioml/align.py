"""
bioml.align — put batches of unequal length onto a common axis (Week 6, mostly).

IndPenSim batches run 167-290 h, a 74% spread. Multivariate SPC needs batches
to be commensurable; natively they are not. There is no obvious way to align
them, so this function takes the decision as an argument:

    "clock"     same elapsed hour. Assumes batches progress at the same rate,
                which is precisely what differs.
    "fraction"  same fraction of the batch's own duration. Needs the duration
                in advance, so it CANNOT be used on a running batch — useless
                for live fault detection even though it looks best offline.
    "trim"      cut every batch to the shortest. Throws away late-phase data,
                which for penicillin is where the product is.

Deliberately absent: dynamic time warping. It is a modelling decision with its
own failure modes and belongs in a student's Week 6 justification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import resolve

VALID_METHODS = ("clock", "fraction", "trim")


def align_batches(
    df: pd.DataFrame,
    method: str,                       # no default — see module docstring
    reference: int | None = None,
    n_points: int | None = None,
    columns: list[str] | None = None,
    time_col: str | None = None,
    batch_col: str | None = None,
) -> pd.DataFrame:
    """
    Long frame in which every batch is sampled at the same alignment points.

    method    : "clock" | "fraction" | "trim". Required.
    reference : batch whose time grid defines the target grid. Required for
                "clock" and "fraction". Your choice of reference is a modelling
                choice and belongs in the report.
    n_points  : number of alignment points (defaults from the reference or the
                shortest batch).
    columns   : columns to interpolate; defaults to every numeric non-identifier.

    Linear interpolation in time, within each batch, no extrapolation: points
    outside a batch's own span come back NaN rather than invented.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method={method!r} not in {VALID_METHODS}")

    time_col = time_col or resolve(df, "time")
    batch_col = batch_col or resolve(df, "batch")
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in (time_col, batch_col)]

    durations = df.groupby(batch_col)[time_col].max()

    # --- target axis ---------------------------------------------------------
    if method == "trim":
        n = n_points or int(df.groupby(batch_col).size().min())
        targets = np.linspace(float(df[time_col].min()), float(durations.min()), n)
        as_fraction = False
    else:
        if reference is None:
            raise ValueError(f'method="{method}" needs reference=<batch number>. '
                             f'Which batch defines "normal progress" is a decision.')
        ref = df.loc[df[batch_col] == reference, time_col]
        if ref.empty:
            raise ValueError(f"reference batch {reference} not present in df")
        n = n_points or len(ref)
        targets = np.linspace(float(ref.min()), float(ref.max()), n)
        as_fraction = (method == "fraction")
        if as_fraction:
            targets = (targets - targets.min()) / (targets.max() - targets.min())

    # --- resample every batch onto it ---------------------------------------
    out = []
    for b, sub in df.groupby(batch_col, sort=True):
        sub = sub.sort_values(time_col)
        t = sub[time_col].to_numpy(dtype=float)
        t_query = targets * float(durations.loc[b]) if as_fraction else targets
        row = {batch_col: b, "align_point": np.arange(len(t_query)), time_col: t_query}
        for c in columns:
            v = sub[c].to_numpy(dtype=float)
            ok = ~np.isnan(v)                  # interpolate on observed points only
            row[c] = (np.interp(t_query, t[ok], v[ok], left=np.nan, right=np.nan)
                      if ok.sum() >= 2 else np.full(len(t_query), np.nan))
        out.append(pd.DataFrame(row))
    return pd.concat(out, ignore_index=True)
