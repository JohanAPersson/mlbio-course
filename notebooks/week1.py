# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Week 1 — Bioprocess data and the ML framing
#
# **Level: Specified.** Every modelling decision this week is made for you and
# stated as a choice. Your job is to understand why each was made.
#
# **This notebook runs. It produces the wrong answer. There are exactly two
# errors, both in the reasoning, neither in the syntax.** Both are in
# `derive_rates()` (§5). Nothing raises an exception and nothing prints a
# warning.
#
# Rules of the week: write a prediction in every *Predict before running* cell
# **before** you run the cell below it; swap driver and navigator every 30
# minutes; submit only after *Restart session and run all*.
#
# ## What Week 1 is for
#
# 1. **The ratio.** 2.57 GB of file, n ≈ 100 independent runs. Both numbers
#    surprise, in opposite directions, and both mislead the same way.
# 2. **The observability boundary.** Which columns would exist on a 100,000 L
#    fermenter at hour 100, and which exist only because a simulator wrote its
#    own state to disk?
# 3. **Rates, not levels.** A fermentation is understood through µ, q_P and
#    q_O₂, and computing them correctly in a fed-batch requires knowing the
#    difference between a concentration and an amount.

# %%
# --- Setup: install the course package and fetch the data --------------------
# Run this cell first, every session. In Colab it installs `bioml` (the course
# package) and downloads ~20 MB of course data into /content/data. On a machine
# where the data is already present (BIOML_DATA set), it downloads nothing.
import os
import subprocess
import sys

# Pinned release: every pair runs the identical package and data (Decision 9).
BIOML_PIP = "git+https://github.com/COURSE-ORG/ml-bioprocesses.git@v1.0.0"
DATA_URL = ("https://github.com/COURSE-ORG/ml-bioprocesses/releases/download/"
            "v1.0.0/course-data-core.zip")

if "google.colab" in sys.modules:
    # pip is called as a subprocess rather than a "!pip" magic so that this
    # notebook also runs as a plain Python script outside Jupyter.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", BIOML_PIP], check=True)

import bioml

# ensure_data() downloads only if the files are not already there, then points
# every bioml loader at that folder.
DATA_DIR = bioml.ensure_data(os.environ.get("BIOML_DATA", "/content/data"), DATA_URL)

# Provenance, printed at the top of every notebook. When a number in your report
# is questioned weeks from now, this line says which package and data made it.
print(bioml.about())

manifest = bioml.load_manifest()
if manifest.get("synthetic"):
    print("\n*** SYNTHETIC FIXTURE — code verification only, not IndPenSim ***\n")

# %%
# Standard scientific Python, used throughout the course.
import inspect

import matplotlib.pyplot as plt   # plotting
import numpy as np                # arrays and maths
import pandas as pd               # tables (DataFrames)

# %% [markdown]
# ## 1. The ratio
#
# Compute it yourself; do not take it from the lecture slides.
#
# > **Predict before running the next cell.** How many fermentation runs are in
# > the file, and how many rows per run? Write numbers, not "a lot".
# >
# > *Prediction:*

# %%
# Size of the source distribution, from the manifest written by the conversion
# script. The synthetic fixture has no source file, so fall back to the
# documented figure rather than printing a wrong number.
source_bytes = manifest.get("source_size_bytes")
source_gb = source_bytes / 1e9 if source_bytes else 2.57

n_rows = manifest["rows"]              # every logged time point, all batches
n_batches = manifest["n_batches"]      # independently executed fermentations

print(f"source file        {source_gb:.2f} GB")
print(f"rows               {n_rows:,}")
print(f"batches            {n_batches}")
print(f"bytes per batch    {source_gb * 1e9 / n_batches:,.0f}")
print(f"rows per batch     {n_rows / n_batches:,.0f}")

# The file is too large to open in Excel; the experiment behind it is small
# enough that a hundred-parameter model has no business being fitted to it.
# Rows are not independent samples. Batches are.

# %% [markdown]
# ## 2. Load the process variables
#
# Weeks 1–2 use batches 1–30 only: recipe-driven, no Raman. One regime, so any
# structure you find is not a control-strategy artefact.
#
# > **Predict before running the next cell.** Roughly how many megabytes will
# > batches 1–30 occupy in memory, given what you computed above?
# >
# > *Prediction:*

# %%
# Parquet is columnar: this reads only the non-Raman block off disk.
WEEK1_BATCHES = range(1, 31)
proc = bioml.load_process_vars(batches=WEEK1_BATCHES)

print(f"{proc.shape[0]:,} rows x {proc.shape[1]} columns")
print(f"memory: {proc.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Bind short keys to the file's actual column names once, so nothing below
# contains a 60-character string literal. resolve_many() raises if a key does
# not match exactly one column.
col = bioml.resolve_many(proc, [
    "time", "batch", "volume", "our",
    "x_offline", "p_offline",
    "temperature", "ph", "do", "o2_offgas", "co2_offgas",
])
for k, v in col.items():
    print(f"  {k:<12} -> {v}")

# %% [markdown]
# ## 3. Profile the data before modelling anything
#
# Four questions, in order: how long are the batches, how complete is each
# column, which columns never move, and what does a trajectory look like.
#
# > **Predict before running the next three cells.** (i) Do all batches run the
# > same length? (ii) Roughly what fraction of rows carries a laboratory
# > biomass measurement?
# >
# > *Prediction:*

# %%
# --- Batch durations ---------------------------------------------------------
# groupby(batch) splits the table into one group per batch; .max() of the time
# column is then each batch's duration.
dur = proc.groupby(col["batch"])[col["time"]].max()
rows_per = proc.groupby(col["batch"]).size()

print("batch duration (h)   "
      f"min {dur.min():.1f}   median {dur.median():.1f}   max {dur.max():.1f}"
      f"   spread {100 * (dur.max() / dur.min() - 1):.0f}%")
print("rows per batch       "
      f"min {rows_per.min()}   median {rows_per.median():.0f}   max {rows_per.max()}")

# Unequal lengths mean the batches cannot be stacked into a rectangular
# batch x time x variable array without an alignment decision (Week 6).

# %%
# --- Completeness ------------------------------------------------------------
# .notna().mean() is the fraction of rows in which each column has a value.
observed = proc.notna().mean().sort_values()
sparse = observed[observed < 0.5]
dense = observed[observed >= 0.5]

print(f"{len(sparse)} sparse columns (<50% observed):")
for name, frac in sparse.items():
    n_labels = int(proc[name].notna().sum())
    print(f"  {name[:58]:<60} {frac:6.2%}  ({n_labels:,} values, ~1 per {1 / frac:.0f} rows)")
print(f"\n{len(dense)} dense columns (>=50% observed)")

# The offline sampling interval is a measured property of the file, read from
# the manifest rather than assumed.
print("\noffline sampling, from manifest.json:")
for name, stats in manifest.get("offline_sampling", {}).items():
    print(f"  {name[:50]:<52} 1 label per {stats['approx_rows_per_label']} rows "
          f"(~{stats.get('approx_hours_per_label')} h)")

# Dense cheap sensors beside sparse expensive assays is not a defect in the
# file. It is the soft-sensing problem, stated as a table.

# %%
# --- Columns that never move -------------------------------------------------
# A column with a single distinct value carries no information, and breaks any
# scaler that divides by the standard deviation.
nunique = proc.nunique(dropna=True)
dead_cols = nunique[nunique <= 1].index.tolist()
print(f"zero-variance columns ({len(dead_cols)}):")
for c in dead_cols:
    print(f"  {c}  = {proc[c].dropna().iloc[0] if proc[c].notna().any() else 'all NaN'}")

# %%
# --- What a batch looks like -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
show = [3, 7, 11, 19, 25]   # five batches; thirty lines is a smear

bioml.batch_trajectory_plot(proc, col["volume"], batches=show, ax=axes[0, 0])
axes[0, 0].set_title("vessel volume — the fed-batch fill")

bioml.batch_trajectory_plot(proc, col["our"], batches=show, ax=axes[0, 1], legend=False)
axes[0, 1].set_title("oxygen uptake rate (whole vessel)")

bioml.batch_trajectory_plot(proc, col["x_offline"], batches=show, ax=axes[1, 0], legend=False)
axes[1, 0].set_title("biomass — offline assay, sparse")

bioml.batch_trajectory_plot(proc, col["p_offline"], batches=show, ax=axes[1, 1], legend=False)
axes[1, 1].set_title("penicillin — offline assay, sparse")

fig.tight_layout()
plt.show()

# Look at the top-left panel: by what factor does the volume change over a
# batch? Section 5 depends on that number.

# %% [markdown]
# ## 4. The observability boundary — deliverable (a)
#
# The file contains columns a plant would have, and columns that exist only
# because a simulator wrote its internal state to disk. Telling them apart is a
# bioprocess question, not a programming question.
#
# **Order matters here.** The next cell lists the columns. Classify them in the
# markdown cell below it *before* running §4b, which loads the course's own
# classification. Disagreeing with it and defending your view scores better
# than agreeing without a reason.

# %%
# Every non-Raman column, with how often it is observed — the raw material for
# your classification.
for c in proc.columns:
    print(f"  {proc[c].notna().mean():6.1%}   {c}")

# %% [markdown]
# **Our classification** (one line per column: *plant has it at hour 100? yes /
# no / lab only* — and why). Pay particular attention to
# `Penicillin concentration(P:g/L)` and `Substrate concentration(S:g/L)`.
#
# | column | on the plant at hour 100? | why |
# |---|---|---|
# | | | |

# %% [markdown]
# ### 4b. Compare with the shipped classification

# %%
roles = bioml.load_column_roles()
print(roles.groupby("role").size().to_string())

# The columns that will silently destroy a soft sensor if left in a feature
# matrix. They are shipped, labelled, on purpose: deleting them would hide the
# reason soft sensing exists at all.
print("\n--- simulator_state: present in the file, absent on a plant ---")
for _, r in roles[roles.role == "simulator_state"].iterrows():
    print(f"\n  {r['column']}\n    {r['reason']}")

# Numerical evidence that the online penicillin column and the offline assay are
# the same quantity, one dense and one sparse. Looking at it for analysis is
# legitimate; using it as a model INPUT to predict the assay is not.
p_state = bioml.try_resolve(proc, "p_state")
if p_state:
    print(f"\n  online  P max = {proc[p_state].max():.2f} g/L")
    print(f"  offline P max = {proc[col['p_offline']].max():.2f} g/L")

# %% [markdown]
# ## 5. Derived rates
#
# A fermentation is described by rates, not levels: how fast the organism grows,
# makes product, consumes oxygen. All three are computed from the sparse offline
# assays.

# %%
def derive_rates(batch: pd.DataFrame, col: dict[str, str]) -> pd.DataFrame:
    """
    Compute µ, q_P and q_O2 for a single batch from its offline assays.

    batch : rows for ONE batch, sorted by time.
    col   : mapping of short keys to this file's column names.

    Returns one row per assay interval, with the rates attached to the midpoint
    of the interval they were computed over.
    """
    # Pull the columns out as plain float arrays. .to_numpy() detaches them from
    # the DataFrame index, so no later operation can silently align on labels.
    t = batch[col["time"]].to_numpy(dtype=float)
    V = batch[col["volume"]].to_numpy(dtype=float)
    X = batch[col["x_offline"]].to_numpy(dtype=float)
    P = batch[col["p_offline"]].to_numpy(dtype=float)
    our_g_per_min = batch[col["our"]].to_numpy(dtype=float)

    # Keep only rows where BOTH assays were taken; the rest are online-only
    # samples without a laboratory measurement.
    obs = np.isfinite(X) & np.isfinite(P)
    if obs.sum() < 3:
        return pd.DataFrame(columns=["time_h", "mu", "qP", "qO2", "mX", "mP"])

    t, V, X, P = t[obs], V[obs], X[obs], P[obs]
    our_g_per_h = our_g_per_min[obs] * 60.0    # g/min -> g/h, once, here

    # Total amounts in the vessel (g): concentration (g/L) times volume (L).
    mX = X * V
    mP = P * V

    # Time step for the finite differences below, in hours.
    dt = 0.2

    # Specific growth rate: difference of the natural log, per hour.
    mu = np.diff(np.log(mX)) / dt

    # Logarithmic mean of the biomass over each interval.
    mX_logmean = (mX[1:] - mX[:-1]) / np.log(mX[1:] / mX[:-1])

    # Specific productivity: product formed per hour, per gram of biomass.
    qP = (np.diff(mP) / dt) / mX_logmean

    # Specific oxygen uptake: oxygen uptake rate divided by biomass.
    qO2 = our_g_per_h / X

    return pd.DataFrame({
        "time_h": 0.5 * (t[:-1] + t[1:]),      # rates describe an interval: report at its midpoint
        "mu": mu,
        "qP": qP,
        "qO2": 0.5 * (qO2[:-1] + qO2[1:]),     # point values averaged onto the same midpoints
        "mX": mX_logmean,
        "mP": 0.5 * (mP[:-1] + mP[1:]),
    })


# %%
# Apply derive_rates() to every batch and stack the results, labelled by batch.
rates = []
for b, sub in proc.groupby(col["batch"], sort=True):
    r = derive_rates(sub, col)
    r.insert(0, "batch", int(b))
    rates.append(r)
rates = pd.concat(rates, ignore_index=True)
print(f"{len(rates):,} rate estimates from {rates['batch'].nunique()} batches")

# %% [markdown]
# ### Sanity-check the rates against what you know
#
# These are numbers about *Penicillium chrysogenum*, and you already know
# roughly what they should be. That knowledge is the error detector.
#
# > **Predict before running the next cell.** Write the range you expect for
# > each rate, with units: µ (1/h), q_P (g P per g X per h), q_O₂ (g O₂ per g X
# > per h). Then compare.
# >
# > *Prediction:*

# %%
# Median and 5th/95th percentiles of each derived rate, over all intervals.
for name, unit in [("mu", "1/h"), ("qP", "g P/(g X·h)"), ("qO2", "g O2/(g X·h)")]:
    v = rates[name].to_numpy()
    print(f"{name:<4} median {np.nanmedian(v):>10.4g}   p5 {np.nanpercentile(v, 5):>10.4g}"
          f"   p95 {np.nanpercentile(v, 95):>10.4g}   [{unit}]")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, name, ylab in zip(axes, ["mu", "qP", "qO2"],
                          ["µ (1/h)", "q_P (g/g/h)", "q_O2 (g/g/h)"]):
    for b, sub in rates.groupby("batch"):
        ax.plot(sub["time_h"], sub[name], lw=0.8, alpha=0.35, color="tab:blue")
    ax.set_xlabel("process time (h)")
    ax.set_ylabel(ylab)
    ax.axhline(0, color="k", lw=0.8, zorder=0)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Deliverable (b): a mechanistic hypothesis
#
# Two or three sentences: which variables you expect to drive final titer, the
# physical reason for each, and what in these plots supports or undermines it.
# It will be held against your Week 3–4 models and is allowed to turn out wrong.
#
# *Our hypothesis:*

# %%
# A starting point, not an answer: correlate each batch's mean rates with its
# final titer (the largest offline penicillin value in the batch).
final_titer = proc.groupby(col["batch"])[col["p_offline"]].max().rename("final_titer_g_L")
per_batch = rates.groupby("batch")[["mu", "qP", "qO2"]].mean().join(final_titer)
print(per_batch.corr()["final_titer_g_L"].round(3).to_string())
print(f"\nn = {len(per_batch)} batches. With n = 30, |r| must exceed about 0.36 "
      f"to reach 5% significance. Read these accordingly.")

# %% [markdown]
# ## 7. Deliverable (c): the reasoning in `derive_rates()`
#
# For each problem you find: which line, what is wrong with the reasoning, the
# corrected line, and **by what factor the result was off** — and what in the
# data sets that factor.
#
# *Our findings:*

# %% [markdown]
# ## Before you submit
#
# *Runtime → Restart session and run all.* Check that the execution numbers on
# the left run 1, 2, 3 … in order with no gaps, then *File → Download →
# Download .ipynb* and upload that file to the LMS. Output produced any other way
# is not accepted.

