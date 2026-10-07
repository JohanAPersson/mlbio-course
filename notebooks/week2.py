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
# # Week 2 — Validation and the leakage lab
#
# **Level: Specified.** The model is chosen for you and held fixed. Exactly one
# thing changes between the runs you make, and it is one word.
#
# There are no planted bugs this week. This notebook is correct. The wrong
# answer is one you will produce deliberately, by choosing it.
#
# ## The decision of the week
#
# `bioml.split(df, how=...)` has no default. `"random"` and `"batchwise"` are
# one word each and nothing in the API prefers either.

# %%
# --- Setup: install the course package and fetch the data --------------------
# Run this cell first, every session. In Colab it installs `bioml` and downloads
# the course data into /content/data; elsewhere it uses BIOML_DATA.
import os
import subprocess
import sys

# Pinned release: every pair runs the identical package and data (Decision 9).
BIOML_PIP = "git+https://github.com/COURSE-ORG/ml-bioprocesses.git@v1.0.0"
DATA_URL = ("https://github.com/COURSE-ORG/ml-bioprocesses/releases/download/"
            "v1.0.0/course-data-core.zip")

if "google.colab" in sys.modules:
    # subprocess rather than "!pip", so this file also runs as a plain script.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", BIOML_PIP], check=True)

import bioml

DATA_DIR = bioml.ensure_data(os.environ.get("BIOML_DATA", "/content/data"), DATA_URL)
print(bioml.about())

manifest = bioml.load_manifest()
if manifest.get("synthetic"):
    print("\n*** SYNTHETIC FIXTURE — code verification only, not IndPenSim ***\n")

# %%
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge              # penalised linear regression
from sklearn.metrics import r2_score                # coefficient of determination
from sklearn.pipeline import make_pipeline          # chain scaling + model as one object
from sklearn.preprocessing import StandardScaler    # rescale each column to mean 0, sd 1

# Same 30 recipe-driven batches as Week 1.
WEEK2_BATCHES = range(1, 31)
proc = bioml.load_process_vars(batches=WEEK2_BATCHES)
roles = bioml.load_column_roles()
col = bioml.resolve_many(proc, ["time", "batch", "p_offline"])

# %% [markdown]
# ## 1. Build the feature matrix
#
# Two decisions are made for you here, and both are stated.
#
# **Which rows.** The target, offline penicillin, is measured on ~2% of rows. A
# row without an assay has no target, so the modelling rows are the assay rows.
#
# **Which columns.** Only those classified `online_measurable`, plus elapsed
# time. That excludes `simulator_state` (`P:g/L` *is* the answer), the other
# offline assays, design metadata, identifiers and zero-variance columns.
#
# > **Predict before running the next cell.** How many modelling rows will
# > there be, and how many *independent* experiments do they come from?
# >
# > *Prediction:*

# %%
# Start from the role table rather than a hand-typed list, which would have to
# be kept in sync with the data by hand.
online = roles.loc[roles.role == "online_measurable", "column"].tolist()
feature_cols = [c for c in online if c in proc.columns]

# Drop columns that do not vary within these batches: they carry no information
# and make StandardScaler divide by zero.
nunique = proc[feature_cols].nunique(dropna=True)
feature_cols = [c for c in feature_cols if nunique[c] > 1]

# Elapsed process time is included on purpose: a real soft sensor knows how
# long the batch has been running. It is a decision; §2b examines its cost.
feature_cols = [col["time"]] + feature_cols

# Keep only rows where the target was measured; drop any with a missing input.
model_df = (proc.loc[proc[col["p_offline"]].notna(),
                     [col["batch"]] + feature_cols + [col["p_offline"]]]
                .dropna()
                .reset_index(drop=True))
X_cols, y_col = feature_cols, col["p_offline"]

n_b = model_df[col["batch"]].nunique()
print(f"{len(model_df)} modelling rows from {n_b} batches, {len(X_cols)} features")
print(f"~{len(model_df) / n_b:.0f} rows per batch; independent experiments: {n_b}")

# Assert rather than trust: if a forbidden column ever reached the feature list,
# every number below would be meaningless and nothing else would catch it.
forbidden = roles.loc[roles.role.isin(["simulator_state", "design_metadata"]), "column"].tolist()
leaked = [c for c in X_cols if c in forbidden]
assert not leaked, f"forbidden columns in the feature matrix: {leaked}"
print("no simulator_state or design_metadata column reached the feature matrix")

# %% [markdown]
# ## 2. One model, two validation protocols
#
# The model is fixed so that the only thing that changes is how the data is
# divided.
#
# > **Predict before running the next cell.** Write down the R² you expect for
# > `how="random"` and for `how="batchwise"`. Commit to two numbers.
# >
# > *Prediction:*

# %%
def make_model():
    """
    Standardise, then ridge regression.

    StandardScaler: columns are in very different units (K, L/h, g/min), and an
    unscaled penalised regression penalises large-unit columns less. Inside the
    pipeline, it is refitted on each training set and never sees the test set.

    Ridge rather than ordinary least squares: feeds, volume and time all rise
    together, so the inputs are strongly collinear and an unpenalised fit is
    unstable. alpha is fixed so that Week 2 varies exactly one thing.
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def fit_and_score(df: pd.DataFrame, how: str, cols: list[str]) -> dict:
    """Split `df` with the given strategy, fit on train, score on test."""
    train, test = bioml.split(df, how=how, test_size=0.3, seed=0)
    model = make_model().fit(train[cols], train[y_col])
    pred = model.predict(test[cols])
    return {
        "r2": r2_score(test[y_col], pred),
        "rmse": float(np.sqrt(np.mean((test[y_col] - pred) ** 2))),
        # How many batches sit on BOTH sides of the split: the mechanism as a count.
        "shared_batches": len(set(train[col["batch"]]) & set(test[col["batch"]])),
        "y_true": test[y_col].to_numpy(), "y_pred": pred,
    }


results = {}
for how in ["random", "batchwise"]:
    results[how] = fit_and_score(model_df, how, X_cols)   # the ONLY thing that varies is `how`
    r = results[how]
    print(f'how="{how}":{" " * (10 - len(how))} R² = {r["r2"]:6.3f}   '
          f'RMSE = {r["rmse"]:5.2f} g/L   batches in both train and test: {r["shared_batches"]}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
for ax, how in zip(axes, ["random", "batchwise"]):
    r = results[how]
    bioml.parity_plot(r["y_true"], r["y_pred"], ax=ax, title=f'how="{how}"   R² = {r["r2"]:.3f}')
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2b. What does the model know beyond the clock?
#
# In a recipe-driven process, titer rises with time in every batch. A model can
# score well simply by reading the clock. The honest comparison is therefore not
# "R² versus zero" but "R² versus a model that only knows elapsed time" — the
# simplest possible baseline, and the first instance of the course's
# baseline rule.
#
# > **Predict before running the next cell.** Will the full model beat the
# > clock-only model under `how="batchwise"`? By how much?
# >
# > *Prediction:*

# %%
baseline = {how: fit_and_score(model_df, how, [col["time"]]) for how in ["random", "batchwise"]}

table = pd.DataFrame({
    "features": ["all online", "all online", "time only", "time only"],
    "how": ["random", "batchwise", "random", "batchwise"],
    "R2": [results["random"]["r2"], results["batchwise"]["r2"],
           baseline["random"]["r2"], baseline["batchwise"]["r2"]],
    "RMSE (g/L)": [results["random"]["rmse"], results["batchwise"]["rmse"],
                   baseline["random"]["rmse"], baseline["batchwise"]["rmse"]],
})
print(table.round(3).to_string(index=False))

# %% [markdown]
# ## 3. Cross-validation, same distinction
#
# A single split is one draw. Cross-validation gives the spread, which matters
# more than the mean with only 30 batches.

# %%
X = model_df[X_cols].to_numpy(dtype=float)
y = model_df[y_col].to_numpy(dtype=float)
groups = model_df[col["batch"]].to_numpy()    # batch number per row

cv = {}
for how in ["random", "batchwise"]:
    cv[how] = bioml.cv_score(make_model(), X, y, groups=groups, how=how, n_splits=5)
    print(f'how="{how}":  R² per fold {np.array2string(cv[how], precision=3, floatmode="fixed")}'
          f'   mean {cv[how].mean():6.3f}   sd {cv[how].std():.3f}')

# %% [markdown]
# ## 4. Read the source — deliverable
#
# The next cell prints the source of `bioml.split`. Read it, then explain **in
# exactly two sentences** why `how="random"` inflates the score on this data. A
# complete answer names the unit of independence.
#
# *Our two sentences:*

# %%
# inspect.getsource() returns the Python source of a function as text.
print(inspect.getsource(bioml.split))

# %% [markdown]
# ## 5. Which protocol is right? — deliverable
#
# State what question each protocol answers, and a situation in which each is
# the correct choice. A third strategy, `how="regime"`, holds out a whole
# control regime; there is only one regime in batches 1–30, so it cannot be run
# here, but say what question it would answer.
#
# *Our answer:*

# %% [markdown]
# ## Before you submit
#
# *Runtime → Restart session and run all*, check the execution numbers run
# 1, 2, 3 … in order, then *File → Download → Download .ipynb* and upload to the
# LMS.

