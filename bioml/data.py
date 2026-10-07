"""
bioml.data — fetching and loading the distributed IndPenSim Parquet files.

These functions read what `prepare_indpensim.py` wrote. They do no cleaning,
no imputation, no feature selection and no reshaping. Anything that looks like
a modelling choice is deliberately absent (Decision 9's exclusion list).

DATA LAYOUT EXPECTED
--------------------
    <data_dir>/
        process_vars.parquet    all 100 batches, all non-Raman columns
        column_roles.csv        one row per column: column, role, reason
        manifest.json           what the conversion found in the source file
        raman/batch_061.parquet ... batch_090.parquet   (from Week 3)
        fault_labels.parquet    only if --withhold-fault-labels was used

In Colab, the first cell of every notebook calls `ensure_data()`, which
downloads and unpacks the course data once per session. Elsewhere, set the
directory with the BIOML_DATA environment variable or `set_data_dir()`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .columns import resolve

# Module-level default. Overridden by set_data_dir(), ensure_data() or BIOML_DATA.
_DATA_DIR = Path(os.environ.get("BIOML_DATA", "data"))

# Files that must exist for Weeks 1-2. Raman files are fetched separately from
# Week 3, so a ~20 MB download is all the first two weeks ever need.
_CORE_FILES = ("process_vars.parquet", "column_roles.csv", "manifest.json")


def set_data_dir(path: str | Path) -> Path:
    """Point bioml at the folder holding process_vars.parquet. Returns the path."""
    global _DATA_DIR
    _DATA_DIR = Path(path)
    if not _DATA_DIR.exists():
        raise FileNotFoundError(f"data directory does not exist: {_DATA_DIR.resolve()}")
    return _DATA_DIR


def data_dir() -> Path:
    """The folder bioml is currently reading from."""
    return _DATA_DIR


def ensure_data(dest: str | Path, url: str | None = None) -> Path:
    """
    Make sure the course data is present in `dest`, downloading it if needed.

    dest : folder to hold the data. If it already contains the core files, no
           download happens — so re-running the setup cell is cheap, and the
           instructor's machine (with BIOML_DATA pointing at local data) never
           touches the network.
    url  : address of the course data zip. Only used if the files are missing.

    Afterwards bioml reads from `dest`. Returns `dest` as a Path.

    This is plumbing, not judgement: it copies files that `prepare_indpensim.py`
    produced, unchanged. Nothing is cleaned, filtered or re-typed on the way.
    """
    dest = Path(dest)
    if all((dest / f).exists() for f in _CORE_FILES):
        return set_data_dir(dest)

    if not url:
        raise FileNotFoundError(
            f"course data not found in {dest.resolve()} and no download URL given."
        )
    if "COURSE-ORG" in url:
        # Guard against shipping a notebook before the hosting location is set.
        raise RuntimeError(
            "The data URL in the setup cell is still a placeholder. "
            "This is a course-preparation error, not yours — tell the instructor."
        )

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "course_data.zip"
        print(f"downloading course data from {url} ...")
        with urllib.request.urlopen(url) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)

    missing = [f for f in _CORE_FILES if not (dest / f).exists()]
    if missing:
        raise FileNotFoundError(f"download did not contain {missing}; check the zip layout")
    print(f"course data ready in {dest.resolve()}")
    return set_data_dir(dest)


# ---------------------------------------------------------------------------
# Process variables
# ---------------------------------------------------------------------------

def load_process_vars(
    batches: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load the non-Raman block.

    batches : iterable of batch numbers (1-100), or None for all 100.
    columns : actual column names to read, or None for all. Parquet is
              columnar, so naming columns reads only those columns off disk —
              the whole reason the course does not ship CSV.

    Rows are returned sorted by (batch, time), with a plain RangeIndex: an
    accidental index-alignment join is a bug class this cohort cannot debug.
    """
    path = _DATA_DIR / "process_vars.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the setup cell, or call "
            f"bioml.set_data_dir(...) to point at the right folder."
        )

    # If the caller names columns, make sure the identifiers come along — a
    # frame without Batch ID cannot be split correctly, and returning one would
    # invite exactly the leakage the course is about.
    read_cols = None
    if columns is not None:
        head = pd.read_parquet(path, engine="pyarrow").head(0)
        ident = [resolve(head, "batch"), resolve(head, "time")]
        read_cols = ident + [c for c in columns if c not in ident]

    df = pd.read_parquet(path, columns=read_cols, engine="pyarrow")

    batch_col = resolve(df, "batch")
    time_col = resolve(df, "time")

    if batches is not None:
        wanted = [int(b) for b in batches]
        df = df[df[batch_col].isin(wanted)]
        missing = sorted(set(wanted) - set(df[batch_col].unique().tolist()))
        if missing:
            raise ValueError(f"batches not present in the file: {missing}")

    return df.sort_values([batch_col, time_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Raman spectra (Week 3 onward)
# ---------------------------------------------------------------------------

def load_raman(batches: Iterable[int]) -> dict[int, pd.DataFrame]:
    """
    Load Raman spectra, ONE FRAME PER BATCH: {batch_number: DataFrame}.

    Each frame has a Time column and the wavenumber columns; it has NO Batch ID
    column, because the batch number is the dictionary key.

    Deliberately does not concatenate (Decision 10). Stacking batches 61-90,
    labelling each row with its batch, and joining to the ~98%-sparse offline
    assay is the Week 3 lab. Only batches 61-90 have spectra (PAT_ref = 2).
    """
    raman_dir = _DATA_DIR / "raman"
    if not raman_dir.exists():
        raise FileNotFoundError(f"{raman_dir} not found (Raman data is fetched from Week 3).")

    out: dict[int, pd.DataFrame] = {}
    for b in batches:
        b = int(b)
        path = raman_dir / f"batch_{b:03d}.parquet"
        if not path.exists():
            available = sorted(int(p.stem.split("_")[1]) for p in raman_dir.glob("*.parquet"))
            raise FileNotFoundError(
                f"no Raman file for batch {b}. Available: "
                f"{available[0]}-{available[-1]} ({len(available)} batches)."
            )
        out[b] = pd.read_parquet(path, engine="pyarrow")
    return out


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_column_roles() -> pd.DataFrame:
    """
    The column-role table: one row per column, with role and reason.

    Roles: identifier, design_metadata, simulator_state, offline_assay,
    online_measurable, raman, dead. `simulator_state` columns exist in the file
    and not on a plant; using one as a feature is not a bug any validation
    protocol will catch.
    """
    path = _DATA_DIR / "column_roles.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)


def load_manifest() -> dict:
    """
    manifest.json — what the conversion measured in the source file.

    Use it instead of hard-coding numbers. The offline sampling interval is a
    property of the data, not a constant: an earlier version of this course
    assumed it and was wrong by a factor of ten.
    """
    path = _DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return json.loads(path.read_text())


def load_fault_labels() -> pd.DataFrame:
    """The withheld Week 6 answer key, if the conversion withheld it."""
    path = _DATA_DIR / "fault_labels.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Either the conversion was run without "
            f"--withhold-fault-labels (so Fault flag is in process_vars.parquet), "
            f"or you are pointed at the wrong data directory."
        )
    return pd.read_parquet(path, engine="pyarrow")
