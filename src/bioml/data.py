"""
bioml.data — fetching and loading the distributed IndPenSim Parquet files.

These functions read what `prepare_indpensim.py` wrote. They do no cleaning,
no imputation, no feature selection and no reshaping. Anything that looks like
a modelling choice is deliberately absent (Decision 9's exclusion list).

WHERE THE DATA COMES FROM
-------------------------
The course data is published as individual files on an immutable GitHub
release (tag DATA_TAG below). A tag is never edited after publication: if the
data ever changes, a new tag is published and a new bioml release points at it.
Pinning the bioml version in the setup cell therefore also pins the data
version — every pair works on byte-identical files.

DATA LAYOUT EXPECTED
--------------------
    <data_dir>/
        process_vars.parquet    all 100 batches, all non-Raman columns
        column_roles.csv        one row per column: column, role, reason
        manifest.json           what the conversion found in the source file
        raman/batch_061.parquet ... batch_090.parquet   (from Week 3)
        fault_labels.parquet    only if --withhold-fault-labels was used
                                (never published on the public release)

In Colab, the first cell of every notebook calls `ensure_data()`. Weeks 1-2
download only the three core files (~20 MB); from Week 3 the same call is
given the Raman batches to fetch. On the instructor's machine, set the
BIOML_DATA environment variable to the local output of prepare_indpensim.py:
all files are then already present and nothing is downloaded.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .columns import resolve

# ---------------------------------------------------------------------------
# Where the data lives
# ---------------------------------------------------------------------------

# The GitHub release holding the data files. Changing this value is a new
# bioml release, never a silent edit (Decision 9: bioml is frozen in term).
DATA_TAG = "data-v1"
DATA_URL = f"https://github.com/OWNER/mlbio-course/releases/download/{DATA_TAG}"

# Placeholder repository owner. If it is still in the URL, hosting has not
# been set up yet and every download would fail with an unhelpful 404.
_PLACEHOLDER = "/OWNER/"

# Files that must exist for Weeks 1-2 (~20 MB in total).
_CORE_FILES = ("process_vars.parquet", "column_roles.csv", "manifest.json")

# Only these batches have Raman spectra: PAT_ref = 2 ("Raman recorded") in the
# file. This is a fact about the data, not a modelling choice.
RAMAN_BATCHES = range(61, 91)

# Seconds to wait for the server before giving up on a download.
_TIMEOUT_S = 120

# Module-level default. Overridden by set_data_dir(), ensure_data() or BIOML_DATA.
_DATA_DIR = Path(os.environ.get("BIOML_DATA", "data"))


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


def ensure_data(
    raman_batches: Iterable[int] | None = None,
    dest: str | Path | None = None,
    url: str = DATA_URL,
) -> Path:
    """
    Make sure the course data is present locally, downloading what is missing.

    raman_batches : batch numbers whose Raman spectra you need, e.g.
                    range(61, 91). Leave as None in Weeks 1-2: only the three
                    core files (~20 MB) are downloaded.
    dest          : folder to hold the data. Default: the BIOML_DATA
                    environment variable if set (the instructor's local data),
                    otherwise data/<DATA_TAG>. The tag is part of the folder
                    name, so files from two data versions can never be mixed.
    url           : base address of the release. Students never change this.

    Files already on disk are skipped, so re-running the setup cell is cheap.
    Afterwards every bioml loader reads from `dest`. Returns `dest` as a Path.

    This is plumbing, not judgement: it copies the files prepare_indpensim.py
    produced, unchanged. Nothing is cleaned, filtered or re-typed on the way.
    """
    if dest is None:
        dest = os.environ.get("BIOML_DATA", f"data/{DATA_TAG}")
    dest = Path(dest)

    # Build the list of (file name on the release, where to save it locally).
    wanted = [(name, dest / name) for name in _CORE_FILES]
    for b in raman_batches or []:
        b = int(b)
        if b not in RAMAN_BATCHES:
            raise ValueError(
                f"Batch {b} has no Raman spectra: only batches "
                f"{RAMAN_BATCHES.start}-{RAMAN_BATCHES.stop - 1} had the probe "
                f"recording (PAT_ref = 2). Check column_roles.csv and the manifest."
            )
        name = f"batch_{b:03d}.parquet"          # naming used by prepare_indpensim.py
        wanted.append((name, dest / "raman" / name))

    missing = [(name, target) for name, target in wanted if not target.exists()]

    # Only check the placeholder if something actually has to be downloaded,
    # so the instructor's local copy works before hosting is set up.
    if missing and _PLACEHOLDER in url:
        raise RuntimeError(
            "The data URL in bioml is still a placeholder. This is a "
            "course-preparation error, not yours — tell the instructor."
        )

    if missing:
        print(f"downloading {len(missing)} file(s) from release {DATA_TAG} ...")
    for name, target in missing:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Download to a temporary name, then rename. An interrupted download
        # never leaves a half-written file that a later run would skip.
        tmp = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(f"{url}/{name}", timeout=_TIMEOUT_S) as r, \
                    open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {name} ({e}). Re-run the setup cell; "
                f"if it fails again, tell the instructor."
            ) from e
        tmp.replace(target)

    print(f"course data ({DATA_TAG}) ready in {dest.resolve()}")
    # Point the loaders at this folder explicitly, rather than relying on a
    # default path that happens to match.
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
    hint = "In the setup cell, run bioml.ensure_data(raman_batches=range(61, 91))."
    if not raman_dir.exists():
        raise FileNotFoundError(f"{raman_dir} not found. {hint}")

    # Batch numbers already downloaded, read from the file names.
    available = sorted(int(p.stem.split("_")[1]) for p in raman_dir.glob("batch_*.parquet"))
    if not available:
        # The folder can exist but be empty; say so instead of failing obscurely.
        raise FileNotFoundError(f"No Raman files downloaded yet. {hint}")

    out: dict[int, pd.DataFrame] = {}
    for b in batches:
        b = int(b)
        path = raman_dir / f"batch_{b:03d}.parquet"
        if not path.exists():
            reason = (f"batch {b} has no spectra (only {RAMAN_BATCHES.start}-"
                      f"{RAMAN_BATCHES.stop - 1} do)." if b not in RAMAN_BATCHES
                      else f"batch {b} has not been downloaded. {hint}")
            raise FileNotFoundError(
                f"No Raman file: {reason} Downloaded so far: {len(available)} "
                f"batch(es), {available[0]}-{available[-1]}."
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
