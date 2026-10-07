#!/usr/bin/env python3
"""
prepare_indpensim.py — convert the IndPenSim distribution CSV into the Parquet
files used in ML for Bioprocesses.

This script is published to students as reference material. It is the answer to
"what was done to my data before I got it, and why". Read it before Week 1.

================================================================================
1. WHY THIS EXISTS

The IndPenSim distribution is a single CSV of ~2.57 GB. A plain pd.read_csv()
peaks at several times that in RAM. On an 8 GB laptop that is a swap-thrash or
an outright kill.

The size comes from the SHAPE of the file, not from how much information is in
it: ~113,935 rows (100 batches x ~1,139 time points at 0.2 h) x 2,237 columns,
of which 2,200 are Raman wavenumber channels that are near-perfectly collinear
with each other.

Note the ratio before you model anything:
    2.57 GB of file, n = 100 independent experimental runs.
Rows are not independent samples. Everything the course says about validation
follows from that one fact.

================================================================================
2. THE HEADER DEFECT (read this part properly)

The distributed CSV has 2,239 header fields but 2,237 data fields per row.

Cause: one column name contains unquoted commas. The header field is

    1- No Raman spec, 2-Raman spec recorded, 3-PAT control(PAT_ref:PAT ref)

which a CSV parser splits into THREE names for ONE column. Three names, one
data field, so the header runs two long.

Consequence, if you do nothing: pandas assigns names left-to-right and pads the
last two with NaN. Every column from that point onward is labelled with the
name of the column two positions earlier. In particular:

    labelled 'Batch ID'                   actually holds  Raman channel 2400
    labelled 'Batch reference(Batch_ref)' actually holds  Fault flag
    labelled '2-PAT control(PAT_ref)'     actually holds  Batch ID

A column named 'Batch ID' containing Raman intensities in the tens of thousands
is not a hypothetical hazard. It is what this file does. Anyone who groups by
'Batch ID' as distributed gets 47,361 groups instead of 100 batches, and every
train/test split built on it is meaningless.

The fix is to merge the three split header fields back into one name. That
yields exactly 2,237 names, which then align with the data end to end. No
positional shift, no guessing. This script verifies the merged length against
the actual field count and refuses to run if they disagree.

Verified mapping after repair (columns 1-33 were never affected):
    34  PAT_ref (the merged column)
    35  Batch reference(Batch_ref)
    36  Batch ID                     <- 100 unique, 1-100, the real identifier
    37  Fault flag
    38  Raman 2400 ... 2237  Raman 201      (2,200 channels)

================================================================================
3. COLUMN ROLES — the observability boundary

Not every column in this file exists on a real plant. The dataset is a
simulator dump, so it contains simulator STATE variables alongside the sensor
readings a plant would actually have. Two of them matter enormously:

    Penicillin concentration(P:g/L)   -- simulator ground truth for titer
    Substrate concentration(S:g/L)    -- no routine online glucose at this scale

There is no online penicillin probe on a 100,000 L fermenter. That is the whole
reason Raman soft sensing exists. If you leave P:g/L in a feature matrix
predicting P_offline you will get R^2 near 1.0 and you will have learned
nothing, because you have handed the model the answer.

This script does NOT delete those columns. It labels them, in column_roles.csv,
with the reason. Deciding what you are allowed to use is part of the work.

Roles emitted:
    identifier        Time, Batch ID, Batch reference
    design_metadata   experimental design flags, not measurements
    simulator_state   exists in the simulator, not on a plant
    offline_assay     sparse laboratory measurement (~98% missing)
    online_measurable a real plant sensor would provide this
    raman             spectral channel
    dead              zero variance across the whole file

================================================================================
4. WHAT THE FILE ACTUALLY CONTAINS (verified, not assumed)

Everything below was checked against the file rather than taken from the
dataset's published description. Where the two disagree, that is noted.

Time grid. Strictly increasing within every batch. No duplicate timestamps, no
non-monotonic steps. Uniform 0.2 h. The raw CSV stores clean decimals (0.2,
0.4, 0.6, ...), so 'Time (h)' is kept in float64 here. Casting it to float32
introduces representation noise of order 1e-5 h — large enough to be visible in
any histogram of the step size, and manufactured entirely by this script.
Every other float column is float32.

Be clear about what that fixes and what it does not. In float64 the step is
correct to ~2e-14; in float32 it is correct to ~1e-5. But 0.2 has no exact
binary representation in EITHER type, so

    dt == 0.2

is still false for most rows after the fix. That residual is floating-point
arithmetic, not this dataset and not this script; use np.isclose, or work from
the actual differences. The distinction matters: one of those two hazards we
would have created and then taught as if it were the world.

Batch length. 835-1,450 rows, i.e. 167-290 h. That is a 74% spread. Batches
cannot be stacked into a rectangular time-by-batch array without alignment.
This is a real property of the dataset, not a defect.

Regimes. Control_ref and PAT_ref are constant within every batch, so both are
batch-level design labels. The observed combinations are:

    batches   1-30    Control_ref 0 (recipe)    PAT_ref 1 (no Raman)
    batches  31-60    Control_ref 1 (operator)  PAT_ref 1 (no Raman)
    batches  61-90    Control_ref 0 (recipe)    PAT_ref 2 (Raman recorded)
    batches  91-100   Control_ref 0 (recipe)    PAT_ref 1 (no Raman)

PAT_ref = 3 ('PAT control') does not occur anywhere in the file.

Note what that means, and note that the published description of this dataset
says otherwise. By the file's own design flags, batches 61-90 are recipe-driven
runs with a Raman probe RECORDING; they are not runs where Raman closes a
control loop. Whether the spectrometer is actually acting on those batches is
decidable from the feed trajectories, and this script does not decide it for
you. Metadata is a claim. Signals are evidence.

Note also that batches 91-100 (fault-injected) carry design labels identical to
batches 1-30. There is no metadata route to identifying which runs contain a
fault. 'Fault flag' is the only tell, which is why --withhold-fault-labels
exists and why it is airtight.

Batch reference. Byte-identical to Batch ID. Redundant; kept and labelled.

================================================================================
5. USAGE

    # look before you leap: header repair + classification, writes nothing
    python prepare_indpensim.py --inspect --input 100_Batches_IndPenSim_V3.csv

    # convert
    python prepare_indpensim.py --input 100_Batches_IndPenSim_V3.csv --outdir data/

    # convert, holding fault labels out of the main file (Week 6 answer key)
    python prepare_indpensim.py --input ... --outdir data/ --withhold-fault-labels

Requires: pandas, pyarrow.   (pip install pandas pyarrow)

DATA CITATION — required by CC BY 4.0, keep it attached to the data:
  Goldrick, S., Stefan, A., Lovett, D., Montague, G., Lennox, B. (2015).
  The development of an industrial-scale fed-batch fermentation simulation.
  Dataset: Mendeley Data, https://data.mendeley.com/datasets/pdnjz7zz5x/2
  Licence: CC BY 4.0
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPECTED_BATCHES = 100
SAMPLE_INTERVAL_H = 0.2

TIME_COL = "Time (h)"
BATCH_COL = "Batch ID"

# Columns kept in float64. See to_float32() for why this is not an oversight.
FLOAT64_KEEP = (TIME_COL,)

# Tolerance for calling the time grid uniform, in hours. float64 parsing of
# clean decimals lands well inside this; float32 does not.
GRID_TOL_H = 1e-9

# ------------------------------------------------------------------------------
# Section A — header repair
# ------------------------------------------------------------------------------

# The three fragments of the split column name, in order.
FRAGMENT_START = re.compile(r"no\s+raman\s+spec", re.IGNORECASE)
FRAGMENT_END = re.compile(r"pat\s*control", re.IGNORECASE)
MERGED_NAME = "PAT_ref"


def read_header_and_width(path: Path) -> tuple[list[str], int]:
    """Return the raw header fields and the number of fields in the first data row."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)
    return [h.strip() for h in header], len(first_row)


def repair_header(header: list[str], n_data_fields: int) -> tuple[list[str], dict]:
    """
    Merge the comma-split column name back into one field.

    Returns (repaired_names, report). Exits if the repair does not produce a
    name list matching the data width — a silent mismatch here corrupts every
    column downstream, so failing loudly is the only safe behaviour.
    """
    report = {"raw_header_fields": len(header), "data_fields": n_data_fields}

    if len(header) == n_data_fields:
        report["repair"] = "none needed"
        return header, report

    start = next((i for i, h in enumerate(header) if FRAGMENT_START.search(h)), None)
    end = next((i for i, h in enumerate(header) if FRAGMENT_END.search(h)), None)

    if start is None or end is None or end < start:
        sys.exit(
            f"\nERROR: header has {len(header)} fields but data rows have "
            f"{n_data_fields}, and the known comma-split column name was not "
            f"found. Do not convert — the column mapping is unknown. Inspect the "
            f"header manually."
        )

    fragments = header[start : end + 1]
    repaired = header[:start] + [MERGED_NAME] + header[end + 1 :]

    if len(repaired) != n_data_fields:
        sys.exit(
            f"\nERROR: merged the split name at positions {start + 1}-{end + 1} "
            f"but got {len(repaired)} names for {n_data_fields} data fields. "
            f"The defect is not the one this script knows how to repair. "
            f"Do not convert."
        )

    report["repair"] = "merged comma-split column name"
    report["merged_positions"] = [start + 1, end + 1]
    report["merged_fragments"] = fragments
    report["merged_into"] = MERGED_NAME
    report["dropped_phantom_names"] = header[-(len(header) - n_data_fields) :]
    return repaired, report


# ------------------------------------------------------------------------------
# Section B — column roles
# ------------------------------------------------------------------------------

RAMAN_NAME = re.compile(r"^\s*\d+(\.\d+)?\s*$")

IDENTIFIERS = {
    "Time (h)": "process time within the batch",
    "Batch ID": "batch number, 1-100 — the statistically independent unit",
    "Batch reference(Batch_ref:Batch ref)": (
        "secondary batch index; verified byte-identical to Batch ID — redundant, "
        "carries no additional information"
    ),
}

DESIGN_METADATA = {
    "Fault reference(Fault_ref:Fault ref)": "which fault was injected; experimental design, not a measurement",
    "Fault flag": "ANSWER KEY for Week 6 fault detection — do not use as a feature",
    "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)": (
        "control regime label, constant within a batch; 0 = recipe driven "
        "(batches 1-30, 61-90, 91-100), 1 = operator controlled (batches 31-60)"
    ),
    "PAT_ref": (
        "Raman/PAT regime label, constant within a batch; 1 = no Raman, "
        "2 = Raman recorded (batches 61-90), 3 = PAT control (never occurs in "
        "this file, despite the published description of batches 61-90)"
    ),
}

SIMULATOR_STATE = {
    "Penicillin concentration(P:g/L)": (
        "SIMULATOR GROUND TRUTH. No online penicillin sensor exists on a 100,000 L "
        "fermenter — titer is why Raman soft sensing exists. Using this as a feature "
        "to predict P_offline leaks the target completely."
    ),
    "Substrate concentration(S:g/L)": (
        "SIMULATOR STATE. Routine online glucose measurement is not standard at "
        "production scale; substrate is inferred, not measured."
    ),
}

OFFLINE_ASSAY_PATTERN = re.compile(r"offline", re.IGNORECASE)


def classify(names: list[str], constant_cols: set[str]) -> dict[str, tuple[str, str]]:
    """Map each column name to (role, reason)."""
    roles: dict[str, tuple[str, str]] = {}
    for name in names:
        if name in IDENTIFIERS:
            roles[name] = ("identifier", IDENTIFIERS[name])
        elif name in DESIGN_METADATA:
            roles[name] = ("design_metadata", DESIGN_METADATA[name])
        elif name in SIMULATOR_STATE:
            roles[name] = ("simulator_state", SIMULATOR_STATE[name])
        elif RAMAN_NAME.match(name):
            roles[name] = ("raman", f"Raman channel at {name.strip()} cm-1")
        elif OFFLINE_ASSAY_PATTERN.search(name):
            roles[name] = ("offline_assay", "sparse laboratory measurement")
        elif name in constant_cols:
            roles[name] = ("dead", "zero variance across the whole file")
        else:
            roles[name] = ("online_measurable", "available from a plant sensor")
    return roles


# ------------------------------------------------------------------------------
# Section C — inspection
# ------------------------------------------------------------------------------

def inspect(path: Path, preview_rows: int = 3000):
    header, width = read_header_and_width(path)
    names, report = repair_header(header, width)

    print("=" * 78)
    print(f"FILE   {path}")
    print(f"SIZE   {path.stat().st_size / 1e9:.2f} GB")
    print("=" * 78)
    print("\nHEADER REPAIR")
    print(f"  raw header fields   {report['raw_header_fields']}")
    print(f"  data fields         {report['data_fields']}")
    print(f"  action              {report['repair']}")
    if "merged_fragments" in report:
        print(f"  merged positions    {report['merged_positions'][0]}"
              f"-{report['merged_positions'][1]} -> '{MERGED_NAME}'")
        for frag in report["merged_fragments"]:
            print(f"      | {frag!r}")
        print(f"  phantom names dropped: {report['dropped_phantom_names']}")
    print(f"  repaired name count {len(names)}  (matches data width)")

    non_raman = [n for n in names if not RAMAN_NAME.match(n)]
    raman = [n for n in names if RAMAN_NAME.match(n)]

    sample = pd.read_csv(path, names=names, skiprows=1, nrows=preview_rows,
                         usecols=non_raman, low_memory=False)
    constant = {c for c in sample.columns if sample[c].nunique(dropna=True) <= 1}
    roles = classify(names, constant)

    print(f"\nCOLUMNS  {len(names)} total = {len(non_raman)} non-Raman "
          f"+ {len(raman)} Raman ({raman[0]} .. {raman[-1]} cm-1)")

    by_role: dict[str, list[str]] = {}
    for n in non_raman:
        by_role.setdefault(roles[n][0], []).append(n)

    for role in ["identifier", "design_metadata", "simulator_state",
                 "offline_assay", "online_measurable", "dead"]:
        cols = by_role.get(role, [])
        if not cols:
            continue
        print(f"\n{role.upper()} ({len(cols)}):")
        for c in cols:
            nan = sample[c].isna().mean() if c in sample else float("nan")
            flag = "  <-- LEAKS TARGET" if role == "simulator_state" else ""
            print(f"    {c[:60]:<62} nan={nan:4.0%}{flag}")

    print("\n" + "-" * 78)
    print("VERIFY BEFORE CONVERTING:")
    print("  - 'Batch ID' should have 100 unique values (checked during convert)")
    print("  - Temperature ~298 K, pH ~6.5, Volume 5.7e4-9.0e4 L")
    print("  - simulator_state columns are labelled, not deleted — that is deliberate")
    print("-" * 78)
    return names, roles


# ------------------------------------------------------------------------------
# Section D — conversion
# ------------------------------------------------------------------------------

PHYSICS_CHECKS = [
    ("Temperature(T:K)", 295.0, 303.0, "controlled at ~298 K"),
    ("pH(pH:pH)", 5.0, 7.5, "controlled at ~6.5"),
    ("Vessel Volume(V:L)", 4.0e4, 1.0e5, "fed-batch fill of a 100,000 L vessel"),
    ("Oxygen in percent in off-gas(O2:O2  (%))", 0.10, 0.25, "~21% falling with respiration"),
    ("Dissolved oxygen concentration(DO2:mg/L)", 0.0, 20.0, "aerobic culture"),
]


def to_float32(df: pd.DataFrame, exclude: tuple[str, ...] = FLOAT64_KEEP) -> pd.DataFrame:
    """
    Downcast float64 -> float32, except for columns in `exclude`.

    'Time (h)' is excluded deliberately. The source CSV stores clean decimals
    (0.2, 0.4, ...) and float32 cannot represent them to better than ~1e-7
    relative, which turns a uniform 0.2 h step into several distinct values in
    the 1e-5 h range. Any student filter of the form `dt == 0.2` then returns
    nothing, for a reason that has nothing to do with the process and
    everything to do with this script. One float64 column costs ~0.9 MB.
    """
    cols = df.select_dtypes(include=["float64"]).columns.difference(exclude)
    if len(cols):
        df[cols] = df[cols].astype("float32")
    return df


def convert(path: Path, outdir: Path, chunksize: int, withhold_faults: bool,
            expect_batches: int = EXPECTED_BATCHES):
    names, roles = inspect(path, preview_rows=3000)

    raman_cols = [n for n in names if roles[n][0] == "raman"]
    non_raman = [n for n in names if roles[n][0] != "raman"]

    batch_col = BATCH_COL
    time_col = TIME_COL
    if batch_col not in names or time_col not in names:
        sys.exit(f"\nERROR: expected '{batch_col}' and '{time_col}' after repair.")

    fault_cols = [n for n in names if roles[n][0] == "design_metadata"
                  and "fault" in n.lower()]

    outdir.mkdir(parents=True, exist_ok=True)
    raman_dir = outdir / "raman"
    raman_dir.mkdir(exist_ok=True)

    proc_parts: list[pd.DataFrame] = []
    buffers: dict[int, list[pd.DataFrame]] = {}
    n_rows = 0
    raman_read = [batch_col, time_col] + raman_cols

    print(f"\nStreaming in chunks of {chunksize:,} rows...")
    reader = pd.read_csv(path, names=names, skiprows=1, chunksize=chunksize,
                         low_memory=False)

    for i, chunk in enumerate(reader, start=1):
        n_rows += len(chunk)
        proc_parts.append(to_float32(chunk[non_raman].copy()))

        sub = to_float32(chunk[raman_read].copy())
        for bid, group in sub.groupby(batch_col, sort=False):
            buffers.setdefault(int(bid), []).append(group)
        open_batch = int(sub[batch_col].iloc[-1])
        for bid in [b for b in list(buffers) if b != open_batch]:
            _write_batch(buffers.pop(bid), bid, batch_col, raman_dir)

        print(f"  chunk {i:>4}   rows {n_rows:,}", end="\r", flush=True)

    for bid in list(buffers):
        _write_batch(buffers.pop(bid), bid, batch_col, raman_dir)
    print()

    proc = pd.concat(proc_parts, ignore_index=True)
    del proc_parts
    proc[batch_col] = proc[batch_col].astype("int16")
    proc = proc.sort_values([batch_col, time_col]).reset_index(drop=True)

    # --- assertions -----------------------------------------------------------
    problems = []
    n_batches = proc[batch_col].nunique()
    if n_batches != expect_batches:
        problems.append(f"expected {expect_batches} batches, found {n_batches}")

    for col, lo, hi, why in PHYSICS_CHECKS:
        if col not in proc:
            problems.append(f"missing expected column {col!r}")
            continue
        mn, mx = float(proc[col].min()), float(proc[col].max())
        if mn < lo or mx > hi:
            problems.append(f"{col}: range [{mn:.4g}, {mx:.4g}] outside "
                            f"[{lo}, {hi}] — {why}")

    if problems:
        print("\n" + "!" * 78)
        print("PHYSICS / STRUCTURE CHECKS FAILED — the column mapping may be wrong:")
        for p in problems:
            print(f"  - {p}")
        print("!" * 78)
        sys.exit("Refusing to write. Investigate before converting.")

    # --- timestamp integrity --------------------------------------------------
    dt = proc.groupby(batch_col)[time_col].diff()
    nonincreasing = proc.loc[dt.notna() & (dt <= 0), batch_col].unique().tolist()
    duplicate_ts = int((dt == 0).sum())

    # Grid uniformity. If this reports more than one step, either the source is
    # non-uniform or something upstream has quantised the column — check the
    # dtype of time_col before blaming the simulator.
    dt_clean = dt.dropna()
    try:
        off_grid = int((dt_clean.sub(SAMPLE_INTERVAL_H).abs() > GRID_TOL_H).sum())
        dt_unique = sorted(float(v) for v in dt_clean.round(9).unique())[:10]
    except Exception as exc:  # noqa: BLE001
        print(f"\n  NOTE  could not check grid uniformity: {exc!r}")
        off_grid, dt_unique = None, []

    # --- batch durations ------------------------------------------------------
    # Everything from here to the write is DIAGNOSTIC. The expensive streaming
    # pass is already done, so a failure in reporting must never cost the
    # conversion. Structural correctness is enforced above, by the physics and
    # batch-count checks, which do still exit.
    def _safe(label, fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — diagnostics must not be fatal
            print(f"\n  NOTE  could not compute {label}: {exc!r}")
            print("        conversion continues; this field is absent from the manifest.")
            return default

    duration_stats = _safe("batch durations", lambda: {
        "min_h": round(float(proc.groupby(batch_col)[time_col].max().min()), 2),
        "median_h": round(float(proc.groupby(batch_col)[time_col].max().median()), 2),
        "max_h": round(float(proc.groupby(batch_col)[time_col].max().max()), 2),
        "note": ("batch lengths differ by ~74%; trajectories are not "
                 "commensurable without alignment (matters for Week 6 MSPC)"),
    }, None)

    # --- regime structure -----------------------------------------------------
    regimes = _safe("regime summary",
                    lambda: _regime_summary(proc, roles, batch_col),
                    {"available": False, "reason": "computation failed"})

    # --- redundant identifier check -------------------------------------------
    ref_col = next((n for n in non_raman if "Batch_ref" in n), None)
    ref_identical = _safe("Batch reference identity check", lambda: bool(
        ref_col is not None
        and ref_col in proc
        and (proc[ref_col].astype("float64")
             == proc[batch_col].astype("float64")).all()
    ), None)

    # --- offline sampling structure -------------------------------------------
    offline = [n for n in non_raman if roles[n][0] == "offline_assay"]
    offline_stats = {}
    for col in offline:
        frac = float(proc[col].notna().mean())
        offline_stats[col] = {
            "fraction_observed": round(frac, 5),
            "approx_rows_per_label": round(1 / frac, 1) if frac else None,
            "approx_hours_per_label": round(SAMPLE_INTERVAL_H / frac, 2) if frac else None,
        }

    # --- write ----------------------------------------------------------------
    if withhold_faults and fault_cols:
        labels = proc[[batch_col, time_col] + fault_cols].copy()
        labels.to_parquet(outdir / "fault_labels.parquet", index=False)
        proc = proc.drop(columns=fault_cols)

    proc_path = outdir / "process_vars.parquet"
    proc.to_parquet(proc_path, index=False, compression="snappy")

    roles_rows = [{"column": n, "role": roles[n][0], "reason": roles[n][1]}
                  for n in names if roles[n][0] != "raman"]
    roles_rows.append({"column": f"<{len(raman_cols)} Raman channels: "
                                 f"{raman_cols[0]}..{raman_cols[-1]} cm-1>",
                       "role": "raman",
                       "reason": "spectral channels, stored per batch in raman/"})
    pd.DataFrame(roles_rows).to_csv(outdir / "column_roles.csv", index=False)

    sizes = proc.groupby(batch_col).size()
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": path.name,
        "source_size_bytes": path.stat().st_size,
        "source_dataset": "IndPenSim — Industrial Penicillin Fermentation Simulator",
        "source_url": "https://data.mendeley.com/datasets/pdnjz7zz5x/2",
        "source_reference": ("Goldrick, S., Stefan, A., Lovett, D., Montague, G., "
                             "Lennox, B. (2015). The development of an industrial-scale "
                             "fed-batch fermentation simulation."),
        "licence": "CC BY 4.0",
        "header_defect": (
            "Distributed CSV has 2 more header fields than data fields because one "
            "column name contains unquoted commas. Repaired by merging the split "
            "name; verified against the data field count. Without this repair the "
            "column labelled 'Batch ID' holds Raman channel 2400."
        ),
        "rows": int(n_rows),
        "n_batches": int(n_batches),
        "rows_per_batch": {"min": int(sizes.min()), "median": int(sizes.median()),
                           "max": int(sizes.max())},
        "sample_interval_h": SAMPLE_INTERVAL_H,
        "n_raman_channels": len(raman_cols),
        "raman_range_cm-1": [raman_cols[0].strip(), raman_cols[-1].strip()],
        "batch_duration_h": duration_stats,
        "offline_sampling": offline_stats,
        "timestamp_integrity": {
            "duplicate_timestamps": duplicate_ts,
            "batches_with_nonincreasing_time": nonincreasing,
            "steps_off_grid": off_grid,
            "distinct_step_sizes": dt_unique,
            "time_dtype": str(proc[time_col].dtype),
            "note": ("Time (h) is kept in float64 on purpose. In float32 the "
                     "uniform 0.2 h step spreads over several values ~1e-5 h "
                     "apart and exact comparisons silently fail."),
        },
        "regimes": regimes,
        "batch_reference_identical_to_batch_id": ref_identical,
        "fault_labels_withheld": bool(withhold_faults and fault_cols),
        "dtype_note": "floats stored as float32; float64 would store noise at double cost",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    _report(outdir, path, proc_path, raman_dir, manifest)


def _contiguous_ranges(ids: list[int]) -> str:
    """'1-30, 91-100' from a sorted list of batch numbers."""
    if not ids:
        return ""
    out, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i != prev + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in out)


def _regime_summary(proc: pd.DataFrame, roles: dict, batch_col: str) -> dict:
    """
    Report the Control_ref / PAT_ref structure as it is in the file.

    Two things are worth knowing and neither is in the dataset documentation:
    whether the regime flags are constant within a batch (they are, so they are
    batch-level design labels), and which batch ranges each combination covers.
    """
    ctrl = next((n for n in proc.columns if "Control_ref" in n), None)
    pat = next((n for n in proc.columns if n == "PAT_ref"), None)
    if ctrl is None or pat is None:
        return {"available": False,
                "reason": "Control_ref and/or PAT_ref not found after repair"}

    per_batch = proc.groupby(batch_col)[[ctrl, pat]].nunique()
    varying = per_batch[(per_batch > 1).any(axis=1)].index.tolist()

    first = proc.groupby(batch_col)[[ctrl, pat]].first()
    combos = []
    for (c, p), grp in first.groupby([ctrl, pat]):
        ids = sorted(int(i) for i in grp.index)
        combos.append({
            "control_ref": int(c),
            "pat_ref": int(p),
            "n_batches": len(ids),
            "batches": _contiguous_ranges(ids),
        })
    combos.sort(key=lambda d: d["batches"])

    return {
        "available": True,
        "constant_within_batch": not varying,
        "batches_with_varying_regime": varying,
        "combinations": combos,
        "pat_control_present": any(c["pat_ref"] == 3 for c in combos),
        "note": ("PAT_ref 3 is 'PAT control'. If pat_control_present is false, "
                 "no batch in this file is flagged as Raman-in-the-loop, "
                 "whatever the published description of batches 61-90 says. "
                 "Decide that from the feed trajectories, not from this flag."),
    }


def _write_batch(parts, bid: int, batch_col: str, raman_dir: Path):
    df = pd.concat(parts, ignore_index=True).drop(columns=[batch_col])
    df.to_parquet(raman_dir / f"batch_{bid:03d}.parquet", index=False,
                  compression="snappy")


def _report(outdir, src_path, proc_path, raman_dir, m):
    src = src_path.stat().st_size
    proc = proc_path.stat().st_size
    ram = sum(p.stat().st_size for p in raman_dir.glob("*.parquet"))
    n = len(list(raman_dir.glob("*.parquet")))

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"  {'source CSV':<22}{src / 1e9:9.2f} GB")
    print(f"  {'process_vars.parquet':<22}{proc / 1e6:9.1f} MB   <- Weeks 1-2 use only this")
    print(f"  {f'raman/ ({n} files)':<22}{ram / 1e6:9.1f} MB   <- Week 3 onward")
    print(f"  {'total':<22}{(proc + ram) / 1e9:9.2f} GB   ({(proc + ram) / src:.0%} of source)")
    print()
    print(f"  rows            {m['rows']:,}")
    print(f"  batches         {m['n_batches']}  "
          f"({m['rows_per_batch']['min']}-{m['rows_per_batch']['max']} rows each)")
    print(f"  Raman channels  {m['n_raman_channels']}  "
          f"({m['raman_range_cm-1'][0]}-{m['raman_range_cm-1'][1]} cm-1)")

    d = m.get("batch_duration_h")
    if d:
        print(f"  duration        {d['min_h']}-{d['max_h']} h "
              f"(median {d['median_h']}) — not stackable without alignment")

    ts = m["timestamp_integrity"]
    if ts["duplicate_timestamps"] or ts["batches_with_nonincreasing_time"]:
        print(f"\n  WARNING  {ts['duplicate_timestamps']} duplicate timestamps; "
              f"{len(ts['batches_with_nonincreasing_time'])} batches have "
              f"non-increasing time.")
        print("           dt = 0 gives inf/NaN in any derived rate. Handle in Week 1.")
        print(f"           affected batches: {ts['batches_with_nonincreasing_time'][:15]}")
    else:
        n_steps = len(ts.get("distinct_step_sizes") or [])
        print(f"  time grid       strictly increasing, {n_steps} "
              f"distinct step(s), {ts.get('steps_off_grid')} off-grid "
              f"[{ts.get('time_dtype')}]")
        if n_steps > 1:
            print("           NOTE  more than one step size. If Time (h) is float32, "
                  "that is quantisation,")
            print("                 not the process. Check the dtype before "
                  "drawing conclusions.")

    r = m.get("regimes", {})
    if r.get("available"):
        print("\n  Regimes (constant within batch: "
              f"{'yes' if r['constant_within_batch'] else 'NO — see manifest'}):")
        for c in r["combinations"]:
            print(f"    Control_ref {c['control_ref']}  PAT_ref {c['pat_ref']}   "
                  f"{c['n_batches']:>3} batches   {c['batches']}")
        if not r["pat_control_present"]:
            print("    No batch carries PAT_ref 3 (Raman closing a control loop).")

    if m.get("batch_reference_identical_to_batch_id"):
        print("\n  Batch reference is identical to Batch ID (redundant column).")

    if m["offline_sampling"]:
        print("\n  Offline assay sampling:")
        for col, s in m["offline_sampling"].items():
            print(f"    {col[:46]:<48} 1 label per {s['approx_rows_per_label']:>6} rows "
                  f"(~{s['approx_hours_per_label']} h)")

    if m["fault_labels_withheld"]:
        print("\n  Fault labels written separately to fault_labels.parquet (withheld).")

    print("\n" + "=" * 78)
    print(f"Output in {outdir.resolve()}")
    print("  process_vars.parquet   column_roles.csv   manifest.json   raman/")
    print("\nRead column_roles.csv before you build a feature matrix.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Convert the IndPenSim CSV to course Parquet files.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--inspect", action="store_true",
                    help="report header repair and column roles, write nothing")
    ap.add_argument("--chunksize", type=int, default=5000,
                    help="rows per chunk; lower it if memory is tight")
    ap.add_argument("--expect-batches", type=int, default=EXPECTED_BATCHES,
                    help="fail loudly if segmentation does not yield this many batches")
    ap.add_argument("--withhold-fault-labels", action="store_true",
                    help="write Fault flag / Fault_ref to a separate file (Week 6 answer key)")
    a = ap.parse_args()

    if not a.input.exists():
        sys.exit(f"ERROR: {a.input} not found")
    if a.inspect:
        inspect(a.input)
    else:
        convert(a.input, a.outdir, a.chunksize, a.withhold_fault_labels,
                a.expect_batches)


if __name__ == "__main__":
    main()
