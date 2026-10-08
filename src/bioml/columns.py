"""
bioml.columns — bind short, stable keys to IndPenSim's actual column names.

WHY THIS EXISTS
---------------
IndPenSim's column names are simulator variable names with units glued on, e.g.

    Offline Biomass concentratio(X_offline:X(g L^{-1}))
         ^ the typo is in the source file, not here

Hard-coding strings like that into eight weeks of teaching material means one
upstream rename breaks every notebook in the course at once, and fills every
notebook with 60-character string literals that obscure what the line does.
So every column is referred to by a short key ("x_offline"), and this module
maps keys to whatever the file actually calls them.

This module encodes NO modelling judgement. It is pure name-binding, which is
why it lives inside `bioml` rather than in a notebook (Decision 9: plumbing in,
judgement out).

FAILURE BEHAVIOUR
-----------------
`resolve()` raises, with candidate columns printed. It never guesses and never
returns None silently — a silently missing column becomes a silently wrong
model three cells later, which is the failure class Decision 8 exists to stop.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# The alias table.
#
# Each key maps to a list of regular expressions, tried in order. First pattern
# that matches exactly one column wins. Matching is case-insensitive and uses
# re.search (substring semantics), because unit suffixes vary between releases.
#
# VERIFY THIS TABLE against the real column_roles.csv before Week 1:
# `bioml.audit(df)` does exactly that in one call.
# ---------------------------------------------------------------------------
ALIASES: dict[str, list[str]] = {
    # --- identifiers -------------------------------------------------------
    "time":            [r"^Time\s*\(h\)$", r"^time"],
    "batch":           [r"^Batch ID$", r"^Batch_ID$"],
    "batch_ref":       [r"Batch_ref"],

    # --- batch-level design labels (constant within a batch) ---------------
    "control_ref":     [r"Control_ref"],
    "pat_ref":         [r"^PAT_ref$", r"PAT_ref"],
    "fault_flag":      [r"^Fault flag$", r"fault\s*flag"],
    "fault_ref":       [r"Fault_ref"],

    # --- online process measurements (a real plant would have these) -------
    "volume":          [r"Vessel Volume", r"\(V:L\)"],
    "weight":          [r"Vessel Weight", r"\(Wt:"],
    "temperature":     [r"Temperature\(T:K\)", r"\(T:K\)"],
    "ph":              [r"^pH\(", r"\(pH:"],
    "do":              [r"Dissolved oxygen", r"\(DO2:"],
    "o2_offgas":       [r"Oxygen in percent in off-gas", r"\(O2:"],
    "co2_offgas":      [r"carbon dioxide percent in off-gas", r"CO2outgas"],
    "our":             [r"Oxygen Uptake Rate", r"\(OUR:"],
    "cer":             [r"Carbon evolution rate", r"\(CER:"],
    "heat":            [r"Generated heat", r"\(Q:"],
    "pressure":        [r"Air head pressure", r"pressure:bar"],
    "rpm":             [r"Agitator RPM", r"\(RPM:"],
    "aeration":        [r"Aeration rate", r"\(Fg:"],

    # --- manipulated inputs (feeds and actuators) --------------------------
    "feed_sugar":      [r"Sugar feed rate", r"\(Fs:"],
    "feed_paa":        [r"PAA flow", r"\(Fpaa:"],
    "feed_oil":        [r"Oil flow", r"\(Foil:"],
    "feed_water":      [r"Water for injection", r"\(Fw:"],
    "feed_acid":       [r"Acid flow rate", r"\(Fa:"],
    "feed_base":       [r"Base flow rate", r"\(Fb:"],
    "cool_water":      [r"Heating/cooling water", r"\(Fc:"],
    "heat_water":      [r"Heating water flow", r"\(Fh:"],
    "removed":         [r"Dumped broth", r"Fremoved"],
    "nh3_shots":       [r"Ammonia shots", r"NH3_shots"],

    # --- simulator state: present in the file, absent on a plant -----------
    # These keys exist so a notebook can NAME them in order to EXCLUDE them.
    # Resolving one is not permission to use it as a model input.
    "p_state":         [r"Penicillin concentration\(P:", r"\(P:g/L\)"],
    "s_state":         [r"Substrate concentration\(S:", r"\(S:g/L\)"],

    # --- offline laboratory assays (~98% missing) --------------------------
    "p_offline":       [r"P_offline"],
    "x_offline":       [r"X_offline"],
    "viscosity":       [r"Viscosity_offline", r"Viscosity"],
    "nh3_offline":     [r"NH3_offline"],
    "paa_offline":     [r"PAA_offline"],
}

# Keys naming simulator ground truth or answer keys. Resolving these is
# legitimate (you must be able to name a column to drop it); using them as
# model inputs is not.
FORBIDDEN_AS_FEATURES = ("p_state", "s_state", "fault_flag", "fault_ref")


def _columns_of(obj: pd.DataFrame | Iterable[str]) -> list[str]:
    """Accept either a DataFrame or a bare list of column names."""
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)
    return list(obj)


def resolve(obj: pd.DataFrame | Iterable[str], key: str) -> str:
    """
    Return the actual column name in `obj` for the short key `key`.

    Raises KeyError if the key is unknown or no column matches, listing
    plausible candidates so the alias table can be corrected in one edit.
    """
    if key not in ALIASES:
        raise KeyError(f"unknown column key {key!r}. Known keys: {sorted(ALIASES)}")

    columns = _columns_of(obj)
    for pattern in ALIASES[key]:
        rx = re.compile(pattern, re.IGNORECASE)
        hits = [c for c in columns if rx.search(c)]   # substring match, see above
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise KeyError(
                f"column key {key!r} pattern {pattern!r} matched {len(hits)} "
                f"columns: {hits}. Tighten the pattern in bioml.columns.ALIASES."
            )

    # Nothing matched: show what IS there, cheaply.
    token = key.split("_")[0][:4]
    near = [c for c in columns if token.lower() in c.lower()][:10]
    raise KeyError(
        f"no column matches key {key!r} (patterns {ALIASES[key]}).\n"
        f"  columns containing {token!r}: {near or '(none)'}\n"
        f"  Fix bioml.columns.ALIASES rather than renaming the data."
    )


def resolve_many(obj: pd.DataFrame | Iterable[str], keys: Iterable[str]) -> dict[str, str]:
    """Resolve several keys at once. Fails on the first key that cannot be resolved."""
    return {k: resolve(obj, k) for k in keys}


def try_resolve(obj: pd.DataFrame | Iterable[str], key: str) -> str | None:
    """Resolve, or return None. Use only where absence is genuinely acceptable."""
    try:
        return resolve(obj, key)
    except KeyError:
        return None


def audit(obj: pd.DataFrame | Iterable[str]) -> pd.DataFrame:
    """
    Report which alias keys bind against these columns and which do not.

    Run once, on the real distributed Parquet, before the course starts. Every
    unresolved key is either a column IndPenSim does not have or a pattern that
    needs fixing — and you want to know which, once, in advance.
    """
    columns = _columns_of(obj)
    rows = []
    for key in ALIASES:
        name = try_resolve(columns, key)
        rows.append({
            "key": key,
            "resolved": name is not None,
            "column": name or "",
            "forbidden_as_feature": key in FORBIDDEN_AS_FEATURES,
        })
    return pd.DataFrame(rows)
