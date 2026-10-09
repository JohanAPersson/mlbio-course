"""
bioml — the ML for Bioprocesses course package.

WHAT IS IN HERE
---------------
Plumbing: fetching and loading files, splitting data, drawing the three plots
this course uses, and putting batches of different length on a common axis.

WHAT IS DELIBERATELY NOT IN HERE
--------------------------------
Anything that makes a modelling decision for you.

    derive_rates()           Week 1 writes this. The physics is the lesson.
    feature-matrix builders  Choosing what enters the matrix IS the
                             observability-boundary lesson.
    a default for how=       A default is a recommendation, and a recommended
                             split would make the Week 2 leakage lab
                             unfalsifiable.
    preprocessing chains     Week 3 content, not infrastructure.
    multi-batch Raman concat load_raman() returns per-batch frames only.

THE SOURCE IS PUBLISHED AND YOU ARE EXPECTED TO READ IT
-------------------------------------------------------
You must be able to explain what any function you called does, why you chose
it over the alternative, and what would change if you had chosen differently.
In a notebook, `import inspect; print(inspect.getsource(bioml.split))` shows
the source of any function. "I don't know what it does" is not available.

This package is FROZEN before Week 1. If you think you have found a bug, report
it; do not patch your own copy, or your numbers become incomparable with
everyone else's.
"""

from __future__ import annotations

__version__ = "1.0.1"

from .align import align_batches
from .columns import ALIASES, FORBIDDEN_AS_FEATURES, audit, resolve, resolve_many, try_resolve
from .data import (
    data_dir,
    ensure_data,
    load_column_roles,
    load_fault_labels,
    load_manifest,
    load_process_vars,
    load_raman,
    set_data_dir,
)
from .plots import batch_trajectory_plot, parity_plot, residual_plot
from .splits import cv_score, split

__all__ = [
    "__version__",
    "set_data_dir", "data_dir", "ensure_data", "load_process_vars", "load_raman",
    "load_column_roles", "load_manifest", "load_fault_labels",
    "resolve", "resolve_many", "try_resolve", "audit", "ALIASES", "FORBIDDEN_AS_FEATURES",
    "split", "cv_score",
    "parity_plot", "residual_plot", "batch_trajectory_plot",
    "align_batches",
]


def about() -> str:
    """One-line provenance string. Printed at the top of every notebook."""
    return f"bioml {__version__} — data dir: {data_dir().resolve()}"
