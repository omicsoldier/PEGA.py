"""
pega.cutoffs
============
Per-column cutoffs, the PASS/FAIL verdict they add up to, and the column
grouping of the results table.

A candidate is worth carrying forward when it looks active *and* clears
every liability it was measured on. Those are two different kinds of
test, so the verdict combines them that way:

* **Activity** — at least one activity or ensemble score is
  ``>= 0.5``. Any single predictor or ensemble calling it active is
  enough; they disagree often and routinely, and requiring consensus
  would reject almost everything.
* **Liabilities** — *every* liability score is strictly below its own
  ceiling (:data:`LIABILITY_GROUPS`). These are veto tests: one high
  haemolysis score is a reason to drop a peptide however active it is.

Absent columns and absent values mean different things
------------------------------------------------------
A **column that is not in the table** — the predictor was not installed,
or was left out with ``--exclude`` — is a test that was never run, so it
is skipped. A run without HemoPI2 does not fail every peptide for
haemolysis.

A **missing value in a column that is present** is a different matter:
that predictor ran and returned nothing for this sequence. There is no
evidence it clears the cutoff, so it does not. Such a row gets ``FAIL``.
This is rare — PEGA validates the FASTA before scoring — and when it
happens it marks a sequence that something went wrong on.

The degenerate case, a table with no activity *and* no liability columns
at all, leaves nothing to judge and every row passes.

Cutoffs
-------
Each ceiling is the tool's own published cutoff where it has one
(ToxinPred3's 0.38, AlgPred2's 0.3) and 0.5 otherwise. They are
deliberately in one place: change a number here and the verdict, the
documentation and every output follow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Name of the column :func:`compute_verdict` fills in.
VERDICT_COLUMN = "verdict"

PASS = "PASS"
FAIL = "FAIL"

#: An activity or ensemble score at or above this counts as active.
ACTIVITY_CUTOFF = 0.5

#: Liability columns by endpoint, in the order they appear in the output,
#: each mapped to the ceiling it must stay *strictly below* to pass.
#: Higher is worse for all of them. Add new ones to the group they
#: belong to.
LIABILITY_GROUPS: dict[str, dict[str, float]] = {
    "hemolysis": {
        "hemolysis_macrel_score": 0.5,
        "hemopi2_score": 0.45,
        "hemopi2_esm_score": 0.55,
    },
    "toxicity": {
        "toxinpred3_score": 0.38,
        "toxipep_score": 0.5,
    },
    "allergenicity": {
        "algpred2_score": 0.3,
        "algpred3_score": 0.3,
    },
}

#: Every liability column, flattened, in output order.
LIABILITY_COLUMNS: tuple[str, ...] = tuple(
    col for cols in LIABILITY_GROUPS.values() for col in cols
)

#: Column -> ceiling, flattened.
LIABILITY_CUTOFFS: dict[str, float] = {
    col: ceiling
    for cols in LIABILITY_GROUPS.values()
    for col, ceiling in cols.items()
}

_ID_COLUMNS = ("seq_name", "pep_id", "original_header", "seq_aa")


# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------


def activity_columns(df: pd.DataFrame) -> list[str]:
    """Return the individual activity score columns, in table order.

    Everything that scores a peptide's activity: a ``*_score`` column
    that is neither an ensemble nor a liability.
    """
    return [
        c for c in df.columns
        if c.endswith("_score")
        and not c.startswith("ensemble_")
        and c not in LIABILITY_CUTOFFS
    ]


def ensemble_columns(df: pd.DataFrame) -> list[str]:
    """Return the ``ensemble_*`` columns, in table order."""
    return [c for c in df.columns if c.startswith("ensemble_")]


def liability_columns(df: pd.DataFrame, group: str | None = None) -> list[str]:
    """Return the liability columns present, in output order.

    Parameters
    ----------
    group : str or None
        Restrict to one endpoint — a key of :data:`LIABILITY_GROUPS`.
        ``None`` returns all of them.
    """
    wanted = LIABILITY_GROUPS[group] if group else LIABILITY_CUTOFFS
    return [c for c in wanted if c in df.columns]


def order_columns(df: pd.DataFrame) -> list[str]:
    """Return ``df``'s columns grouped for output.

    Identifiers, the verdict, activity scores, ensembles, then the
    liability groups: haemolysis, toxicity, allergenicity. Anything that
    fits no group keeps its relative order and goes last, so this never
    drops a column.
    """
    ordered: list[str] = [c for c in _ID_COLUMNS if c in df.columns]
    if VERDICT_COLUMN in df.columns:
        ordered.append(VERDICT_COLUMN)
    ordered += activity_columns(df)
    ordered += ensemble_columns(df)
    for group in LIABILITY_GROUPS:
        ordered += liability_columns(df, group)

    placed = set(ordered)
    return ordered + [c for c in df.columns if c not in placed]


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def compute_verdict(df: pd.DataFrame) -> pd.Series:
    """Return a ``PASS``/``FAIL`` Series over ``df``'s rows.

    See the module docstring for what each verdict means and how absent
    columns and absent values are treated.

    Returns
    -------
    pandas.Series
        Of dtype ``object``, indexed like ``df``.
    """
    passing = pd.Series(True, index=df.index)

    active_cols = activity_columns(df) + ensemble_columns(df)
    if active_cols:
        scores = df[active_cols].apply(pd.to_numeric, errors="coerce")
        # NaN >= cutoff is False, so a column that failed on this row
        # simply does not vouch for it; another column still can.
        passing &= (scores >= ACTIVITY_CUTOFF).any(axis=1)

    for col in liability_columns(df):
        values = pd.to_numeric(df[col], errors="coerce")
        # NaN < ceiling is False, which is the intended answer: the
        # predictor ran and said nothing about this row.
        passing &= values < LIABILITY_CUTOFFS[col]

    return pd.Series(np.where(passing, PASS, FAIL), index=df.index)


def add_verdict(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the verdict column added and columns grouped."""
    df = df.copy()
    df[VERDICT_COLUMN] = compute_verdict(df)
    return df[order_columns(df)]


def describe_cutoffs() -> str:
    """Return a human-readable summary of the rules, one line each."""
    lines = [f"any activity or ensemble score >= {ACTIVITY_CUTOFF}"]
    for group, cols in LIABILITY_GROUPS.items():
        for col, ceiling in cols.items():
            lines.append(f"{col} < {ceiling}  ({group})")
    return "\n".join(lines)
