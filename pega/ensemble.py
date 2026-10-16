"""
pega.ensemble
=============
Validated ensemble scores for antimicrobial peptide prediction.

Each ensemble was identified by stochastic search over predictor subsets,
score transformations and combination strategies on the full nr90 data.
Every activity has two ensembles, optimised for different jobs:

``*_class_score``   MCC-optimised — for labelling at a 0.5 threshold.
``*_screen_score``  BEDROC-optimised — for ranked virtual screening.

Transformations
---------------
raw         : f(x) = x
gamma0.25   : f(x) = clip(x, 0, 1) ** 0.25
gamma0.5    : f(x) = clip(x, 0, 1) ** 0.5
gamma2      : f(x) = clip(x, 0, 1) ** 2
gamma3      : f(x) = clip(x, 0, 1) ** 3
logistic5   : f(x) = 1 / (1 + exp(-5  × (x - 0.5)))
logistic10  : f(x) = 1 / (1 + exp(-10 × (x - 0.5)))

Combination strategies
----------------------
weighted_mean : Σ wᵢ × tᵢ(scoreᵢ),  Σwᵢ = 1
mean          : arithmetic mean of the transformed member scores
median        : element-wise median of the transformed member scores
vote          : fraction of members whose transformed score is ≥ 0.5

``vote`` only takes n+1 distinct values for n members, so it cannot rank
candidates finely. Use it for labels, never for screening — that is what
the BEDROC ensembles are for.

Not every ensemble beats its best single predictor out of fold. Those are
marked ``deploy = False`` with a ``deploy_note`` naming what to use
instead; :func:`compute_ensembles` still computes them on request.

Component format
----------------
Each component is a tuple ``(column, transform, weight)``. ``weight`` is
``None`` for the strategies that do not use weights (mean, median, vote).

Usage
-----
>>> from pega.ensemble import compute_ensembles
>>> df = compute_ensembles(df)

Available ensemble columns
--------------------------
ensemble_AMP_class_score  — mean,          3 predictors, MCC=0.839 (nested CV)
ensemble_AVP_class_score  — weighted_mean, 3 predictors, MCC=0.412 — do not deploy
ensemble_AFP_class_score  — vote,          4 predictors, MCC=0.700 (nested CV)
ensemble_APP_class_score  — weighted_mean, 2 predictors, MCC=0.623 (nested CV)
ensemble_ABP_class_score  — weighted_mean, 4 predictors, MCC=0.854 (nested CV)
ensemble_AMP_screen_score — weighted_mean, 4 predictors, BEDROC=0.891
ensemble_AVP_screen_score — weighted_mean, 3 predictors, BEDROC=0.412 — do not deploy
ensemble_AFP_screen_score — median,        4 predictors, BEDROC=0.809
ensemble_APP_screen_score — median,        4 predictors, BEDROC=0.621
ensemble_ABP_screen_score — weighted_mean, 5 predictors, BEDROC=0.890
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Transformation functions
# Each accepts (x: np.ndarray) and returns np.ndarray
# ---------------------------------------------------------------------------

def _raw(x):
    return x.copy()

def _gamma025(x):
    return np.clip(x, 0, 1) ** 0.25

def _gamma05(x):
    return np.clip(x, 0, 1) ** 0.5

def _gamma2(x):
    return np.clip(x, 0, 1) ** 2

def _gamma3(x):
    return np.clip(x, 0, 1) ** 3

def _logistic5(x):
    return 1.0 / (1.0 + np.exp(-5.0 * (x - 0.5)))

def _logistic10(x):
    return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))


_TRANSFORMS = {
    "raw":        _raw,
    "gamma0.25":  _gamma025,
    "gamma0.5":   _gamma05,
    "gamma2":     _gamma2,
    "gamma3":     _gamma3,
    "logistic5":  _logistic5,
    "logistic10": _logistic10,
}


# ---------------------------------------------------------------------------
# Ensemble definitions
# Each component: (column, transform, weight); weight is None when the
# strategy does not use weights.
# ---------------------------------------------------------------------------

ENSEMBLES: dict[str, dict] = {

    # ═══ MCC-optimised — classification ═════════════════════════════════════

    # ── ensemble_AMP_class_score ────────────────────────────────────────────
    # General antimicrobial — dataset_d1
    "ensemble_AMP_class_score": {
        "method": "mean",
        "objective": "classification",
        "components": [
            ("AMPnet_score",           "gamma3",    None),
            ("Macrel_score",           "logistic5", None),
            ("AMPlify_balanced_score", "logistic5", None),
        ],
        "description": "Mean, 3 predictors, general antimicrobial (d1).",
        "metrics": {"mcc_in_sample": 0.844, "mcc_nested_cv": 0.839,
                    "selection_bias": 0.005, "delta_mcc_vs_best_individual": 0.015,
                    "f1": 0.912, "auc": 0.962,
                    "sensitivity": 0.852, "specificity": 0.984},
        "deploy": True,
    },

    # ── ensemble_AVP_class_score ────────────────────────────────────────────
    # Antiviral — dataset_d2
    "ensemble_AVP_class_score": {
        "method": "weighted_mean",
        "objective": "classification",
        "components": [
            ("AMP_CG_score",           "logistic10", 0.6726940695374691),
            ("AMPlify_balanced_score", "gamma0.25",  0.21215878202884145),
            ("modlAMP_RF_score",       "gamma2",     0.11514714843368948),
        ],
        "description": "Weighted mean, 3 predictors, antiviral (d2).",
        "metrics": {"mcc_in_sample": 0.446, "mcc_nested_cv": 0.412,
                    "selection_bias": 0.034, "delta_mcc_vs_best_individual": -0.015,
                    "f1": 0.610, "auc": 0.765,
                    "sensitivity": 0.472, "specificity": 0.925},
        "deploy": False,
        "deploy_note": "Does NOT generalise out of fold (nested-CV delta MCC "
                       "= -0.015). Prefer the single predictor AMP_CG_score.",
    },

    # ── ensemble_AFP_class_score ────────────────────────────────────────────
    # Antifungal — dataset_d3
    "ensemble_AFP_class_score": {
        "method": "vote",
        "objective": "classification",
        "components": [
            ("Macrel_score",       "logistic10", None),
            ("AMP_CG_score",       "gamma2",     None),
            ("ampir_mature_score", "gamma2",     None),
            ("modlAMP_RF_score",   "gamma0.25",  None),
        ],
        "description": "Vote, 4 predictors, antifungal (d3).",
        "metrics": {"mcc_in_sample": 0.770, "mcc_nested_cv": 0.700,
                    "selection_bias": 0.070, "delta_mcc_vs_best_individual": 0.060,
                    "f1": 0.872, "auc": 0.881,
                    "sensitivity": 0.812, "specificity": 0.950},
        "deploy": True,
        "deploy_note": "Provisional (small n). Vote classifies well but "
                       "collapses early enrichment (BEDROC 0.552) — for "
                       "screening use ensemble_AFP_screen_score.",
    },

    # ── ensemble_APP_class_score ────────────────────────────────────────────
    # Antiparasitic — dataset_d4
    "ensemble_APP_class_score": {
        "method": "weighted_mean",
        "objective": "classification",
        "components": [
            ("Macrel_score", "gamma2", 0.5983914520689867),
            ("AMPnet_score", "gamma2", 0.40160854793101325),
        ],
        "description": "Weighted mean, 2 predictors, antiparasitic (d4).",
        "metrics": {"mcc_in_sample": 0.674, "mcc_nested_cv": 0.623,
                    "selection_bias": 0.051, "delta_mcc_vs_best_individual": 0.016,
                    "f1": 0.813, "auc": 0.867,
                    "sensitivity": 0.739, "specificity": 0.923},
        "deploy": True,
        "deploy_note": "Modest honest gain (+0.016); the single predictor "
                       "ampir_mature_score is also reasonable here.",
    },

    # ── ensemble_ABP_class_score ────────────────────────────────────────────
    # Antibacterial — dataset_d5
    "ensemble_ABP_class_score": {
        "method": "weighted_mean",
        "objective": "classification",
        "components": [
            ("ampir_mature_score", "logistic5", 0.1875790416192226),
            ("AMP_CG_score",       "gamma2",    0.4061519374849166),
            ("AMPnet_score",       "gamma0.5",  0.1696023310651469),
            ("modlAMP_RF_score",   "raw",       0.23666668983071384),
        ],
        "description": "Weighted mean, 4 predictors, antibacterial (d5).",
        "metrics": {"mcc_in_sample": 0.861, "mcc_nested_cv": 0.854,
                    "selection_bias": 0.007, "delta_mcc_vs_best_individual": 0.054,
                    "f1": 0.926, "auc": 0.967,
                    "sensitivity": 0.889, "specificity": 0.970},
        "deploy": True,
    },

    # ═══ BEDROC-optimised — virtual screening ═══════════════════════════════

    # ── ensemble_AMP_screen_score ───────────────────────────────────────────
    # General antimicrobial — dataset_d1
    "ensemble_AMP_screen_score": {
        "method": "weighted_mean",
        "objective": "screening",
        "components": [
            ("AMPnet_score",             "gamma0.25", 0.41056016275350893),
            ("Macrel_score",             "gamma0.25", 0.47788312685935536),
            ("AMPlify_balanced_score",   "gamma2",    0.05263419554143787),
            ("AMPlify_imbalanced_score", "gamma0.5",  0.05892251484569787),
        ],
        "description": "Weighted mean, 4 predictors, general antimicrobial (d1).",
        "metrics": {"bedroc": 0.891, "auprc": 0.844},
        "deploy": True,
    },

    # ── ensemble_AVP_screen_score ───────────────────────────────────────────
    # Antiviral — dataset_d2
    "ensemble_AVP_screen_score": {
        "method": "weighted_mean",
        "objective": "screening",
        "components": [
            ("AMP_CG_score",       "gamma2", 0.8053897789407212),
            ("ampir_mature_score", "gamma2", 0.10206818446754327),
            ("modlAMP_SVM_score",  "gamma2", 0.09254203659173553),
        ],
        "description": "Weighted mean, 3 predictors, antiviral (d2).",
        "metrics": {"bedroc": 0.412, "auprc": 0.271},
        "deploy": False,
        "deploy_note": "Only ties the single AMP_CG_score predictor (0.406); "
                       "a single predictor is preferable.",
    },

    # ── ensemble_AFP_screen_score ───────────────────────────────────────────
    # Antifungal — dataset_d3
    "ensemble_AFP_screen_score": {
        "method": "median",
        "objective": "screening",
        "components": [
            ("Macrel_score",       "raw",       None),
            ("AMP_CG_score",       "gamma0.25", None),
            ("AMPnet_score",       "gamma0.25", None),
            ("ampir_mature_score", "gamma2",    None),
        ],
        "description": "Median, 4 predictors, antifungal (d3).",
        "metrics": {"bedroc": 0.809, "auprc": 0.751},
        "deploy": True,
    },

    # ── ensemble_APP_screen_score ───────────────────────────────────────────
    # Antiparasitic — dataset_d4
    "ensemble_APP_screen_score": {
        "method": "median",
        "objective": "screening",
        "components": [
            ("Macrel_score",       "gamma0.5",   None),
            ("AMPnet_score",       "logistic10", None),
            ("modlAMP_SVM_score",  "gamma3",     None),
            ("ampir_mature_score", "logistic10", None),
        ],
        "description": "Median, 4 predictors, antiparasitic (d4).",
        "metrics": {"bedroc": 0.621, "auprc": 0.533},
        "deploy": True,
    },

    # ── ensemble_ABP_screen_score ───────────────────────────────────────────
    # Antibacterial — dataset_d5
    "ensemble_ABP_screen_score": {
        "method": "weighted_mean",
        "objective": "screening",
        "components": [
            ("AMP_CG_score",           "gamma2",   0.3108101545898291),
            ("ampir_mature_score",     "gamma0.5", 0.2580969531528011),
            ("AMPlify_balanced_score", "gamma3",   0.07571342346357708),
            ("AMPnet_score",           "raw",      0.1212471126950315),
            ("Macrel_score",           "raw",      0.23413235609876137),
        ],
        "description": "Weighted mean, 5 predictors, antibacterial (d5).",
        "metrics": {"bedroc": 0.890, "auprc": 0.836},
        "deploy": True,
    },
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    """Apply a named transformation."""
    fn = _TRANSFORMS.get(transform)
    if fn is None:
        raise ValueError(
            f"Unknown transformation '{transform}'. "
            f"Available: {list(_TRANSFORMS)}"
        )
    return fn(values)


def compute_ensembles(
    df: pd.DataFrame,
    ensemble_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute validated ensemble scores and append them to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`pega.utils.calculate_scores`.
    ensemble_names : list[str] or None
        Ensembles to compute. ``None`` computes all defined ensembles.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with one new column per ensemble.
    """
    import warnings
    names = ensemble_names or list(ENSEMBLES)

    for name in names:
        if name not in ENSEMBLES:
            raise KeyError(
                f"Ensemble '{name}' is not defined. "
                f"Available: {list(ENSEMBLES)}"
            )

        cfg        = ENSEMBLES[name]
        components = cfg["components"]
        method     = cfg["method"]

        missing = [col for col, _, _ in components if col not in df.columns]
        if missing:
            warnings.warn(
                f"Ensemble '{name}' skipped — missing columns: {missing}",
                stacklevel=2,
            )
            continue

        transformed = np.vstack([
            _apply_transform(df[col].fillna(0).values, transform)
            for col, transform, _ in components
        ])

        if method == "weighted_mean":
            weights = np.array([w for _, _, w in components], dtype=float)
            weights = weights / weights.sum()   # re-normalise (guard rounding)
            df[name] = (weights[:, None] * transformed).sum(axis=0)
        elif method == "mean":
            df[name] = transformed.mean(axis=0)
        elif method == "median":
            df[name] = np.median(transformed, axis=0)
        elif method == "vote":
            df[name] = (transformed >= 0.5).mean(axis=0)
        else:
            raise ValueError(f"Unknown ensemble method '{method}'.")

    return df


def required_predictors(ensemble_names: list[str]) -> set[str]:
    """Return the predictor score-column names needed for the given ensembles.

    Used to auto-restrict which predictors are run when the caller asks for
    specific ensembles only (e.g. ``PEGA score --ensembles AMP_class``). Names
    not found in :data:`ENSEMBLES` are ignored here — callers that need to warn
    about unknown ensemble names already validate them separately.

    Parameters
    ----------
    ensemble_names : list[str]
        Ensemble names, e.g. ``["ensemble_AMP_class_score"]``.

    Returns
    -------
    set[str]
        Score-column names (e.g. ``{"AMP_CG_score", "ampir_mature_score"}``)
        required by at least one of the given ensembles.
    """
    needed: set[str] = set()
    for name in ensemble_names:
        cfg = ENSEMBLES.get(name)
        if cfg is None:
            continue
        needed.update(col for col, _, _ in cfg["components"])
    return needed


def list_ensembles() -> list[dict]:
    """Return metadata for all defined ensembles."""
    return [
        {
            "name":         name,
            "method":       cfg["method"],
            "objective":    cfg["objective"],
            "n_predictors": len(cfg["components"]),
            "description":  cfg["description"],
            "metrics":      cfg["metrics"],
            "deploy":       cfg["deploy"],
            "deploy_note":  cfg.get("deploy_note"),
        }
        for name, cfg in ENSEMBLES.items()
    ]
