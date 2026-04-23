"""
Effect size calculations.

Implements Cohen (1988) effect sizes plus related measures with
interpretation labels using standard thresholds (overridable).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Default interpretation thresholds (Cohen, 1988)
# ---------------------------------------------------------------------------

_D_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}
_R_THRESHOLDS = {"small": 0.1, "medium": 0.3, "large": 0.5}
_ETA_THRESHOLDS = {"small": 0.01, "medium": 0.06, "large": 0.14}
_OMEGA_THRESHOLDS = {"small": 0.01, "medium": 0.06, "large": 0.14}


def _interpret(abs_val: float, thresholds: dict) -> str:
    """Return 'small', 'medium', or 'large' label."""
    if abs_val >= thresholds["large"]:
        return "large"
    if abs_val >= thresholds["medium"]:
        return "medium"
    return "small"


def cohens_d(
    x: "array-like",
    y: "array-like",
    paired: bool = False,
    thresholds: Optional[dict] = None,
) -> Tuple[float, str]:
    """
    Compute Cohen's d for two independent (or paired) samples.

    Parameters
    ----------
    x, y:
        Sample arrays.
    paired:
        If True, compute paired Cohen's d (mean difference / SD of differences).
    thresholds:
        Override default interpretation thresholds.

    Returns
    -------
    (d, interpretation)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = thresholds or _D_THRESHOLDS

    if paired:
        diff = x - y
        sd_diff = diff.std(ddof=1)
        d = diff.mean() / sd_diff if sd_diff != 0 else 0.0
    else:
        n1, n2 = len(x), len(y)
        pooled_sd = math.sqrt(
            ((n1 - 1) * x.std(ddof=1) ** 2 + (n2 - 1) * y.std(ddof=1) ** 2)
            / (n1 + n2 - 2)
        )
        if pooled_sd == 0:
            d = 0.0
        else:
            d = (x.mean() - y.mean()) / pooled_sd

    return d, _interpret(abs(d), t)


def cohens_d_from_t(
    t: float,
    n1: int,
    n2: Optional[int] = None,
    thresholds: Optional[dict] = None,
) -> Tuple[float, str]:
    """
    Estimate Cohen's d from a t-statistic when raw data are unavailable.

    For independent samples (n2 provided):
        d = t * sqrt(1/n1 + 1/n2)
    For one-sample or paired (n2=None):
        d = t / sqrt(n1)
    """
    thr = thresholds or _D_THRESHOLDS
    if n2 is None:
        d = t / math.sqrt(n1)
    else:
        d = t * math.sqrt((n1 + n2) / (n1 * n2))
    return d, _interpret(abs(d), thr)


def pearson_r_from_t(t: float, df: float, thresholds: Optional[dict] = None) -> Tuple[float, str]:
    """
    Compute Pearson r from a t-statistic and degrees of freedom.

    r = t / sqrt(t² + df)
    """
    thr = thresholds or _R_THRESHOLDS
    r = t / math.sqrt(t ** 2 + df)
    return r, _interpret(abs(r), thr)


def eta_squared(
    ss_effect: float,
    ss_total: float,
    thresholds: Optional[dict] = None,
) -> Tuple[float, str]:
    """
    Compute η² (eta squared).

    η² = SS_effect / SS_total
    """
    thr = thresholds or _ETA_THRESHOLDS
    if ss_total == 0:
        return 0.0, "small"
    eta2 = ss_effect / ss_total
    return eta2, _interpret(eta2, thr)


def omega_squared(
    F: float,
    df_effect: int,
    df_error: int,
    n: int,
    thresholds: Optional[dict] = None,
) -> Tuple[float, str]:
    """
    Compute ω² (omega squared) from an F-ratio.

    ω² = (df_effect * (F - 1)) / (df_effect * (F - 1) + n)
    """
    thr = thresholds or _OMEGA_THRESHOLDS
    numerator = df_effect * (F - 1)
    denominator = df_effect * (F - 1) + n
    if denominator <= 0:
        omega2 = 0.0
    else:
        omega2 = max(0.0, numerator / denominator)
    return omega2, _interpret(omega2, thr)


def odds_ratio(
    table: "array-like",
    ci_level: float = 0.95,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute the odds ratio and CI from a 2×2 contingency table.

    Uses the log method for the CI.

    Parameters
    ----------
    table:
        2×2 array [[a, b], [c, d]].
    ci_level:
        Confidence level.

    Returns
    -------
    (OR, (lower, upper))
    """
    table = np.asarray(table, dtype=float)
    a, b = table[0, 0], table[0, 1]
    c, d = table[1, 0], table[1, 1]

    # Haldane-Anscombe correction if any cell is 0
    if 0 in [a, b, c, d]:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)
    log_or = math.log(or_val)
    se_log = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)

    z = _z_critical(ci_level)
    lower = math.exp(log_or - z * se_log)
    upper = math.exp(log_or + z * se_log)

    return or_val, (lower, upper)


def rank_biserial(U: float, n1: int, n2: int) -> Tuple[float, str]:
    """
    Compute rank-biserial correlation r from Mann-Whitney U.

    r = 1 - (2U) / (n1 * n2)
    """
    r = 1 - (2 * U) / (n1 * n2)
    return r, _interpret(abs(r), _R_THRESHOLDS)


def _z_critical(level: float) -> float:
    """Return z critical value for a two-tailed CI."""
    from scipy import stats  # lazy import

    return stats.norm.ppf(1 - (1 - level) / 2)
