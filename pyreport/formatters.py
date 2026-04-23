"""
APA 7 statistical formatters.

All numeric formatting follows APA 7th edition guidelines:
- No leading zero before decimal point for values bounded between -1 and 1
  (p-values, correlations, effect sizes r/d when < 1)
- Exact p-values unless p < .001
- Round to specified decimal places
- Guard against negative zero
"""

from __future__ import annotations


def _remove_neg_zero(val: float, decimals: int = 2) -> float:
    """Return 0.0 instead of -0.0."""
    rounded = round(val, decimals)
    if rounded == 0.0:
        return 0.0
    return rounded


def fmt_stat(val: float, decimals: int = 2) -> str:
    """
    Format a general statistic to *decimals* decimal places.

    Applies APA leading-zero rule: values whose absolute value is strictly
    less than 1 (and are bounded, e.g. correlations, betas) should *not*
    carry a leading zero.  For statistics that can exceed 1 (F, t, χ²) the
    leading zero is retained because it is informative.

    Parameters
    ----------
    val:
        Numeric value to format.
    decimals:
        Number of decimal places.

    Returns
    -------
    str
        Formatted string.
    """
    val = _remove_neg_zero(val, decimals)
    formatted = f"{val:.{decimals}f}"
    return formatted


def fmt_p(p: float) -> str:
    """
    Format a p-value per APA 7 conventions.

    - p < .001  → "< .001"
    - otherwise → exact value with 3 dp, no leading zero

    Parameters
    ----------
    p:
        p-value (0 ≤ p ≤ 1).

    Returns
    -------
    str
        e.g. "< .001" or ".043" or "1.000"
    """
    if p < 0.001:
        return "< .001"
    # Round to 3 dp
    p_rounded = _remove_neg_zero(p, 3)
    raw = f"{p_rounded:.3f}"
    # Remove leading zero: "0.043" → ".043"
    if raw.startswith("0."):
        return raw[1:]
    if raw.startswith("-0."):
        return "-" + raw[2:]
    return raw


def fmt_p_full(p: float) -> str:
    """
    Return the full APA p-value expression suitable for embedding in text.

    Examples: "p < .001", "p = .043", "p = 1.000"
    """
    p_str = fmt_p(p)
    if p_str.startswith("<"):
        return f"p {p_str}"
    return f"p = {p_str}"


def fmt_bound(val: float, decimals: int = 2) -> str:
    """
    Format a value that is bounded between −1 and 1 (no leading zero).

    Parameters
    ----------
    val:
        Value to format.
    decimals:
        Decimal places.

    Returns
    -------
    str
        e.g. ".75" or "-.32"
    """
    val = _remove_neg_zero(val, decimals)
    raw = f"{val:.{decimals}f}"
    if raw.startswith("0."):
        return raw[1:]
    if raw.startswith("-0."):
        return "-" + raw[2:]
    return raw


def fmt_ci(lower: float, upper: float, decimals: int = 2, level: float = 0.95) -> str:
    """
    Format a confidence interval per APA 7.

    Parameters
    ----------
    lower:
        Lower bound.
    upper:
        Upper bound.
    decimals:
        Decimal places for bounds.
    level:
        Confidence level (default 0.95).

    Returns
    -------
    str
        e.g. "95% CI [0.21, 1.28]"
    """
    pct = int(round(level * 100))
    lo = fmt_stat(lower, decimals)
    hi = fmt_stat(upper, decimals)
    return f"{pct}% CI [{lo}, {hi}]"


def fmt_df(df: float) -> str:
    """
    Format degrees of freedom.

    Integer df → no decimal places; Welch df (float) → 2 dp.
    NaN → "?".
    """
    import math
    if df is None or (isinstance(df, float) and math.isnan(df)):
        return "?"
    if df == int(df):
        return str(int(df))
    return fmt_stat(df, 2)
