"""
Chi-square and Fisher's exact test reporters.
"""

from __future__ import annotations

import math

import numpy as np

from ..core import Report
from ..formatters import fmt_stat, fmt_p_full, fmt_ci, fmt_df
from ..effect_sizes import odds_ratio
from .base import BaseReporter


class ChiSquareReporter(BaseReporter):
    """
    Report a chi-square test of independence.

    Accepts scipy ``chi2_contingency`` result (has ``.statistic``,
    ``.pvalue``, ``.dof``, ``.expected_freq``).

    kwargs
    ------
    n : int
        Total sample size (for Cramér's V).
    var_names : tuple[str, str]
        Names of the two variables (default: ("Variable A", "Variable B")).
    effectsize : bool
        Whether to include Cramér's V (default: True).
    """

    def report(self) -> Report:
        obj = self.obj
        chi2_val = float(obj.statistic)
        p_val = float(obj.pvalue)
        dof = int(obj.dof)
        expected = obj.expected_freq

        n: int = self.kwargs.get("n", int(np.round(expected.sum())))
        var_names = self.kwargs.get("var_names", ("Variable A", "Variable B"))
        include_es: bool = self.kwargs.get("effectsize", True)
        v1, v2 = var_names

        # Cramér's V
        cramers_v = v_interp = None
        if include_es and n > 0:
            k = min(expected.shape)  # min(rows, cols)
            cramers_v = math.sqrt(chi2_val / (n * (k - 1))) if k > 1 else None
            if cramers_v is not None:
                v_interp = _cramers_v_interp(cramers_v, k - 1)

        sig_word = "significant" if p_val < 0.05 else "non-significant"
        chi2_str = fmt_stat(chi2_val)
        p_str = fmt_p_full(p_val)
        df_str = fmt_df(dof)

        text = (
            f"A chi-square test of independence indicated a {sig_word} association "
            f"between {v1} and {v2}, χ²({df_str}) = {chi2_str}, {p_str}"
        )
        if cramers_v is not None:
            text += f", Cramér's V = {fmt_stat(cramers_v)}"
            if v_interp:
                text += f". The effect size is considered {v_interp}."
            else:
                text += "."
        else:
            text += "."

        statistics: dict = {
            "test": "chi-square",
            "chi2": round(chi2_val, 4),
            "df": dof,
            "p": round(p_val, 4),
            "n": n,
        }
        if cramers_v is not None:
            statistics["cramers_v"] = round(cramers_v, 4)
        if v_interp:
            statistics["effect_size_label"] = v_interp

        return Report(text, statistics, self._warnings)


class FisherReporter(BaseReporter):
    """
    Report Fisher's exact test.

    Accepts scipy ``fisher_exact`` result (has ``.statistic``, ``.pvalue``).

    kwargs
    ------
    table : array-like
        2×2 contingency table (required for odds ratio CI).
    var_names : tuple[str, str]
        Names of the two variables (default: ("Variable A", "Variable B")).
    effectsize : bool
        Whether to include odds ratio (default: True).
    ci_level : float
        Confidence level for OR CI (default: 0.95).
    """

    def report(self) -> Report:
        obj = self.obj
        or_val = float(obj.statistic)
        p_val = float(obj.pvalue)

        var_names = self.kwargs.get("var_names", ("Variable A", "Variable B"))
        table = self.kwargs.get("table", None)
        include_es: bool = self.kwargs.get("effectsize", True)
        ci_level: float = self.kwargs.get("ci_level", 0.95)
        v1, v2 = var_names

        ci_lower = ci_upper = None
        if include_es and table is not None:
            try:
                _, (ci_lower, ci_upper) = odds_ratio(table, ci_level=ci_level)
            except Exception:
                pass

        sig_word = "significant" if p_val < 0.05 else "non-significant"
        p_str = fmt_p_full(p_val)
        or_str = fmt_stat(or_val)

        text = (
            f"Fisher's exact test indicated a {sig_word} association "
            f"between {v1} and {v2}, {p_str}"
        )
        if include_es:
            text += f", OR = {or_str}"
            if ci_lower is not None:
                text += f", {fmt_ci(ci_lower, ci_upper, level=ci_level)}"
        text += "."

        statistics: dict = {
            "test": "Fisher's exact",
            "odds_ratio": round(or_val, 4),
            "p": round(p_val, 4),
        }
        if ci_lower is not None:
            statistics["ci_lower"] = round(ci_lower, 4)
            statistics["ci_upper"] = round(ci_upper, 4)

        return Report(text, statistics, self._warnings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cramers_v_interp(v: float, min_dim: int) -> str:
    """Cramér's V interpretation (Cohen, 1988 thresholds for df=1)."""
    # Thresholds vary with min_dim; use df=1 as default
    if min_dim == 1:
        if v >= 0.5:
            return "large"
        if v >= 0.3:
            return "medium"
        return "small"
    if min_dim == 2:
        if v >= 0.35:
            return "large"
        if v >= 0.21:
            return "medium"
        return "small"
    # df >= 3
    if v >= 0.29:
        return "large"
    if v >= 0.17:
        return "medium"
    return "small"
