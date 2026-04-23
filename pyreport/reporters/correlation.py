"""
Correlation reporter — handles scipy pearsonr/spearmanr/kendalltau
and pingouin correlation output.
"""

from __future__ import annotations

import math
from typing import Optional

from ..core import Report
from ..formatters import fmt_stat, fmt_bound, fmt_p_full, fmt_ci, fmt_df
from ..effect_sizes import _interpret, _R_THRESHOLDS
from .base import BaseReporter


class CorrelationReporter(BaseReporter):
    """
    Report a correlation result.

    Accepts:
    - scipy ``PearsonRResult``, ``SpearmanrResult``, or ``KendalltauResult``
    - pingouin correlation DataFrame (columns: r, p-val, CI95%, …)

    kwargs
    ------
    var_names : tuple[str, str]
        Names of the two variables (default: ("X", "Y")).
    method : str
        Correlation method label override (default: auto-detected).
    ci_level : float
        Confidence level (default: 0.95).
    effectsize : bool
        Whether to include effect size label (default: True).
    n : int
        Sample size (used to compute CI when not available directly).
    """

    def report(self) -> Report:
        obj = self.obj
        var_names = self.kwargs.get("var_names", ("X", "Y"))
        ci_level: float = self.kwargs.get("ci_level", 0.95)
        include_es: bool = self.kwargs.get("effectsize", True)
        n: Optional[int] = self.kwargs.get("n", None)

        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            # Pingouin
            r_val = float(obj["r"].iloc[0])
            p_val = float(obj["p-val"].iloc[0])
            ci_raw = obj["CI95%"].iloc[0] if "CI95%" in obj.columns else None
            if ci_raw is not None and len(ci_raw) == 2:
                ci_lower, ci_upper = float(ci_raw[0]), float(ci_raw[1])
            else:
                ci_lower = ci_upper = None
            method = self.kwargs.get("method", "Pearson")
            if "n" in obj.columns:
                n = int(obj["n"].iloc[0])
        else:
            r_val = float(obj.statistic)
            p_val = float(obj.pvalue)
            ci_lower = ci_upper = None

            # Try to get CI from the result object (scipy >= 1.9)
            if hasattr(obj, "confidence_interval"):
                try:
                    ci_obj = obj.confidence_interval(confidence_level=ci_level)
                    ci_lower = float(ci_obj.low)
                    ci_upper = float(ci_obj.high)
                except Exception:
                    pass

            # Auto-detect method
            cls = type(obj).__name__.lower()
            if "pearson" in cls:
                method = "Pearson"
            elif "spearman" in cls:
                method = "Spearman"
            elif "kendall" in cls:
                method = "Kendall's tau"
            else:
                method = self.kwargs.get("method", "Pearson")

        # Fallback CI via Fisher z-transform (Pearson only)
        if ci_lower is None and n is not None and method == "Pearson":
            ci_lower, ci_upper = _pearson_ci(r_val, n, ci_level)

        # Interpretation
        r_interp = _interpret(abs(r_val), _R_THRESHOLDS) if include_es else None

        sig_word = "significant" if p_val < 0.05 else "non-significant"
        direction = "positive" if r_val >= 0 else "negative"
        v1, v2 = var_names

        # Degrees of freedom for Pearson r
        df_val = (n - 2) if n is not None else None

        r_str = fmt_bound(r_val)
        p_str = fmt_p_full(p_val)

        if df_val is not None:
            r_clause = f"r({fmt_df(df_val)}) = {r_str}"
        else:
            r_clause = f"r = {r_str}"

        text = (
            f"A {method} correlation indicated a {sig_word} {direction} association "
            f"between {v1} and {v2}, {r_clause}, {p_str}"
        )

        if ci_lower is not None:
            text += f", {fmt_ci(ci_lower, ci_upper, level=ci_level)}"

        if r_interp:
            text += f". The effect size is considered {r_interp}."
        else:
            text += "."

        statistics: dict = {
            "method": method,
            "r": round(r_val, 4),
            "p": round(p_val, 4),
        }
        if df_val is not None:
            statistics["df"] = df_val
        if ci_lower is not None:
            statistics["ci_lower"] = round(ci_lower, 4)
            statistics["ci_upper"] = round(ci_upper, 4)
        if r_interp:
            statistics["effect_size_label"] = r_interp

        return Report(text, statistics, self._warnings)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _pearson_ci(r: float, n: int, level: float = 0.95):
    """Fisher z-transform confidence interval for Pearson r."""
    from scipy import stats

    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    crit = stats.norm.ppf(1 - (1 - level) / 2)
    lower = math.tanh(z - crit * se)
    upper = math.tanh(z + crit * se)
    return lower, upper
