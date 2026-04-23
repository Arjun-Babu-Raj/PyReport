"""
T-test reporter — handles scipy TtestResult and pingouin output.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core import Report
from ..formatters import fmt_stat, fmt_p_full, fmt_ci, fmt_df
from ..effect_sizes import cohens_d, cohens_d_from_t
from .base import BaseReporter


class TTestReporter(BaseReporter):
    """
    Report a t-test result.

    Accepts:
    - scipy ``TtestResult`` (has ``.statistic``, ``.pvalue``, ``.df``)
    - pingouin t-test DataFrame (columns: T, dof, p-val, cohen-d, CI95%, …)

    kwargs
    ------
    group_names : tuple[str, str]
        Names of the two groups (default: ("Group A", "Group B")).
    group_data : tuple[array-like, array-like]
        Raw data for each group; used to compute M/SD and Cohen's d.
    paired : bool
        Whether this is a paired test (default: False).
    effectsize : bool
        Whether to include effect size (default: True).
    ci_level : float
        Confidence level (default: 0.95).
    """

    def report(self) -> Report:
        obj = self.obj
        group_names: Tuple[str, str] = self.kwargs.get("group_names", ("Group A", "Group B"))
        group_data = self.kwargs.get("group_data", None)
        paired: bool = self.kwargs.get("paired", False)
        include_es: bool = self.kwargs.get("effectsize", True)
        ci_level: float = self.kwargs.get("ci_level", 0.95)

        # ----------------------------------------------------------------
        # Extract statistics from the result object
        # ----------------------------------------------------------------
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            # Pingouin output
            t_val = float(obj["T"].iloc[0])
            df_val = float(obj["dof"].iloc[0])
            p_val = float(obj["p-val"].iloc[0])
            d_val = float(obj["cohen-d"].iloc[0]) if "cohen-d" in obj.columns else None
            ci_raw = obj["CI95%"].iloc[0] if "CI95%" in obj.columns else None
            if ci_raw is not None and len(ci_raw) == 2:
                ci_lower, ci_upper = float(ci_raw[0]), float(ci_raw[1])
            else:
                ci_lower = ci_upper = None
            alt = obj.get("alternative", pd.Series(["two-sided"])).iloc[0] if "alternative" in obj.columns else "two-sided"
        else:
            # scipy result
            t_val = float(obj.statistic)
            df_val = float(obj.df)
            p_val = float(obj.pvalue)
            d_val = None
            ci_lower = ci_upper = None
            # scipy >= 1.9 has .confidence_interval()
            if hasattr(obj, "confidence_interval"):
                try:
                    ci_obj = obj.confidence_interval(confidence_level=ci_level)
                    ci_lower = float(ci_obj.low)
                    ci_upper = float(ci_obj.high)
                except Exception:
                    pass
            alt = "two-sided"

        # ----------------------------------------------------------------
        # Compute Cohen's d
        # ----------------------------------------------------------------
        d_interp = None
        if include_es:
            if group_data is not None:
                x, y = group_data
                d_val, d_interp = cohens_d(x, y, paired=paired)
            elif d_val is None:
                n_approx = int(df_val) + (1 if not paired else 2)
                d_val, d_interp = cohens_d_from_t(t_val, n_approx)
            else:
                from ..effect_sizes import _interpret, _D_THRESHOLDS
                d_interp = _interpret(abs(d_val), _D_THRESHOLDS)

        # ----------------------------------------------------------------
        # Group descriptives (if raw data available)
        # ----------------------------------------------------------------
        g1_name, g2_name = group_names
        m1 = sd1 = m2 = sd2 = None
        if group_data is not None:
            x, y = np.asarray(group_data[0], float), np.asarray(group_data[1], float)
            m1, sd1 = x.mean(), x.std(ddof=1)
            m2, sd2 = y.mean(), y.std(ddof=1)
            n1, n2 = len(x), len(y)
            if n1 < 30 or n2 < 30:
                self._warn(
                    "Sample size < 30; normality assumption may be violated. "
                    "Consider a non-parametric alternative (Mann-Whitney U)."
                )

        # ----------------------------------------------------------------
        # Build text
        # ----------------------------------------------------------------
        test_name = "paired-samples t-test" if paired else "Welch two-sample t-test"
        sig_word = "significant" if p_val < 0.05 else "non-significant"

        # Group descriptives clause
        desc_clause = ""
        if m1 is not None:
            desc_clause = (
                f" between {g1_name} (M = {fmt_stat(m1)}, SD = {fmt_stat(sd1)}) "
                f"and {g2_name} (M = {fmt_stat(m2)}, SD = {fmt_stat(sd2)})"
            )

        t_str = fmt_stat(t_val)
        df_str = fmt_df(df_val)
        p_str = fmt_p_full(p_val)

        text = (
            f"A {test_name} indicated a {sig_word} difference{desc_clause}, "
            f"t({df_str}) = {t_str}, {p_str}"
        )

        if include_es and d_val is not None:
            text += f", d = {fmt_stat(d_val)}"

        if ci_lower is not None and ci_upper is not None:
            text += f", {fmt_ci(ci_lower, ci_upper, level=ci_level)}"

        if include_es and d_interp:
            text += f". The effect size is considered {d_interp}."
        else:
            text += "."

        # ----------------------------------------------------------------
        # Statistics dict
        # ----------------------------------------------------------------
        statistics: dict = {
            "test": test_name,
            "t": round(t_val, 4),
            "df": round(df_val, 4),
            "p": round(p_val, 4),
        }
        if d_val is not None:
            statistics["cohens_d"] = round(d_val, 4)
        if d_interp:
            statistics["effect_size_label"] = d_interp
        if ci_lower is not None:
            statistics["ci_lower"] = round(ci_lower, 4)
            statistics["ci_upper"] = round(ci_upper, 4)
        if m1 is not None:
            statistics.update({
                f"{g1_name}_mean": round(m1, 4),
                f"{g1_name}_sd": round(sd1, 4),
                f"{g2_name}_mean": round(m2, 4),
                f"{g2_name}_sd": round(sd2, 4),
            })

        return Report(text, statistics, self._warnings)
