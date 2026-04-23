"""
ANOVA reporters — pingouin ANOVA DataFrames.
"""

from __future__ import annotations

from ..core import Report
from ..formatters import fmt_stat, fmt_p_full, fmt_df
from .base import BaseReporter


class PingouinAnovaReporter(BaseReporter):
    """
    Report a pingouin ANOVA result DataFrame.

    The DataFrame has columns: Source, ddof1, ddof2, F, p-unc,
    and optionally np2 (partial η²), ng2, or eps.

    kwargs
    ------
    effectsize : bool
        Whether to include partial η² (default: True).
    """

    def report(self) -> Report:
        df = self.obj
        include_es: bool = self.kwargs.get("effectsize", True)

        lines = []
        statistics: dict = {"test": "ANOVA", "terms": []}

        for _, row in df.iterrows():
            source = row.get("Source", "?")
            F_val = row["F"]
            p_val = row["p-unc"]

            # Skip rows without valid F/p (e.g. Error rows in some pingouin outputs)
            import math as _math
            try:
                if _math.isnan(float(F_val)) or _math.isnan(float(p_val)):
                    continue
            except (TypeError, ValueError):
                continue

            F_val = float(F_val)
            p_val = float(p_val)
            df1 = float(row["ddof1"])
            df2 = float(row["ddof2"])

            eta2 = None
            if include_es and "np2" in df.columns:
                eta2 = float(row["np2"])

            F_str = fmt_stat(F_val)
            df1_str = fmt_df(df1)
            df2_str = fmt_df(df2)
            p_str = fmt_p_full(p_val)

            sig = "significant" if p_val < 0.05 else "non-significant"
            line = (
                f"The main effect of {source} was {sig}, "
                f"F({df1_str}, {df2_str}) = {F_str}, {p_str}"
            )
            if eta2 is not None:
                line += f", partial η² = {fmt_stat(eta2)}"
            line += "."

            lines.append(line)
            term_stats: dict = {
                "source": source,
                "F": round(F_val, 4),
                "df1": df1,
                "df2": df2,
                "p": round(p_val, 4),
            }
            if eta2 is not None:
                term_stats["partial_eta2"] = round(eta2, 4)
            statistics["terms"].append(term_stats)

        text = " ".join(lines)
        return Report(text, statistics, self._warnings)
