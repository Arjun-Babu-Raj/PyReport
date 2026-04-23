"""
Non-parametric test reporters:
  - Mann-Whitney U
  - Wilcoxon signed-rank
  - Kruskal-Wallis H
"""

from __future__ import annotations

from ..core import Report
from ..formatters import fmt_stat, fmt_p_full
from ..effect_sizes import rank_biserial
from .base import BaseReporter


class MannWhitneyReporter(BaseReporter):
    """
    Report a Mann-Whitney U test.

    Accepts scipy ``mannwhitneyu`` result.

    kwargs
    ------
    group_names : tuple[str, str]
        Names of the two groups (default: ("Group A", "Group B")).
    n1, n2 : int
        Sample sizes (required for rank-biserial r).
    effectsize : bool
        Whether to include rank-biserial r (default: True).
    """

    def report(self) -> Report:
        obj = self.obj
        U_val = float(obj.statistic)
        p_val = float(obj.pvalue)

        group_names = self.kwargs.get("group_names", ("Group A", "Group B"))
        n1: int = self.kwargs.get("n1", None)
        n2: int = self.kwargs.get("n2", None)
        include_es: bool = self.kwargs.get("effectsize", True)
        g1, g2 = group_names

        r_val = r_interp = None
        if include_es and n1 is not None and n2 is not None:
            r_val, r_interp = rank_biserial(U_val, n1, n2)

        sig_word = "significant" if p_val < 0.05 else "non-significant"
        U_str = fmt_stat(U_val)
        p_str = fmt_p_full(p_val)

        text = (
            f"A Mann-Whitney U test indicated a {sig_word} difference "
            f"between {g1} and {g2}, U = {U_str}, {p_str}"
        )
        if r_val is not None:
            text += f", r = {fmt_stat(r_val)}"
            if r_interp:
                text += f". The effect size is considered {r_interp}."
            else:
                text += "."
        else:
            text += "."

        statistics: dict = {
            "test": "Mann-Whitney U",
            "U": round(U_val, 4),
            "p": round(p_val, 4),
        }
        if r_val is not None:
            statistics["rank_biserial_r"] = round(r_val, 4)
        if r_interp:
            statistics["effect_size_label"] = r_interp
        if n1 is not None:
            statistics["n1"] = n1
        if n2 is not None:
            statistics["n2"] = n2

        return Report(text, statistics, self._warnings)


class WilcoxonReporter(BaseReporter):
    """
    Report a Wilcoxon signed-rank test.

    Accepts scipy ``wilcoxon`` result.

    kwargs
    ------
    n : int
        Sample size (number of non-zero differences).
    effectsize : bool
        Whether to include r = Z/√N (default: True).
    """

    def report(self) -> Report:
        obj = self.obj
        W_val = float(obj.statistic)
        p_val = float(obj.pvalue)

        n: int = self.kwargs.get("n", None)
        include_es: bool = self.kwargs.get("effectsize", True)

        r_val = r_interp = None
        # r = Z/√N approximation; scipy returns T (sum of signed ranks)
        # We report W directly without Z-conversion since scipy doesn't expose Z
        # But we can compute a rough effect size if n is available
        if include_es and n is not None and n > 0:
            # Use r = 1 - 4W / (n(n+1)) approximation
            r_val = 1 - (4 * W_val) / (n * (n + 1))
            from ..effect_sizes import _interpret, _R_THRESHOLDS
            r_interp = _interpret(abs(r_val), _R_THRESHOLDS)

        sig_word = "significant" if p_val < 0.05 else "non-significant"
        W_str = fmt_stat(W_val)
        p_str = fmt_p_full(p_val)

        text = (
            f"A Wilcoxon signed-rank test indicated a {sig_word} difference, "
            f"W = {W_str}, {p_str}"
        )
        if r_val is not None:
            text += f", r = {fmt_stat(r_val)}"
            if r_interp:
                text += f". The effect size is considered {r_interp}."
            else:
                text += "."
        else:
            text += "."

        statistics: dict = {
            "test": "Wilcoxon signed-rank",
            "W": round(W_val, 4),
            "p": round(p_val, 4),
        }
        if r_val is not None:
            statistics["r"] = round(r_val, 4)
        if r_interp:
            statistics["effect_size_label"] = r_interp
        if n is not None:
            statistics["n"] = n

        return Report(text, statistics, self._warnings)


class KruskalWallisReporter(BaseReporter):
    """
    Report a Kruskal-Wallis H test.

    Accepts scipy ``kruskal`` result.

    kwargs
    ------
    k : int
        Number of groups.
    n : int
        Total sample size (for η²).
    effectsize : bool
        Whether to include η² (default: True).
    """

    def report(self) -> Report:
        obj = self.obj
        H_val = float(obj.statistic)
        p_val = float(obj.pvalue)

        k: int = self.kwargs.get("k", None)
        n: int = self.kwargs.get("n", None)
        include_es: bool = self.kwargs.get("effectsize", True)

        eta2 = eta_interp = None
        if include_es and n is not None and n > 0:
            # η² = (H - k + 1) / (n - k)
            if k is not None and n > k:
                eta2 = (H_val - k + 1) / (n - k)
                eta2 = max(0.0, eta2)
                from ..effect_sizes import _interpret, _ETA_THRESHOLDS
                eta_interp = _interpret(eta2, _ETA_THRESHOLDS)

        sig_word = "significant" if p_val < 0.05 else "non-significant"
        H_str = fmt_stat(H_val)
        p_str = fmt_p_full(p_val)

        df_str = str(k - 1) if k is not None else "?"

        text = (
            f"A Kruskal-Wallis H test indicated a {sig_word} difference "
            f"across groups, H({df_str}) = {H_str}, {p_str}"
        )
        if eta2 is not None:
            text += f", η² = {fmt_stat(eta2)}"
            if eta_interp:
                text += f". The effect size is considered {eta_interp}."
            else:
                text += "."
        else:
            text += "."

        statistics: dict = {
            "test": "Kruskal-Wallis H",
            "H": round(H_val, 4),
            "p": round(p_val, 4),
        }
        if k is not None:
            statistics["k"] = k
            statistics["df"] = k - 1
        if n is not None:
            statistics["n"] = n
        if eta2 is not None:
            statistics["eta_squared"] = round(eta2, 4)
        if eta_interp:
            statistics["effect_size_label"] = eta_interp

        return Report(text, statistics, self._warnings)
