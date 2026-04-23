"""
Core Report class and report() dispatcher.
"""

from __future__ import annotations

import warnings as _warnings
from typing import List, Optional

import pandas as pd

from .utils import detect_type


# ---------------------------------------------------------------------------
# Custom exceptions / warnings
# ---------------------------------------------------------------------------


class ReportError(Exception):
    """Raised when report() receives an unsupported object type."""


class ReportWarning(UserWarning):
    """Emitted for borderline statistical situations."""


# ---------------------------------------------------------------------------
# Report class
# ---------------------------------------------------------------------------


class Report:
    """
    A publication-ready statistical report.

    Attributes
    ----------
    text:
        Human-readable plain-English summary (APA 7 formatted).
    statistics:
        Dictionary of all extracted statistics.
    warnings:
        List of warning messages about assumption violations or borderline cases.
    """

    def __init__(
        self,
        text: str,
        statistics: dict,
        warnings: Optional[List[str]] = None,
    ):
        self._text = text
        self._statistics = statistics
        self._warnings = warnings or []

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        """Return the full report as a plain string, including any warnings."""
        parts = [self._text]
        if self._warnings:
            parts.append("\nWarnings:")
            for w in self._warnings:
                parts.append(f"  • {w}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Return all extracted statistics as a dictionary."""
        return dict(self._statistics)

    def to_dataframe(self) -> pd.DataFrame:
        """Return statistics as a single-row (or multi-row) pandas DataFrame."""
        data = {}
        for k, v in self._statistics.items():
            if isinstance(v, (list, tuple)):
                data[k] = [str(v)]
            else:
                data[k] = [v]
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()

    def __add__(self, other: "Report") -> "Report":
        """Concatenate two Report objects."""
        if not isinstance(other, Report):
            raise TypeError(f"Cannot add Report and {type(other).__name__}")
        combined_text = self._text + "\n\n" + other._text
        combined_stats = {**self._statistics, **other._statistics}
        combined_warnings = self._warnings + other._warnings
        return Report(combined_text, combined_stats, combined_warnings)


# ---------------------------------------------------------------------------
# report() dispatcher
# ---------------------------------------------------------------------------


def report(obj, **kwargs) -> Report:
    """
    Auto-detect *obj* type and return a :class:`Report`.

    Supported input types
    ---------------------
    - ``pd.DataFrame``            → descriptive statistics
    - scipy t-test result         → t-test report
    - scipy correlation result    → correlation report
    - scipy chi2_contingency      → chi-square report
    - scipy fisher_exact          → Fisher's exact report
    - scipy mannwhitneyu          → Mann-Whitney U report
    - scipy kruskal               → Kruskal-Wallis report
    - scipy wilcoxon              → Wilcoxon signed-rank report
    - statsmodels OLS result      → linear regression report
    - statsmodels Logit/GLM       → logistic/GLM regression report
    - pingouin t-test DataFrame   → t-test report (pingouin)
    - pingouin ANOVA DataFrame    → ANOVA report (pingouin)
    - pingouin correlation        → correlation report (pingouin)

    Parameters
    ----------
    obj:
        Statistical object to report.
    **kwargs:
        Passed to the underlying reporter.  Common options:

        - ``effectsize`` (bool, default True) — include effect sizes
        - ``verbose``   (bool, default True) — include interpretation text
        - ``ci_level``  (float, default 0.95) — confidence level
        - ``group_names`` (tuple[str,str])   — names for t-test groups
        - ``outcome_name`` (str)             — name of outcome variable

    Returns
    -------
    Report

    Raises
    ------
    ReportError
        If the object type is not supported.
    """
    obj_type = detect_type(obj)

    # Lazy imports of reporter classes to avoid circular imports
    if obj_type == "dataframe":
        from .reporters.dataframe import DataFrameReporter
        return DataFrameReporter(obj, **kwargs).report()

    elif obj_type in ("ttest", "pingouin_ttest"):
        from .reporters.ttest import TTestReporter
        return TTestReporter(obj, **kwargs).report()

    elif obj_type in ("correlation", "pingouin_correlation"):
        from .reporters.correlation import CorrelationReporter
        return CorrelationReporter(obj, **kwargs).report()

    elif obj_type == "chi2":
        from .reporters.chi_square import ChiSquareReporter
        return ChiSquareReporter(obj, **kwargs).report()

    elif obj_type == "fisher":
        from .reporters.chi_square import FisherReporter
        return FisherReporter(obj, **kwargs).report()

    elif obj_type == "mannwhitney":
        from .reporters.nonparametric import MannWhitneyReporter
        return MannWhitneyReporter(obj, **kwargs).report()

    elif obj_type == "wilcoxon":
        from .reporters.nonparametric import WilcoxonReporter
        return WilcoxonReporter(obj, **kwargs).report()

    elif obj_type == "kruskal":
        from .reporters.nonparametric import KruskalWallisReporter
        return KruskalWallisReporter(obj, **kwargs).report()

    elif obj_type in ("pingouin_anova",):
        from .reporters.anova import PingouinAnovaReporter
        return PingouinAnovaReporter(obj, **kwargs).report()

    elif obj_type == "ols_regression":
        from .reporters.regression import OLSReporter
        return OLSReporter(obj, **kwargs).report()

    elif obj_type == "logistic_regression":
        from .reporters.regression import LogisticReporter
        return LogisticReporter(obj, **kwargs).report()

    else:
        raise ReportError(
            f"Unsupported object type: {type(obj).__name__!r}. "
            "See pyreport.report() docstring for supported types."
        )


# ---------------------------------------------------------------------------
# report_table() convenience function
# ---------------------------------------------------------------------------


def report_table(obj, **kwargs) -> pd.DataFrame:
    """
    Return a formatted pd.DataFrame summary (like report_table() in R).

    Useful for ANOVA and regression tables.

    Parameters
    ----------
    obj:
        Statistical object.
    **kwargs:
        Passed to the underlying reporter.
    """
    r = report(obj, **kwargs)
    return r.to_dataframe()
