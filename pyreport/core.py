"""
Core Report class and report() dispatcher.
"""

from __future__ import annotations

import warnings as _warnings
from typing import List, Optional

import pandas as pd

from .utils import detect_type, _sanitize_stats


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
        self._statistics = _sanitize_stats(statistics)
        self._warnings = warnings or []

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """Human-readable plain-English summary (APA 7 formatted)."""
        return self._text

    @property
    def statistics(self) -> dict:
        """Dictionary of all extracted statistics (read-only copy)."""
        return dict(self._statistics)

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

    Raw data shortcut
    -----------------
    If *obj* is a tuple or list of array-like objects, you can pass
    ``test=`` to have pyreport run the underlying SciPy test and
    immediately return a Report::

        report((x, y), test="ttest")
        report((x, y), test="mannwhitney", n1=30, n2=30)
        report((x, y), test="correlation", n=50)
        report((x, y, z), test="kruskal", k=3, n=90)
        report((x - y,), test="wilcoxon", n=30)

    Supported ``test`` values: ``"ttest"``, ``"mannwhitney"``,
    ``"correlation"``, ``"kruskal"``, ``"wilcoxon"``.

    Parameters
    ----------
    obj:
        Statistical result object, ``pd.DataFrame``, or a tuple/list of
        raw array-like data when used with ``test=``.
    **kwargs:
        Passed to the underlying reporter.  Common options:

        - ``test``        (str)              — raw-data shortcut test name
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
    # ------------------------------------------------------------------
    # Raw-data shortcut: report((x, y), test="ttest")
    # ------------------------------------------------------------------
    test_name = kwargs.pop("test", None)
    if test_name is not None:
        obj = _run_raw_test(obj, test_name)

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
# _run_raw_test() helper for raw-data shortcut
# ---------------------------------------------------------------------------

_RAW_TEST_ALIASES = {
    "ttest": ("ttest", "t-test", "t_test", "ttest_ind"),
    "mannwhitney": ("mannwhitney", "mann_whitney", "mannwhitneyu", "mwu"),
    "correlation": ("correlation", "pearsonr", "pearson", "corr"),
    "kruskal": ("kruskal", "kruskalwallis", "kruskal_wallis"),
    "wilcoxon": ("wilcoxon", "wilcoxon_signed_rank"),
}

_CANONICAL = {alias: canon for canon, aliases in _RAW_TEST_ALIASES.items() for alias in aliases}


def _run_raw_test(obj, test: str):
    """
    Run a SciPy statistical test on raw array-like data and return the result.

    Parameters
    ----------
    obj:
        A single array-like (for wilcoxon) or a tuple/list of array-like objects.
    test:
        Name of the test to run (case-insensitive).  Supported: ``"ttest"``,
        ``"mannwhitney"``, ``"correlation"``, ``"kruskal"``, ``"wilcoxon"``.

    Returns
    -------
    SciPy result object.

    Raises
    ------
    ReportError
        If the test name is unrecognised or the data shape is incompatible.
    """
    from scipy import stats as _stats

    canon = _CANONICAL.get(test.lower().replace("-", "_").replace(" ", "_"))
    if canon is None:
        raise ReportError(
            f"Unknown test name {test!r} for raw-data shortcut. "
            f"Supported values: {sorted(_RAW_TEST_ALIASES)}"
        )

    # Normalise obj to a list of arrays
    import numpy as _np

    if isinstance(obj, (_np.ndarray,)):
        arrays = [obj]
    elif isinstance(obj, (list, tuple)) and all(
        isinstance(a, (_np.ndarray, list)) for a in obj
    ):
        arrays = [_np.asarray(a, float) for a in obj]
    else:
        # Try treating obj itself as a single array
        try:
            arrays = [_np.asarray(obj, float)]
        except (TypeError, ValueError):
            raise ReportError(
                f"Cannot convert {type(obj).__name__!r} to array for test={test!r}. "
                "Pass a numpy array, list, or tuple of arrays."
            )

    if canon == "ttest":
        if len(arrays) < 2:
            raise ReportError("test='ttest' requires two data arrays, e.g. report((x, y), test='ttest').")
        return _stats.ttest_ind(arrays[0], arrays[1])

    if canon == "mannwhitney":
        if len(arrays) < 2:
            raise ReportError("test='mannwhitney' requires two data arrays.")
        return _stats.mannwhitneyu(arrays[0], arrays[1])

    if canon == "correlation":
        if len(arrays) < 2:
            raise ReportError("test='correlation' requires two data arrays.")
        return _stats.pearsonr(arrays[0], arrays[1])

    if canon == "kruskal":
        if len(arrays) < 2:
            raise ReportError("test='kruskal' requires at least two data arrays.")
        return _stats.kruskal(*arrays)

    if canon == "wilcoxon":
        arr = arrays[0] if len(arrays) == 1 else arrays[0] - arrays[1]
        return _stats.wilcoxon(arr)

    raise ReportError(f"Unhandled test name {test!r}.")  # pragma: no cover


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
