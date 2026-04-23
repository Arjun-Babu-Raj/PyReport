"""
Type detection utilities for the report() dispatcher.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _safe_import(module: str):
    """Try to import a module; return None on failure."""
    try:
        import importlib
        return importlib.import_module(module)
    except ImportError:
        return None


def is_dataframe(obj) -> bool:
    pd = _safe_import("pandas")
    if pd is None:
        return False
    return isinstance(obj, pd.DataFrame)


def is_scipy_ttest(obj) -> bool:
    """Detect scipy.stats.TtestResult (or legacy Ttest_indResult etc.)."""
    return (
        hasattr(obj, "statistic")
        and hasattr(obj, "pvalue")
        and hasattr(obj, "df")
        and not hasattr(obj, "params")  # exclude statsmodels
    )


def is_scipy_correlation(obj) -> bool:
    """Detect scipy.stats pearsonr / spearmanr / kendalltau result."""
    return (
        hasattr(obj, "statistic")
        and hasattr(obj, "pvalue")
        and not hasattr(obj, "df")
        and not hasattr(obj, "params")
        and (
            # spearmanr / kendalltau expose a .correlation alias
            hasattr(obj, "correlation")
            # pearsonr has a 'pearson' in the class name
            or "pearson" in type(obj).__name__.lower()
        )
    )


def is_scipy_chi2(obj) -> bool:
    """Detect scipy.stats.chi2_contingency result tuple or named tuple."""
    # chi2_contingency returns a named tuple with statistic/pvalue/dof/expected
    return (
        hasattr(obj, "statistic")
        and hasattr(obj, "pvalue")
        and hasattr(obj, "dof")
        and hasattr(obj, "expected_freq")
    )


def is_scipy_fisher(obj) -> bool:
    """Detect scipy.stats.fisher_exact result."""
    return (
        hasattr(obj, "statistic")
        and hasattr(obj, "pvalue")
        and not hasattr(obj, "dof")
        and not hasattr(obj, "df")
        and not hasattr(obj, "params")
    )


def is_scipy_mannwhitney(obj) -> bool:
    """Detect scipy.stats.mannwhitneyu result."""
    return (
        hasattr(obj, "statistic")
        and hasattr(obj, "pvalue")
        and not hasattr(obj, "df")
        and not hasattr(obj, "dof")
        and not hasattr(obj, "params")
    )


def is_scipy_wilcoxon(obj) -> bool:
    """Detect scipy.stats.wilcoxon result."""
    return is_scipy_mannwhitney(obj)  # same duck-type shape


def is_scipy_kruskal(obj) -> bool:
    """Detect scipy.stats.kruskal result."""
    return (
        hasattr(obj, "statistic")
        and hasattr(obj, "pvalue")
        and not hasattr(obj, "df")
        and not hasattr(obj, "dof")
        and not hasattr(obj, "params")
    )


def is_statsmodels_ols(obj) -> bool:
    """Detect a fitted statsmodels OLS result."""
    return (
        hasattr(obj, "params")
        and hasattr(obj, "rsquared")
        and hasattr(obj, "fvalue")
        and not hasattr(obj, "llf")
    ) or (
        hasattr(obj, "params")
        and hasattr(obj, "rsquared")
        and hasattr(obj, "fvalue")
    )


def is_statsmodels_logistic(obj) -> bool:
    """Detect a fitted statsmodels logit/GLM result."""
    return (
        hasattr(obj, "params")
        and hasattr(obj, "llf")
        and hasattr(obj, "prsquared")
    ) or (
        hasattr(obj, "params")
        and hasattr(obj, "llf")
        and hasattr(obj, "df_resid")
        and not hasattr(obj, "rsquared")
    )


def is_pingouin_ttest(obj) -> bool:
    """Detect a pingouin t-test result (DataFrame with specific columns)."""
    pd = _safe_import("pandas")
    if pd is None:
        return False
    if not isinstance(obj, pd.DataFrame):
        return False
    required = {"T", "dof", "alternative", "p-val", "cohen-d"}
    return required.issubset(set(obj.columns))


def is_pingouin_anova(obj) -> bool:
    """Detect a pingouin ANOVA result DataFrame."""
    pd = _safe_import("pandas")
    if pd is None:
        return False
    if not isinstance(obj, pd.DataFrame):
        return False
    required = {"F", "ddof1", "ddof2", "p-unc"}
    return required.issubset(set(obj.columns))


def is_pingouin_correlation(obj) -> bool:
    """Detect a pingouin correlation result DataFrame."""
    pd = _safe_import("pandas")
    if pd is None:
        return False
    if not isinstance(obj, pd.DataFrame):
        return False
    required = {"r", "p-val", "CI95%"}
    return required.issubset(set(obj.columns))


def detect_type(obj) -> str:
    """
    Return a string tag identifying the statistical object type.

    Returns one of:
        'dataframe', 'ttest', 'correlation', 'chi2', 'fisher',
        'mannwhitney', 'kruskal', 'wilcoxon', 'ols_regression',
        'logistic_regression', 'pingouin_ttest', 'pingouin_anova',
        'pingouin_correlation', 'unknown'
    """
    if is_dataframe(obj):
        # pingouin results are DataFrames — check them first
        if is_pingouin_ttest(obj):
            return "pingouin_ttest"
        if is_pingouin_anova(obj):
            return "pingouin_anova"
        if is_pingouin_correlation(obj):
            return "pingouin_correlation"
        return "dataframe"

    if is_statsmodels_logistic(obj):
        return "logistic_regression"
    if is_statsmodels_ols(obj):
        return "ols_regression"

    if is_scipy_chi2(obj):
        return "chi2"

    if is_scipy_ttest(obj):
        return "ttest"

    # At this point obj has statistic/pvalue but no df/dof/params
    # Distinguish by class name where possible
    cls = type(obj).__name__.lower()
    if "pearson" in cls or "spearman" in cls or "kendall" in cls or "correlation" in cls:
        return "correlation"
    if "mannwhitney" in cls or "ranksum" in cls:
        return "mannwhitney"
    if "wilcoxon" in cls:
        return "wilcoxon"
    if "kruskal" in cls:
        return "kruskal"
    if "fisher" in cls:
        return "fisher"

    # Fall back to attribute-based duck-typing for ambiguous SignificanceResult
    if is_scipy_correlation(obj):
        return "correlation"

    # If statistic is positive and no correlation attribute, likely Fisher
    if (
        hasattr(obj, "statistic")
        and hasattr(obj, "pvalue")
        and not hasattr(obj, "df")
        and not hasattr(obj, "dof")
        and not hasattr(obj, "params")
        and not hasattr(obj, "correlation")
        and float(obj.statistic) > 0
    ):
        return "fisher"

    return "unknown"
