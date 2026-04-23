"""
Tests for individual reporters — synthetic data with np.random.seed(42).
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

import pyreport
from pyreport.core import Report, report


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# DataFrameReporter
# ---------------------------------------------------------------------------

class TestDataFrameReporter:
    def _sample_df(self):
        n = 50
        return pd.DataFrame({
            "age": RNG.integers(18, 65, n).astype(float),
            "income": RNG.normal(52000, 14000, n),
            "score": RNG.normal(71, 12, n),
            "gender": RNG.choice(["Female", "Male"], n),
            "group": RNG.choice(["A", "B", "C"], n),
        })

    def test_returns_report(self):
        r = report(self._sample_df())
        assert isinstance(r, Report)

    def test_observation_count(self):
        df = self._sample_df()
        r = report(df)
        assert "50" in r.to_text()

    def test_numeric_stats_in_dict(self):
        df = self._sample_df()
        r = report(df)
        d = r.to_dict()
        assert "age_mean" in d
        assert "income_sd" in d

    def test_categorical_counts_in_dict(self):
        df = self._sample_df()
        r = report(df)
        d = r.to_dict()
        assert "gender_counts" in d

    def test_no_missing(self):
        df = self._sample_df()
        r = report(df)
        assert "No missing" in r.to_text()

    def test_missing_detected(self):
        df = self._sample_df()
        df.loc[0, "age"] = np.nan
        r = report(df)
        assert "missing" in r.to_text().lower()

    def test_to_dataframe(self):
        df = self._sample_df()
        r = report(df)
        out = r.to_dataframe()
        assert isinstance(out, pd.DataFrame)
        assert "n_observations" in out.columns


# ---------------------------------------------------------------------------
# TTestReporter
# ---------------------------------------------------------------------------

class TestTTestReporter:
    def _groups(self):
        x = RNG.normal(4.23, 1.10, 30)
        y = RNG.normal(3.45, 0.98, 30)
        return x, y

    def test_basic_text(self):
        x, y = self._groups()
        result = stats.ttest_ind(x, y)
        r = report(result, group_data=(x, y), group_names=("Group A", "Group B"))
        text = r.to_text()
        assert "t-test" in text.lower()
        assert "Group A" in text
        assert "Group B" in text

    def test_statistics_keys(self):
        x, y = self._groups()
        result = stats.ttest_ind(x, y)
        r = report(result, group_data=(x, y))
        d = r.to_dict()
        assert "t" in d
        assert "df" in d
        assert "p" in d
        assert "cohens_d" in d

    def test_effect_size_label(self):
        x, y = self._groups()
        result = stats.ttest_ind(x, y)
        r = report(result, group_data=(x, y))
        assert "effect size" in r.to_text().lower()

    def test_no_raw_data(self):
        x, y = self._groups()
        result = stats.ttest_ind(x, y)
        r = report(result)
        assert isinstance(r, Report)

    def test_small_n_warning(self):
        x = RNG.normal(5, 1, 10)
        y = RNG.normal(4, 1, 10)
        result = stats.ttest_ind(x, y)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = report(result, group_data=(x, y))
        # Either a warning was raised or stored in the Report
        assert len(w) > 0 or len(r._warnings) > 0

    def test_p_apa_format(self):
        x, y = self._groups()
        result = stats.ttest_ind(x, y)
        r = report(result)
        text = r.to_text()
        # p= should not have leading zero
        import re
        matches = re.findall(r"p = \.?\d+|p < \.", text)
        assert len(matches) > 0

    def test_paired_ttest(self):
        x = RNG.normal(5, 1, 30)
        y = RNG.normal(4.5, 1, 30)
        result = stats.ttest_rel(x, y)
        r = report(result, paired=True)
        assert "paired" in r.to_text().lower()


# ---------------------------------------------------------------------------
# CorrelationReporter
# ---------------------------------------------------------------------------

class TestCorrelationReporter:
    def test_pearson(self):
        x = RNG.normal(0, 1, 50)
        y = x + RNG.normal(0, 0.3, 50)
        result = stats.pearsonr(x, y)
        r = report(result, n=50)
        text = r.to_text()
        assert "pearson" in text.lower() or "correlation" in text.lower()

    def test_spearman(self):
        x = RNG.normal(0, 1, 50)
        y = x + RNG.normal(0, 0.5, 50)
        result = stats.spearmanr(x, y)
        r = report(result, n=50)
        assert isinstance(r, Report)

    def test_dict_keys(self):
        x = RNG.normal(0, 1, 50)
        y = x + RNG.normal(0, 0.3, 50)
        result = stats.pearsonr(x, y)
        r = report(result, n=50)
        d = r.to_dict()
        assert "r" in d
        assert "p" in d


# ---------------------------------------------------------------------------
# ChiSquareReporter
# ---------------------------------------------------------------------------

class TestChiSquareReporter:
    def test_basic(self):
        table = np.array([[30, 10], [15, 45]])
        result = stats.chi2_contingency(table)
        r = report(result)
        text = r.to_text()
        assert "chi" in text.lower()
        assert "χ²" in text

    def test_cramers_v_in_dict(self):
        table = np.array([[30, 10], [15, 45]])
        result = stats.chi2_contingency(table)
        r = report(result, n=100)
        d = r.to_dict()
        assert "cramers_v" in d

    def test_nonsignificant(self):
        table = np.array([[25, 25], [25, 25]])
        result = stats.chi2_contingency(table)
        r = report(result)
        assert "non-significant" in r.to_text()


# ---------------------------------------------------------------------------
# NonparametricReporters
# ---------------------------------------------------------------------------

class TestNonparametricReporters:
    def test_mann_whitney(self):
        x = RNG.normal(5, 1, 40)
        y = RNG.normal(3, 1, 40)
        result = stats.mannwhitneyu(x, y)
        r = report(result, n1=40, n2=40)
        assert "mann-whitney" in r.to_text().lower()
        assert "rank_biserial_r" in r.to_dict()

    def test_wilcoxon(self):
        x = RNG.normal(5, 1, 30)
        y = RNG.normal(4.5, 1, 30)
        result = stats.wilcoxon(x - y)
        r = report(result, n=30)
        assert "wilcoxon" in r.to_text().lower()

    def test_kruskal(self):
        x = RNG.normal(5, 1, 30)
        y = RNG.normal(4, 1, 30)
        z = RNG.normal(3, 1, 30)
        result = stats.kruskal(x, y, z)
        r = report(result, k=3, n=90)
        assert "kruskal" in r.to_text().lower()
        assert "eta_squared" in r.to_dict()


# ---------------------------------------------------------------------------
# OLSReporter
# ---------------------------------------------------------------------------

class TestOLSReporter:
    def _fit_ols(self):
        import statsmodels.api as sm
        n = 100
        x1 = RNG.normal(0, 1, n)
        x2 = RNG.normal(0, 1, n)
        y = 1.5 * x1 + 0.8 * x2 + RNG.normal(0, 1, n)
        X = sm.add_constant(np.column_stack([x1, x2]))
        X_df = pd.DataFrame(X, columns=["const", "age", "income"])
        model = sm.OLS(y, X_df).fit()
        return model

    def test_basic_text(self):
        model = self._fit_ols()
        r = report(model)
        text = r.to_text()
        assert "regression" in text.lower()
        assert "R²" in text or "R\u00b2" in text or "R2" in text.replace("R²", "R2")

    def test_dict_keys(self):
        model = self._fit_ols()
        r = report(model)
        d = r.to_dict()
        assert "R2" in d
        assert "F" in d
        assert "p_model" in d
        assert "predictors" in d

    def test_predictors_in_dict(self):
        model = self._fit_ols()
        r = report(model)
        d = r.to_dict()
        predictors = d["predictors"]
        assert "age" in predictors
        assert "income" in predictors
        assert "beta" in predictors["age"]
        assert "p" in predictors["age"]
