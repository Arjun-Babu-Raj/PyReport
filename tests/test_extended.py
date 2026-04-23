"""
Extended tests for better coverage of reporters and utils.
"""

import math
import warnings
import pytest
import numpy as np
import pandas as pd
from scipy import stats

import pyreport
from pyreport.core import Report, ReportError, report, report_table


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Fisher's exact test
# ---------------------------------------------------------------------------

class TestFisherReporter:
    def _table(self):
        return np.array([[8, 2], [1, 9]])

    def test_basic(self):
        table = self._table()
        result = stats.fisher_exact(table)
        r = report(result, table=table)
        text = r.to_text()
        assert "fisher" in text.lower()
        assert "OR" in text or "odds" in text.lower()

    def test_p_format(self):
        table = self._table()
        result = stats.fisher_exact(table)
        r = report(result, table=table)
        d = r.to_dict()
        assert "odds_ratio" in d
        assert "p" in d
        assert "ci_lower" in d

    def test_without_table(self):
        table = self._table()
        result = stats.fisher_exact(table)
        r = report(result)
        assert isinstance(r, Report)

    def test_nonsignificant(self):
        table = np.array([[12, 11], [10, 13]])
        result = stats.fisher_exact(table)
        r = report(result)
        assert "non-significant" in r.to_text()


# ---------------------------------------------------------------------------
# OLS Regression — deeper coverage
# ---------------------------------------------------------------------------

class TestOLSReporterExtended:
    def _fit(self, n=100):
        import statsmodels.api as sm
        x1 = RNG.normal(0, 1, n)
        x2 = RNG.normal(0, 1, n)
        x3 = RNG.normal(0, 1, n)
        y = 0.32 * x1 + 0.45 * x2 + 0.08 * x3 + RNG.normal(0, 1, n)
        X = sm.add_constant(np.column_stack([x1, x2, x3]))
        X_df = pd.DataFrame(X, columns=["const", "Age", "income", "Education"])
        return sm.OLS(y, X_df).fit()

    def test_r2_present(self):
        model = self._fit()
        r = report(model)
        d = r.to_dict()
        assert "R2" in d
        assert 0 <= d["R2"] <= 1

    def test_adj_r2_present(self):
        model = self._fit()
        r = report(model)
        d = r.to_dict()
        assert "adj_R2" in d

    def test_report_table(self):
        model = self._fit()
        tbl = report_table(model)
        assert isinstance(tbl, pd.DataFrame)

    def test_text_mentions_predictors(self):
        model = self._fit()
        r = report(model)
        text = r.to_text()
        assert "Age" in text or "income" in text or "Education" in text

    def test_text_mentions_significant_nonsignificant(self):
        model = self._fit()
        r = report(model)
        text = r.to_text()
        # Should contain significance language
        assert "significant" in text.lower()


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class TestLogisticReporter:
    def _fit_logistic(self):
        import statsmodels.api as sm
        n = 200
        x1 = RNG.normal(0, 1, n)
        x2 = RNG.normal(0, 1, n)
        logit_p = 0.5 * x1 - 0.3 * x2
        p = 1 / (1 + np.exp(-logit_p))
        y = RNG.binomial(1, p, n)
        X = sm.add_constant(np.column_stack([x1, x2]))
        X_df = pd.DataFrame(X, columns=["const", "x1", "x2"])
        model = sm.Logit(y, X_df).fit(disp=0)
        return model

    def test_basic(self):
        model = self._fit_logistic()
        r = report(model)
        assert isinstance(r, Report)
        text = r.to_text()
        assert "logistic" in text.lower()

    def test_dict_keys(self):
        model = self._fit_logistic()
        r = report(model)
        d = r.to_dict()
        assert "test" in d
        assert d["test"] == "logistic regression"
        assert "predictors" in d
        assert "log_likelihood" in d

    def test_predictors_have_or(self):
        model = self._fit_logistic()
        r = report(model)
        d = r.to_dict()
        for pred_name, pred_stats in d["predictors"].items():
            assert "OR" in pred_stats
            assert "beta" in pred_stats


# ---------------------------------------------------------------------------
# Pingouin ANOVA
# ---------------------------------------------------------------------------

class TestPingouinAnovaReporter:
    def _make_pg_anova_df(self):
        """Manually construct a pingouin-style ANOVA result DataFrame."""
        return pd.DataFrame({
            "Source": ["group", "Error"],
            "SS": [10.0, 30.0],
            "DF": [2, 57],
            "MS": [5.0, 0.526],
            "F": [9.51, np.nan],
            "p-unc": [0.0003, np.nan],
            "np2": [0.25, np.nan],
            "ddof1": [2, np.nan],
            "ddof2": [57, np.nan],
        })

    def test_pingouin_anova(self):
        df = self._make_pg_anova_df()
        r = report(df)
        assert isinstance(r, Report)
        text = r.to_text()
        assert "anova" in text.lower() or "group" in text.lower()

    def test_statistics_dict(self):
        df = self._make_pg_anova_df()
        r = report(df)
        d = r.to_dict()
        assert "terms" in d
        assert len(d["terms"]) > 0
        term = d["terms"][0]
        assert "F" in term
        assert "p" in term


# ---------------------------------------------------------------------------
# Utils type detection
# ---------------------------------------------------------------------------

class TestUtils:
    def test_detect_dataframe(self):
        from pyreport.utils import detect_type
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert detect_type(df) == "dataframe"

    def test_detect_ttest(self):
        from pyreport.utils import detect_type
        x = RNG.normal(0, 1, 30)
        y = RNG.normal(0, 1, 30)
        result = stats.ttest_ind(x, y)
        assert detect_type(result) == "ttest"

    def test_detect_pearsonr(self):
        from pyreport.utils import detect_type
        x = RNG.normal(0, 1, 50)
        y = RNG.normal(0, 1, 50)
        result = stats.pearsonr(x, y)
        assert detect_type(result) == "correlation"

    def test_detect_chi2(self):
        from pyreport.utils import detect_type
        table = [[10, 20], [30, 40]]
        result = stats.chi2_contingency(table)
        assert detect_type(result) == "chi2"

    def test_detect_mannwhitney(self):
        from pyreport.utils import detect_type
        x = RNG.normal(0, 1, 30)
        y = RNG.normal(0, 1, 30)
        result = stats.mannwhitneyu(x, y)
        assert detect_type(result) == "mannwhitney"

    def test_detect_kruskal(self):
        from pyreport.utils import detect_type
        x = RNG.normal(0, 1, 30)
        y = RNG.normal(0, 1, 30)
        result = stats.kruskal(x, y)
        assert detect_type(result) == "kruskal"

    def test_detect_ols(self):
        from pyreport.utils import detect_type
        import statsmodels.api as sm
        x = RNG.normal(0, 1, 50)
        y = 2 * x + RNG.normal(0, 1, 50)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        assert detect_type(model) == "ols_regression"

    def test_detect_logistic(self):
        from pyreport.utils import detect_type
        import statsmodels.api as sm
        n = 100
        x = RNG.normal(0, 1, n)
        y = RNG.binomial(1, 0.5, n)
        X = sm.add_constant(x)
        model = sm.Logit(y, X).fit(disp=0)
        assert detect_type(model) == "logistic_regression"

    def test_detect_unknown(self):
        from pyreport.utils import detect_type
        assert detect_type(42) == "unknown"
        assert detect_type("hello") == "unknown"

    def test_detect_pingouin_ttest(self):
        from pyreport.utils import detect_type
        pg_df = pd.DataFrame({
            "T": [2.5], "dof": [28.0], "alternative": ["two-sided"],
            "p-val": [0.02], "cohen-d": [0.5], "CI95%": [[0.1, 0.9]],
            "BF10": [3.4], "power": [0.8],
        })
        assert detect_type(pg_df) == "pingouin_ttest"

    def test_detect_pingouin_anova(self):
        from pyreport.utils import detect_type
        pg_df = pd.DataFrame({
            "Source": ["group"], "F": [9.5], "ddof1": [2], "ddof2": [57], "p-unc": [0.001],
        })
        assert detect_type(pg_df) == "pingouin_anova"


# ---------------------------------------------------------------------------
# Pingouin t-test reporter (via detect → TTestReporter)
# ---------------------------------------------------------------------------

class TestPingouinTTest:
    def _make_pg_ttest(self):
        return pd.DataFrame({
            "T": [2.84],
            "dof": [47.32],
            "alternative": ["two-sided"],
            "p-val": [0.0065],
            "cohen-d": [0.75],
            "CI95%": [[0.21, 1.28]],
            "BF10": [7.2],
            "power": [0.82],
        })

    def test_pingouin_ttest(self):
        df = self._make_pg_ttest()
        r = report(df)
        assert isinstance(r, Report)
        text = r.to_text()
        assert "t-test" in text.lower()
        assert "2.84" in text or "47.32" in text

    def test_dict_has_d(self):
        df = self._make_pg_ttest()
        r = report(df)
        d = r.to_dict()
        assert "cohens_d" in d
        assert abs(d["cohens_d"] - 0.75) < 0.01


# ---------------------------------------------------------------------------
# report_table
# ---------------------------------------------------------------------------

class TestReportTable:
    def test_returns_dataframe(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        tbl = report_table(df)
        assert isinstance(tbl, pd.DataFrame)

    def test_ttest_table(self):
        x = RNG.normal(5, 1, 30)
        y = RNG.normal(3, 1, 30)
        result = stats.ttest_ind(x, y)
        tbl = report_table(result, group_data=(x, y))
        assert isinstance(tbl, pd.DataFrame)
        assert "t" in tbl.columns


# ---------------------------------------------------------------------------
# Wilcoxon without n kwarg (branch without effect size)
# ---------------------------------------------------------------------------

class TestWilcoxonNoN:
    def test_no_n(self):
        x = RNG.normal(5, 1, 30)
        y = RNG.normal(4.5, 1, 30)
        result = stats.wilcoxon(x - y)
        r = report(result)  # no n kwarg
        assert "wilcoxon" in r.to_text().lower()
        d = r.to_dict()
        assert "r" not in d  # no effect size without n


# ---------------------------------------------------------------------------
# Report __add__ with overlapping keys
# ---------------------------------------------------------------------------

class TestReportAdd:
    def test_overlapping_keys(self):
        r1 = Report("First.", {"p": 0.01, "t": 2.3}, [])
        r2 = Report("Second.", {"p": 0.05, "F": 4.5}, [])
        combined = r1 + r2
        # r2 p overwrites r1 p due to dict merge
        d = combined.to_dict()
        assert d["p"] == 0.05
        assert d["t"] == 2.3
        assert d["F"] == 4.5


# ---------------------------------------------------------------------------
# Formatters edge cases
# ---------------------------------------------------------------------------

class TestFormattersEdgeCases:
    def test_p_exactly_001(self):
        from pyreport.formatters import fmt_p
        # p = 0.001 is NOT < 0.001 so should not use "< .001"
        result = fmt_p(0.001)
        assert result == ".001"
        assert "<" not in result

    def test_p_exactly_0(self):
        from pyreport.formatters import fmt_p
        result = fmt_p(0.0)
        assert result == "< .001"

    def test_negative_beta_bound(self):
        from pyreport.formatters import fmt_bound
        result = fmt_bound(-0.87)
        assert result == "-.87"

    def test_fmt_stat_three_decimals(self):
        from pyreport.formatters import fmt_stat
        result = fmt_stat(3.14159, decimals=3)
        assert result == "3.142"
