"""
Tests for pyreport.core — Report class and report() dispatcher.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

import pyreport
from pyreport.core import Report, ReportError, report


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Report class tests
# ---------------------------------------------------------------------------

class TestReportClass:
    def _make_report(self, text="Hello.", stats=None, warnings=None):
        return Report(text, stats or {"a": 1}, warnings or [])

    def test_repr(self):
        r = self._make_report("A test result.")
        assert "A test result." in repr(r)

    def test_to_text_no_warnings(self):
        r = self._make_report("Clean.")
        assert r.to_text() == "Clean."

    def test_to_text_with_warnings(self):
        r = Report("Body.", {"x": 1}, ["Watch out!"])
        text = r.to_text()
        assert "Body." in text
        assert "Watch out!" in text

    def test_to_dict(self):
        r = self._make_report(stats={"t": 2.3, "p": 0.04})
        d = r.to_dict()
        assert d["t"] == 2.3
        assert d["p"] == 0.04

    def test_to_dict_is_copy(self):
        r = self._make_report(stats={"t": 2.3})
        d = r.to_dict()
        d["extra"] = 99
        assert "extra" not in r.to_dict()

    def test_to_dataframe(self):
        r = self._make_report(stats={"t": 2.3, "p": 0.04})
        df = r.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "t" in df.columns
        assert "p" in df.columns

    def test_add_two_reports(self):
        r1 = Report("First.", {"a": 1}, ["warn1"])
        r2 = Report("Second.", {"b": 2}, ["warn2"])
        combined = r1 + r2
        assert "First." in combined.to_text()
        assert "Second." in combined.to_text()
        assert combined.to_dict()["a"] == 1
        assert combined.to_dict()["b"] == 2
        assert "warn1" in combined._warnings
        assert "warn2" in combined._warnings

    def test_add_wrong_type_raises(self):
        r = self._make_report()
        with pytest.raises(TypeError):
            _ = r + "not a report"


# ---------------------------------------------------------------------------
# report() dispatcher tests
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_unsupported_raises(self):
        with pytest.raises(ReportError):
            report(42)

    def test_unsupported_list_raises(self):
        with pytest.raises(ReportError):
            report([1, 2, 3])

    def test_dataframe_dispatched(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        r = report(df)
        assert isinstance(r, Report)
        assert "dataframe" in r.to_text().lower() or "observation" in r.to_text().lower()

    def test_ttest_dispatched(self):
        x = RNG.normal(5.0, 1.0, 30)
        y = RNG.normal(3.5, 1.0, 30)
        result = stats.ttest_ind(x, y)
        r = report(result)
        assert isinstance(r, Report)
        assert "t-test" in r.to_text().lower()

    def test_correlation_dispatched(self):
        x = RNG.normal(0, 1, 50)
        y = x + RNG.normal(0, 0.5, 50)
        result = stats.pearsonr(x, y)
        r = report(result, n=50)
        assert isinstance(r, Report)

    def test_chi2_dispatched(self):
        table = [[10, 20], [30, 40]]
        result = stats.chi2_contingency(table)
        r = report(result)
        assert isinstance(r, Report)
        assert "chi" in r.to_text().lower()

    def test_mannwhitney_dispatched(self):
        x = RNG.normal(5, 1, 30)
        y = RNG.normal(4, 1, 30)
        result = stats.mannwhitneyu(x, y)
        r = report(result, n1=30, n2=30)
        assert isinstance(r, Report)
        assert "mann-whitney" in r.to_text().lower()

    def test_kruskal_dispatched(self):
        x = RNG.normal(5, 1, 30)
        y = RNG.normal(4, 1, 30)
        z = RNG.normal(3, 1, 30)
        result = stats.kruskal(x, y, z)
        r = report(result, k=3, n=90)
        assert isinstance(r, Report)
        assert "kruskal" in r.to_text().lower()

    def test_wilcoxon_dispatched(self):
        x = RNG.normal(5, 1, 30)
        y = RNG.normal(4, 1, 30)
        result = stats.wilcoxon(x - y)
        r = report(result, n=30)
        assert isinstance(r, Report)
        assert "wilcoxon" in r.to_text().lower()
