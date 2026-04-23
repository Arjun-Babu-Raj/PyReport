"""
Tests for pyreport.formatters — APA 7 edge cases.
"""

import pytest
from pyreport.formatters import (
    fmt_stat,
    fmt_p,
    fmt_p_full,
    fmt_bound,
    fmt_ci,
    fmt_df,
)


# ---------------------------------------------------------------------------
# fmt_stat
# ---------------------------------------------------------------------------

class TestFmtStat:
    def test_basic_positive(self):
        assert fmt_stat(3.14159) == "3.14"

    def test_basic_negative(self):
        assert fmt_stat(-2.7182) == "-2.72"

    def test_negative_zero_guarded(self):
        assert fmt_stat(-0.001) == "0.00"

    def test_zero(self):
        assert fmt_stat(0.0) == "0.00"

    def test_decimals_override(self):
        assert fmt_stat(1.23456, decimals=4) == "1.2346"

    def test_large_value(self):
        assert fmt_stat(24.213) == "24.21"

    def test_integer_value(self):
        assert fmt_stat(5.0) == "5.00"


# ---------------------------------------------------------------------------
# fmt_p
# ---------------------------------------------------------------------------

class TestFmtP:
    def test_very_small_p(self):
        assert fmt_p(0.0) == "< .001"

    def test_p_below_threshold(self):
        assert fmt_p(0.0009) == "< .001"

    def test_p_at_threshold(self):
        # 0.001 is NOT < 0.001
        assert fmt_p(0.001) == ".001"

    def test_typical_p(self):
        assert fmt_p(0.043) == ".043"

    def test_p_one(self):
        assert fmt_p(1.0) == "1.000"

    def test_no_leading_zero(self):
        result = fmt_p(0.05)
        assert not result.startswith("0")

    def test_p_large(self):
        assert fmt_p(0.365) == ".365"


# ---------------------------------------------------------------------------
# fmt_p_full
# ---------------------------------------------------------------------------

class TestFmtPFull:
    def test_small_p(self):
        assert fmt_p_full(0.0001) == "p < .001"

    def test_medium_p(self):
        assert fmt_p_full(0.043) == "p = .043"

    def test_p_one(self):
        assert fmt_p_full(1.0) == "p = 1.000"


# ---------------------------------------------------------------------------
# fmt_bound
# ---------------------------------------------------------------------------

class TestFmtBound:
    def test_positive_less_than_one(self):
        assert fmt_bound(0.75) == ".75"

    def test_negative_less_than_one(self):
        assert fmt_bound(-0.32) == "-.32"

    def test_exactly_one(self):
        assert fmt_bound(1.0) == "1.00"

    def test_negative_zero(self):
        assert fmt_bound(-0.001) == ".00"

    def test_zero(self):
        assert fmt_bound(0.0) == ".00"


# ---------------------------------------------------------------------------
# fmt_ci
# ---------------------------------------------------------------------------

class TestFmtCI:
    def test_default_95(self):
        result = fmt_ci(0.21, 1.28)
        assert result == "95% CI [0.21, 1.28]"

    def test_99_level(self):
        result = fmt_ci(-1.0, 2.0, level=0.99)
        assert result.startswith("99% CI")

    def test_negative_bounds(self):
        result = fmt_ci(-0.50, -0.10)
        assert "-0.50" in result and "-0.10" in result

    def test_decimals_override(self):
        result = fmt_ci(0.123, 0.456, decimals=3)
        assert "0.123" in result


# ---------------------------------------------------------------------------
# fmt_df
# ---------------------------------------------------------------------------

class TestFmtDf:
    def test_integer_df(self):
        assert fmt_df(10.0) == "10"

    def test_welch_df(self):
        result = fmt_df(47.32)
        assert result == "47.32"

    def test_df_zero_decimal(self):
        assert fmt_df(5.0) == "5"
