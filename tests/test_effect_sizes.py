"""
Tests for pyreport.effect_sizes.
"""

import math
import pytest
import numpy as np
from pyreport.effect_sizes import (
    cohens_d,
    cohens_d_from_t,
    pearson_r_from_t,
    eta_squared,
    omega_squared,
    odds_ratio,
    rank_biserial,
)


RNG = np.random.default_rng(42)


class TestCohensD:
    def test_large_effect(self):
        x = RNG.normal(5.0, 1.0, 50)
        y = RNG.normal(3.0, 1.0, 50)
        d, label = cohens_d(x, y)
        assert abs(d) > 0.8
        assert label == "large"

    def test_small_effect(self):
        x = RNG.normal(5.0, 1.0, 50)
        y = RNG.normal(5.1, 1.0, 50)
        d, label = cohens_d(x, y)
        assert label in ("small", "medium")

    def test_paired(self):
        x = RNG.normal(5.0, 1.0, 30)
        y = x + RNG.normal(0.5, 0.3, 30)  # y is systematically larger
        d, _ = cohens_d(x, y, paired=True)
        assert d < 0  # x - y < 0 since y > x on average

    def test_zero_variance(self):
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([2.0, 2.0, 2.0])
        d, _ = cohens_d(x, y)
        assert d == 0.0  # pooled SD is 0

    def test_custom_thresholds(self):
        x = RNG.normal(5.0, 1.0, 50)
        y = RNG.normal(3.0, 1.0, 50)
        _, label = cohens_d(x, y, thresholds={"small": 0.1, "medium": 0.3, "large": 0.5})
        assert label == "large"


class TestCohensD_from_t:
    def test_independent(self):
        d, _ = cohens_d_from_t(t=2.5, n1=30, n2=30)
        assert abs(d) > 0

    def test_one_sample(self):
        d, _ = cohens_d_from_t(t=2.5, n1=30)
        assert abs(d) > 0

    def test_negative_t(self):
        d, _ = cohens_d_from_t(t=-2.5, n1=30, n2=30)
        assert d < 0


class TestPearsonR_from_t:
    def test_positive_t(self):
        r, label = pearson_r_from_t(t=3.0, df=28)
        assert 0 < r < 1
        assert label in ("small", "medium", "large")

    def test_zero_t(self):
        r, _ = pearson_r_from_t(t=0.0, df=28)
        assert r == 0.0


class TestEtaSquared:
    def test_basic(self):
        eta2, label = eta_squared(ss_effect=30, ss_total=100)
        assert abs(eta2 - 0.3) < 1e-6
        assert label == "large"

    def test_zero_total(self):
        eta2, _ = eta_squared(ss_effect=10, ss_total=0)
        assert eta2 == 0.0


class TestOmegaSquared:
    def test_positive(self):
        omega2, label = omega_squared(F=5.0, df_effect=2, df_error=57, n=60)
        assert omega2 >= 0
        assert label in ("small", "medium", "large")

    def test_negative_clamped(self):
        omega2, _ = omega_squared(F=0.5, df_effect=2, df_error=57, n=60)
        assert omega2 == 0.0


class TestOddsRatio:
    def test_basic_2x2(self):
        table = [[10, 5], [3, 12]]
        or_val, (lo, hi) = odds_ratio(table)
        assert or_val > 1
        assert lo < or_val < hi

    def test_cell_zero(self):
        table = [[10, 0], [3, 12]]
        or_val, (lo, hi) = odds_ratio(table)
        assert or_val > 0
        assert lo < hi


class TestRankBiserial:
    def test_typical(self):
        r, label = rank_biserial(U=80, n1=10, n2=10)
        assert -1 <= r <= 1
        assert label in ("small", "medium", "large")

    def test_max_u(self):
        # U = n1*n2 → r = 1 - 2 = -1
        r, _ = rank_biserial(U=100, n1=10, n2=10)
        assert abs(r - (-1.0)) < 1e-6
