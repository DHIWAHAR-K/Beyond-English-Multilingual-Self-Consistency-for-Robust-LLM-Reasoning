import numpy as np
import pytest

from src.stats.bootstrap import (
    DEFAULT_ITERATIONS,
    bootstrap_ci,
    meets_significance_threshold,
)


def test_bootstrap_ci_returns_expected_keys():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=0.75, scale=0.05, size=200)
    result = bootstrap_ci(values, n=100, rng=rng)

    assert set(result.keys()) == {
        "mean",
        "lower",
        "upper",
        "margin_pct",
        "n_samples",
        "n_iterations",
        "alpha",
    }
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_bootstrap_ci_margin_within_threshold_for_large_sample():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=1.0, scale=0.02, size=1000)
    result = bootstrap_ci(values, n=DEFAULT_ITERATIONS, rng=rng)

    assert meets_significance_threshold(result["margin_pct"], threshold=2.0)


def test_bootstrap_ci_raises_on_empty_input():
    with pytest.raises(ValueError, match="at least one element"):
        bootstrap_ci([])


def test_meets_significance_threshold():
    assert meets_significance_threshold(1.5) is True
    assert meets_significance_threshold(3.0) is False
