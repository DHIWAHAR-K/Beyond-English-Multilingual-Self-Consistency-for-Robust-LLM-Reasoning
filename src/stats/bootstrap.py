from typing import Dict, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]

DEFAULT_ITERATIONS = 702
DEFAULT_ALPHA = 0.01
DEFAULT_MOE_THRESHOLD = 2.0


def bootstrap_ci(
    values: ArrayLike,
    n: int = DEFAULT_ITERATIONS,
    alpha: float = DEFAULT_ALPHA,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for the mean.

    Uses percentile method at alpha/2 and 1-alpha/2 (default 0.5 and 99.5
    for 99% CI with alpha=0.01).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must contain at least one element")

    generator = rng or np.random.default_rng()
    sample_indices = generator.integers(0, arr.size, size=(n, arr.size))
    bootstrap_means = arr[sample_indices].mean(axis=1)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    lower, upper = np.percentile(bootstrap_means, [lower_pct, upper_pct])
    mean = float(arr.mean())
    margin_pct = float((upper - lower) / (2.0 * mean) * 100.0) if mean != 0 else 0.0

    return {
        "mean": mean,
        "lower": float(lower),
        "upper": float(upper),
        "margin_pct": margin_pct,
        "n_samples": int(arr.size),
        "n_iterations": n,
        "alpha": alpha,
    }


def meets_significance_threshold(
    margin_pct: float,
    threshold: float = DEFAULT_MOE_THRESHOLD,
) -> bool:
    """Return True if margin of error is within the ±threshold percent bound."""
    return abs(margin_pct) <= threshold
