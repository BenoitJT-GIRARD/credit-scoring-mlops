import numpy as np

from credexp.modeling.sensitivity import bootstrap_cost_ci, cost_ratio_sweep


def _sample(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.2, size=n)
    proba = np.clip(y * 0.4 + rng.normal(0.3, 0.2, size=n), 0.01, 0.99)
    return y, proba


def test_sweep_returns_one_row_per_ratio() -> None:
    y, proba = _sample()

    rows = cost_ratio_sweep(y, proba, ratios=(1, 5, 10, 20))

    assert [r["ratio"] for r in rows] == [1, 5, 10, 20]
    assert all(0.0 <= r["threshold"] <= 1.0 for r in rows)


def test_a_costlier_false_negative_lowers_the_optimal_threshold() -> None:
    # Missing a defaulter gets more expensive, so the model should refuse more readily.
    y, proba = _sample()

    rows = cost_ratio_sweep(y, proba, ratios=(1, 50))

    assert rows[1]["threshold"] <= rows[0]["threshold"]


def test_bootstrap_interval_brackets_its_median() -> None:
    y, proba = _sample()

    out = bootstrap_cost_ci(
        y, proba, threshold=0.5, cost_fn=10.0, cost_fp=1.0, n_resamples=200, random_state=0
    )

    assert out["low"] <= out["median"] <= out["high"]
    assert out["n_resamples"] == 200


def test_bootstrap_is_reproducible() -> None:
    y, proba = _sample()
    kwargs = {
        "threshold": 0.5,
        "cost_fn": 10.0,
        "cost_fp": 1.0,
        "n_resamples": 100,
        "random_state": 3,
    }

    assert bootstrap_cost_ci(y, proba, **kwargs) == bootstrap_cost_ci(y, proba, **kwargs)
