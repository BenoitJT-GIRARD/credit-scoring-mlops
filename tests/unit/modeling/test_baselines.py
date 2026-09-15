"""The trivial policies cost what arithmetic says they cost, and reproduce.

They are the floor the model's 0.4888 per applicant is read against. A floor computed
wrongly makes the model look better or worse than it is, and nothing else would catch it.
"""

import numpy as np

from credexp.modeling.baselines import trivial_baselines


def test_accept_all_costs_the_defaulters() -> None:
    y = np.array([0, 0, 0, 1])  # one defaulter in four

    rows = {r["name"]: r for r in trivial_baselines(y, cost_fn=10.0, cost_fp=1.0)}

    assert rows["accept-all"]["cost_per_row"] == 10.0 / 4


def test_reject_all_costs_the_good_applicants() -> None:
    y = np.array([0, 0, 0, 1])  # three good applicants refused

    rows = {r["name"]: r for r in trivial_baselines(y, cost_fn=10.0, cost_fp=1.0)}

    assert rows["reject-all"]["cost_per_row"] == 3.0 / 4


def test_the_three_baselines_are_present_and_reproducible() -> None:
    y = np.random.default_rng(0).binomial(1, 0.08, size=500)

    first = trivial_baselines(y, 10.0, 1.0, random_state=5)
    second = trivial_baselines(y, 10.0, 1.0, random_state=5)

    assert [r["name"] for r in first] == ["accept-all", "reject-all", "random-at-base-rate"]
    assert first == second
