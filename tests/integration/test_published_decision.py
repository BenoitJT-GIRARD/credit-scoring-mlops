"""The decision tables the README publishes, re-derived from the run that produced them.

`reports/decision/decision_analysis.json` is what the analysis wrote down;
`reports/decision/decision_summary.md` is what a reader sees. Two files, one run, and
nothing between them but `render_summary`. If the markdown is edited by hand, or the
renderer changes its rounding, or the JSON is replaced by a newer run that nobody
re-rendered, these fail — which is the only way a published table stays true to the number
behind it.

The analysis itself needs the Home Credit holdout, which is not redistributable and is not
in this repository. That is why this reads the artefacts rather than recomputing them, and
why it is here rather than in the unit tier.
"""

from __future__ import annotations

import json

import pytest

from credexp.modeling.decision_report import render_summary
from credexp.utils import DECISION_DIR

pytestmark = [pytest.mark.integration, pytest.mark.claim]

ANALYSIS = DECISION_DIR / "decision_analysis.json"
SUMMARY = DECISION_DIR / "decision_summary.md"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(ANALYSIS.read_text(encoding="utf-8"))


def test_the_published_markdown_is_what_the_renderer_produces(payload: dict) -> None:
    """Byte for byte, from the JSON beside it."""
    assert SUMMARY.read_text(encoding="utf-8") == render_summary(payload)


def test_the_shipped_threshold_of_the_analysis_is_the_one_the_service_loads(payload: dict) -> None:
    """Two files, written by two scripts, that have to agree on one number."""
    shipped = json.loads((DECISION_DIR.parent.parent / "models" / "threshold.json").read_text())

    assert payload["threshold_shift"]["shipped_threshold"] == shipped["threshold"]


def test_the_cost_interval_brackets_the_cost_it_describes(payload: dict) -> None:
    """A confidence interval that does not contain its point estimate is a typing error."""
    interval = payload["cost_interval"]
    cost = payload["threshold_shift"]["shipped_cost"]

    assert interval["low"] <= cost <= interval["high"]


def test_every_baseline_costs_more_than_the_model(payload: dict) -> None:
    """Accept everyone, refuse everyone, or flip a weighted coin: the three the README names."""
    cost = payload["threshold_shift"]["shipped_cost"]
    baselines = {baseline["name"]: baseline["cost_per_row"] for baseline in payload["baselines"]}

    assert set(baselines) == {"accept-all", "reject-all", "random-at-base-rate"}
    assert all(value > cost for value in baselines.values())


def test_the_fairness_groups_partition_the_holdout(payload: dict) -> None:
    """The age bands and the two sexes each cover the same population, counted once."""
    by_age = sum(row["n"] for row in payload["fairness_age"])
    by_gender = sum(row["n"] for row in payload["fairness_gender"])

    assert by_age == by_gender


def test_a_group_with_no_default_reports_no_false_negative_rate(payload: dict) -> None:
    """`None`, rendered as an em dash, rather than a zero that reads as « none missed »."""
    for row in payload["fairness_age"] + payload["fairness_gender"]:
        if row["default_rate"] == 0.0:
            assert row["fnr"] is None
