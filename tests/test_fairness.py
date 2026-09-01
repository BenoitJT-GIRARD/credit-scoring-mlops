import numpy as np

from credexp.modeling.fairness import age_bands, group_report


def test_age_bands_label_by_decade_from_negative_days() -> None:
    # Home Credit stores age as DAYS_BIRTH, negative and counted backwards from today.
    days = np.array([-25 * 365, -35 * 365, -45 * 365, -55 * 365, -70 * 365])

    assert list(age_bands(days)) == ["<30", "30-39", "40-49", "50-59", "60+"]


def test_group_report_returns_one_row_per_group_with_its_size() -> None:
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.1, 0.9, 0.8, 0.2])
    groups = np.array(["F", "F", "M", "M"])

    rows = group_report(y, proba, groups, threshold=0.5, cost_fn=10.0, cost_fp=1.0)

    assert {r["group"] for r in rows} == {"F", "M"}
    assert all(r["n"] == 2 for r in rows)


def test_refusal_rate_counts_predictions_above_the_threshold() -> None:
    y = np.array([0, 0, 0, 0])
    proba = np.array([0.9, 0.9, 0.1, 0.1])
    groups = np.array(["A", "A", "B", "B"])

    rows = {r["group"]: r for r in group_report(y, proba, groups, 0.5, 10.0, 1.0)}

    assert rows["A"]["refusal_rate"] == 1.0
    assert rows["B"]["refusal_rate"] == 0.0


def test_a_group_without_positives_reports_no_false_negative_rate() -> None:
    # A rate over an empty denominator is not zero, it is unknown. Reporting 0.0 would
    # read as perfect performance on a group the model was never tested on.
    y = np.array([0, 0])
    proba = np.array([0.1, 0.9])
    groups = np.array(["X", "X"])

    rows = group_report(y, proba, groups, 0.5, 10.0, 1.0)

    assert rows[0]["fnr"] is None
