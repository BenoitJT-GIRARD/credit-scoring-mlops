"""The two threshold-free scores, and the confusion matrix at one operating point.

They are reported together on purpose. ROC AUC on a population where 8 % default reads
generously: a model can rank well and still be useless at every threshold a lender would
accept. The average precision says what the ranking is worth where the positives are, and
the matrix says what one chosen threshold does to actual applicants.
"""

from __future__ import annotations

import numpy as np
import pytest

from credexp.modeling.metrics import evaluate_binary


def test_a_perfect_ranking_scores_one_on_both() -> None:
    truth = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])

    report = evaluate_binary(truth, proba, threshold=0.5)

    assert report.roc_auc == 1.0
    assert report.pr_auc == 1.0
    assert report.cm == [[2, 0], [0, 2]]


def test_the_matrix_reads_true_negatives_first() -> None:
    """`[[tn, fp], [fn, tp]]`, and a reader who has it the other way round inverts the cost."""
    truth = np.array([0, 0, 1, 1])
    proba = np.array([0.9, 0.1, 0.9, 0.1])

    report = evaluate_binary(truth, proba, threshold=0.5)

    (tn, fp), (fn, tp) = report.cm
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)


def test_the_threshold_moves_the_matrix_and_leaves_the_two_scores_alone() -> None:
    """This is why both are published: one describes the model, the other the decision."""
    truth = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9])

    lenient = evaluate_binary(truth, proba, threshold=0.3)
    strict = evaluate_binary(truth, proba, threshold=0.8)

    assert lenient.roc_auc == strict.roc_auc
    assert lenient.pr_auc == strict.pr_auc
    assert lenient.cm != strict.cm
    assert lenient.cm[1][1] == 2, "both defaults are caught at 0.3"
    assert strict.cm[1][1] == 1, "one is missed at 0.8"


def test_the_report_carries_the_threshold_it_was_computed_at() -> None:
    """A confusion matrix without its threshold is a table nobody can reproduce."""
    report = evaluate_binary(np.array([0, 1]), np.array([0.2, 0.7]), threshold=0.42)

    assert report.threshold == pytest.approx(0.42)


def test_an_unbalanced_population_separates_the_two_scores() -> None:
    """Eight per cent default here, and that is the case the pair exists for."""
    truth = np.array([0] * 92 + [1] * 8)
    rng = np.random.default_rng(0)
    proba = np.concatenate([rng.uniform(0.0, 0.6, 92), rng.uniform(0.4, 1.0, 8)])

    report = evaluate_binary(truth, proba, threshold=0.5)

    assert report.roc_auc > report.pr_auc
