"""Loading the feature matrix, splitting it on what the target says, and fingerprinting it.

The split is not random. Home Credit ships a train table and a test table, and the second
carries no `TARGET`; the two were concatenated during the build so that one encoding pass
covers both, and this is where they come apart again. A row whose target is missing is a
row nobody can score against, and letting one into the training set is how a model learns
from a label that does not exist.

The hash is what makes a manifest able to say which data a run saw.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credexp.modeling.dataset import load_features


@pytest.fixture()
def features(tmp_path: Path) -> Path:
    """Five applicants, three of them labelled, written as the build writes them."""
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5],
            "TARGET": [0.0, 1.0, 0.0, np.nan, np.nan],
            "EXT_SOURCE_1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "AMT_CREDIT": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    path = tmp_path / "features.parquet"
    frame.to_parquet(path)
    return path


def test_the_labelled_rows_are_the_training_set(features: Path) -> None:
    dataset = load_features(features)

    assert len(dataset.train) == 3
    assert list(dataset.y_train) == [0, 1, 0]
    assert dataset.y_train.dtype == int


def test_the_unlabelled_rows_are_the_test_set_and_have_no_target(features: Path) -> None:
    dataset = load_features(features)

    assert len(dataset.test) == 2
    assert dataset.y_test is None
    assert "TARGET" not in dataset.X_test.columns


def test_the_target_never_reaches_the_features(features: Path) -> None:
    """The leak this line prevents is the one that takes a model to an AUC of 1.0."""
    dataset = load_features(features)

    assert "TARGET" not in dataset.X_train.columns
    assert len(dataset.X_train) == len(dataset.y_train)


def test_the_fingerprint_is_the_sha256_of_the_file_on_disk(features: Path) -> None:
    """A manifest says which data a run saw, and this is the sentence it says it with."""
    dataset = load_features(features)

    assert dataset.file_hash == hashlib.sha256(features.read_bytes()).hexdigest()


def test_two_files_of_the_same_rows_hash_the_same_way(features: Path, tmp_path: Path) -> None:
    """The fingerprint is of the bytes, so a copy is recognised and a re-write is not."""
    copy = tmp_path / "copy.parquet"
    copy.write_bytes(features.read_bytes())

    assert load_features(copy).file_hash == load_features(features).file_hash
