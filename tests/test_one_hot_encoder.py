import pandas as pd

from credexp.data.build_features import one_hot_encoder


def test_one_hot_encoder_creates_dummy_columns():
    df = pd.DataFrame({"A": ["x", "y", None], "B": [1, 2, 3]})
    out, new_cols = one_hot_encoder(df, nan_as_category=True)
    assert "B" in out.columns
    assert any(c.startswith("A_") for c in out.columns)
    assert len(new_cols) > 0
