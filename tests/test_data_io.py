from __future__ import annotations

from pathlib import Path

import pandas as pd

from credexp.data.io import load_parquet, save_parquet


def test_save_and_load_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = tmp_path / "test.parquet"
    save_parquet(df, out)
    df2 = load_parquet(out)
    assert df2.shape == df.shape
    assert df2["a"].tolist() == [1, 2]
    assert df2["b"].tolist() == ["x", "y"]
