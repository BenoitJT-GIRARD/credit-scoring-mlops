from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except Exception as e:
    raise RuntimeError("SHAP is required for explainability. Install with: uv add shap") from e


@dataclass(frozen=True)
class ExplainConfig:
    n_background: int = 5000
    n_sample: int = 2000
    random_state: int = 42


def _safe_sample_df(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.copy()
    return df.sample(n=n, random_state=random_state).copy()


def split_X_y_from_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "TARGET" not in df.columns:
        raise ValueError("features dataframe must contain TARGET column.")
    df_train = df[df["TARGET"].notna()].copy()
    y = df_train["TARGET"].astype(int)
    X = df_train.drop(columns=["TARGET"])
    return X, y


def unpack_pipeline(pipeline: Any) -> tuple[Any, Any]:
    """
    Expects a sklearn Pipeline: preprocess/steps + final estimator.
    Returns (preprocess, model).
    """
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Expected a sklearn Pipeline with named_steps.")

    # Convention: last step is model
    model = list(pipeline.named_steps.values())[-1]
    preprocess = pipeline[:-1]  # everything before the model
    return preprocess, model


def get_feature_names(preprocess: Any) -> np.ndarray | None:
    # Many sklearn transformers support get_feature_names_out
    if hasattr(preprocess, "get_feature_names_out"):
        try:
            names = preprocess.get_feature_names_out()
            return np.asarray(names)
        except Exception:
            return None
    return None


def transform_X(preprocess: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    Xt = preprocess.transform(X)
    # sparse -> dense if needed (SHAP likes dense for some plots)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    feature_names = get_feature_names(preprocess)
    return np.asarray(Xt), feature_names


def lgbm_gain_importance(model: Any, feature_names: np.ndarray | None) -> pd.DataFrame:
    """
    Works with lightgbm.LGBMClassifier.
    """
    if not hasattr(model, "booster_"):
        raise TypeError(
            "Model does not look like a fitted LightGBM sklearn estimator (missing booster_)."
        )

    booster = model.booster_

    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    if feature_names is None:
        feature_names = np.array([f"f{i}" for i in range(len(gain))])

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": gain,
            "split": split,
        }
    ).sort_values("gain", ascending=False)

    return df


def plot_feature_importance(df_imp: pd.DataFrame, topn: int, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df_top = df_imp.head(topn).iloc[::-1]  # reverse for horizontal bar plot

    plt.figure(figsize=(10, 8))
    plt.barh(df_top["feature"], df_top["gain"])
    plt.title(f"LightGBM Feature Importance (gain) — Top {topn}")
    plt.xlabel("gain")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def compute_shap_tree(model: Any, X_background: np.ndarray, X_explain: np.ndarray):
    """
    Robust SHAP for LightGBM.

    - Use feature_perturbation="interventional" to avoid leaf coverage errors.
    - Disable additivity check (can fail due to numerical issues / approximations).
    - Prefer explainer(X) which returns an Explanation object in recent SHAP.
    """
    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation="interventional",
    )

    explanation = explainer(X_explain, check_additivity=False)

    shap_values = explanation.values
    base_values = explanation.base_values

    # base_values may be (n_samples,) -> convert to scalar for waterfall plots
    if isinstance(base_values, np.ndarray):
        expected_value = float(np.mean(base_values))
    else:
        expected_value = float(base_values)

    # For binary classification, shap_values should be (n_samples, n_features).
    # If SHAP returns (n_samples, n_features, 2), keep class 1.
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[-1] == 2:
        shap_values = shap_values[..., 1]

    return shap_values, expected_value


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: np.ndarray | None,
    outpath: Path,
    max_display: int = 30,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()


def plot_shap_waterfall(
    shap_values_row: np.ndarray,
    X_row: np.ndarray,
    expected_value: float,
    feature_names: np.ndarray | None,
    outpath: Path,
    max_display: int = 25,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # SHAP Explanation object for waterfall
    exp = shap.Explanation(
        values=shap_values_row,
        base_values=expected_value,
        data=X_row,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()


def pick_examples_by_pred_proba(
    model_pipeline: Any, X: pd.DataFrame, k: int = 1
) -> tuple[int, int]:
    """
    Returns (idx_high_risk, idx_low_risk) indices in X.
    """
    proba = model_pipeline.predict_proba(X)[:, 1]
    idx_high = int(np.argmax(proba))
    idx_low = int(np.argmin(proba))
    return idx_high, idx_low


def run_explainability(
    pipeline: Any,
    X_raw: pd.DataFrame,
    config: ExplainConfig,
    out_dir: Path,
    topn_importance: int = 30,
    max_display_shap: int = 30,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    preprocess, model = unpack_pipeline(pipeline)

    # sample background + explain set on RAW X, then transform consistently
    X_bg_raw = _safe_sample_df(X_raw, config.n_background, config.random_state)
    X_ex_raw = _safe_sample_df(X_raw, config.n_sample, config.random_state)

    X_bg, feature_names = transform_X(preprocess, X_bg_raw)
    X_ex, _ = transform_X(preprocess, X_ex_raw)

    # 1) Feature importance
    imp = lgbm_gain_importance(model, feature_names)
    plot_feature_importance(imp, topn_importance, fig_dir / "feature_importance_gain.png")

    # 2) SHAP beeswarm
    shap_values, expected_value = compute_shap_tree(model, X_bg, X_ex)
    plot_shap_beeswarm(
        shap_values,
        X_ex,
        feature_names,
        fig_dir / "shap_beeswarm.png",
        max_display=max_display_shap,
    )

    # 3) SHAP waterfall for two examples (high risk / low risk)
    idx_high, idx_low = pick_examples_by_pred_proba(pipeline, X_ex_raw, k=1)

    plot_shap_waterfall(
        shap_values_row=shap_values[idx_high],
        X_row=X_ex[idx_high],
        expected_value=float(expected_value),
        feature_names=feature_names,
        outpath=fig_dir / "shap_waterfall_high_risk.png",
        max_display=25,
    )

    plot_shap_waterfall(
        shap_values_row=shap_values[idx_low],
        X_row=X_ex[idx_low],
        expected_value=float(expected_value),
        feature_names=feature_names,
        outpath=fig_dir / "shap_waterfall_low_risk.png",
        max_display=25,
    )

    # Save metadata for traceability
    meta = {
        "n_background": config.n_background,
        "n_sample": config.n_sample,
        "random_state": config.random_state,
        "expected_value": float(expected_value),
        "idx_high_risk_in_sample": idx_high,
        "idx_low_risk_in_sample": idx_low,
        "topn_importance": topn_importance,
    }
    (out_dir / "explainability_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta
