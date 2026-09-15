"""Gain importance and SHAP, on the model that is actually served.

`get_feature_names` carries the interesting part. `get_feature_names_out` refuses as soon
as one step of the pipeline lacks it, and the `InfToNan` transformer does -- so every
published SHAP figure was labelled `Feature 32`, naming no column at all. A plot that
names nothing explains nothing. The fallback is the input column order, which is exact here
because the steps before the estimator are one-to-one, and it falls through to `None`
when the widths disagree, which is safer than a wrong label.
"""

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

from credexp.figure_style import PALETTE, apply_style, close, save_figure, sequential_cmap
from credexp.utils import FIGURES_DIR

SOURCE = "credexp.modeling.explainability"


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


def get_feature_names(preprocess: Any, X: pd.DataFrame | None = None) -> np.ndarray | None:
    """Name the columns the model actually sees.

    ``get_feature_names_out`` is the right answer when every step implements it. The
    shipped pipeline has a ``FunctionTransformer`` that does not, and the whole chain then
    refuses -- which is why every published SHAP figure was labelled "Feature 32" instead
    of naming a column. A plot that names nothing explains nothing.

    The fallback is exact, not approximate: the steps before the estimator here are
    one-to-one on columns, so the input column order *is* the output order. It only applies
    when the transformed width matches the input width, so a step that adds or drops
    columns falls through to ``None``, and the plot says nothing where it knows nothing.
    """
    if hasattr(preprocess, "get_feature_names_out"):
        try:
            return np.asarray(preprocess.get_feature_names_out())
        except Exception:  # noqa: BLE001 - any step may refuse; the fallback handles it
            pass
    if X is not None:
        return np.asarray(list(X.columns))
    return None


def transform_X(preprocess: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    Xt = preprocess.transform(X)
    # sparse -> dense if needed (SHAP likes dense for some plots)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt)
    feature_names = get_feature_names(preprocess, X)
    if feature_names is not None and len(feature_names) != Xt.shape[1]:
        feature_names = None
    return Xt, feature_names


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


def _name_the_axes(figure: Any, *, x: str, y: str) -> None:
    """Name the axes SHAP drew the plot on, and leave its colour bar alone.

    SHAP builds its own axes and names at most one of the two. It also adds a colour bar,
    which is an axes carrying data and no plot: labelling that one writes the x title of
    the figure across the bottom of the legend.
    """
    for axis in figure.axes:
        if not axis.has_data():
            continue
        # A colour bar has neither lines nor scatter points of its own.
        if not axis.lines and not axis.collections and not axis.patches:
            continue
        if not axis.get_xlabel().strip():
            axis.set_xlabel(x)
        if not axis.get_ylabel().strip():
            axis.set_ylabel(y)
        return


def plot_feature_importance(df_imp: pd.DataFrame, topn: int, outpath: Path) -> None:
    """Total gain per feature, which is what the booster split on and not what it is worth.

    Gain answers « how much did the loss drop when this column was used ». It is a property
    of the fitted trees, so a column correlated with a better one can show almost none of
    it while carrying the same information. The SHAP figures beside this one answer the
    other question, on the applicants themselves.
    """
    apply_style()
    df_top = df_imp.head(topn).iloc[::-1]

    figure, axis = plt.subplots(figsize=(9, 8))
    axis.barh(df_top["feature"], df_top["gain"], color=PALETTE["primary"])
    axis.set_title(f"Total gain per feature, top {topn}")
    axis.set_xlabel("Total gain over every split that used the feature")
    axis.set_ylabel("Feature")
    axis.tick_params(axis="y", labelsize=7)
    figure.tight_layout()
    figure.subplots_adjust(bottom=figure.subplotpars.bottom + 0.05)

    save_figure(
        figure,
        outpath,
        n={"features shown": len(df_top), "features fitted": len(df_imp)},
        source=SOURCE,
        note="gain is a property of the fitted trees, not of the applicants",
    )
    close(figure)


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

    apply_style()
    figure = plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
        # The colour is the feature's own value, low to high: a sequential scale, and the
        # project's, in place of the library's red-to-blue.
        cmap=sequential_cmap(),
    )
    # SHAP draws the figure, its colour bar and sometimes a second axes, and it names only
    # the horizontal one. The writer asks every axes holding data what it measures, so the
    # missing names are filled in here, so no reader has to infer one.
    _name_the_axes(
        figure,
        x="SHAP value: contribution to the log-odds of default",
        y="Feature, ordered by mean absolute contribution",
    )
    plt.gca().set_title("How each feature moved each applicant's score")
    figure.tight_layout()

    save_figure(
        figure,
        outpath,
        n={"applicants": int(len(X_explain)), "features shown": max_display},
        source=SOURCE,
        note="one dot per applicant and per feature; colour is the feature's own value",
    )
    close(figure)


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

    apply_style()
    figure = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    _name_the_axes(
        figure,
        x="Contribution to the log-odds of default",
        y="Feature, and this applicant's value for it",
    )
    # SHAP writes the model's average output under the axis, where an x title lands by
    # default. The pad moves the title below that annotation.
    for axis in figure.axes:
        if axis.get_xlabel():
            axis.xaxis.labelpad = 22
    figure.tight_layout()

    save_figure(
        figure,
        outpath,
        n={"applicants": 1, "features shown": max_display},
        source=SOURCE,
        note="one applicant, from the model's average output to this applicant's score",
    )
    close(figure)


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
    # The figures go where every published figure of this repository goes, so that one
    # manifest describes them all and a reader looking for a picture has one place to look.
    fig_dir = FIGURES_DIR
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
