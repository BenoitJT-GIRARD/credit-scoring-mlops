"""Convert the LightGBM estimator to ONNX and time it against the native pipeline.

Only the estimator converts: the preprocessing stays in Python, so this measures the part
that could move and not an end-to-end port. A conversion failure is recorded as such rather
than raising -- an optimisation that did not work out is information.
"""

from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credexp.utils import HOLDOUT_PATH, PERFORMANCE_DIR, PIPELINE_PATH, ROOT_DIR, VAR_DIR

OUT_PATH = PERFORMANCE_DIR / "onnx_benchmark.json"

#: The converted model. It is a build output, not a deliverable: `skl2onnx` writes it from
#: the pipeline that is tracked, and the benchmark is what it exists for.
ONNX_PATH = VAR_DIR / "models" / "lightgbm_model.onnx"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def benchmark_native_pipeline(pipe: Any, X: pd.DataFrame) -> dict[str, float]:
    start = time.perf_counter()
    proba = pipe.predict_proba(X)[:, 1]
    total_ms = (time.perf_counter() - start) * 1000

    return {
        "native_pipeline_total_ms": float(total_ms),
        "native_pipeline_ms_per_row": float(total_ms / len(X)),
        "native_pipeline_mean_proba": float(np.mean(proba)),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "status": "started",
        # Relative to the repository root: an absolute path pins the committed report to
        # whoever happened to run it.
        "pipeline_path": PIPELINE_PATH.relative_to(ROOT_DIR).as_posix(),
        "holdout_path": HOLDOUT_PATH.relative_to(ROOT_DIR).as_posix(),
        "onnx_model_path": ONNX_PATH.relative_to(ROOT_DIR).as_posix(),
    }

    try:
        pipe = joblib.load(PIPELINE_PATH)
        df = pd.read_parquet(HOLDOUT_PATH).head(500)
        X = df.drop(columns=["TARGET"], errors="ignore")

        result["n_rows"] = int(len(X))
        result.update(benchmark_native_pipeline(pipe, X))

        try:
            import onnxruntime as ort
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType

            # In a sklearn Pipeline the preprocessing is every step but the last,
            # and the estimator is the last one -- here a LGBMClassifier.
            preprocessor = pipe[:-1]
            estimator = pipe[-1]

            Xt = preprocessor.transform(X)
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()

            Xt = np.asarray(Xt, dtype=np.float32)

            initial_types = [("input", FloatTensorType([None, Xt.shape[1]]))]
            onnx_model = convert_lightgbm(estimator, initial_types=initial_types)

            ONNX_PATH.write_bytes(onnx_model.SerializeToString())

            session = ort.InferenceSession(
                str(ONNX_PATH),
                providers=["CPUExecutionProvider"],
            )
            input_name = session.get_inputs()[0].name

            # Warmup
            session.run(None, {input_name: Xt[:10]})

            start = time.perf_counter()
            outputs = session.run(None, {input_name: Xt})
            onnx_total_ms = (time.perf_counter() - start) * 1000

            result.update(
                {
                    "status": "success",
                    "onnx_total_ms": float(onnx_total_ms),
                    "onnx_ms_per_row": float(onnx_total_ms / len(Xt)),
                    "native_vs_onnx_speedup": float(
                        result["native_pipeline_total_ms"] / onnx_total_ms
                    )
                    if onnx_total_ms > 0
                    else None,
                    "onnx_outputs_count": int(len(outputs)),
                    "interpretation": (
                        "ONNX Runtime inference was successfully benchmarked on the "
                        "LightGBM estimator after Python preprocessing."
                    ),
                }
            )

        except Exception as exc:
            result.update(
                {
                    "status": "onnx_failed",
                    "onnx_error": repr(exc),
                    "interpretation": (
                        "ONNX conversion/runtime benchmark was attempted but failed. "
                        "This is acceptable for this project because the full pipeline "
                        "contains sklearn preprocessing plus LightGBM, which is not always "
                        "straightforward to convert end-to-end. The retained production "
                        "optimization is API batching, already benchmarked separately."
                    ),
                }
            )

    except Exception as exc:
        result.update(
            {
                "status": "fatal_error",
                "error": repr(exc),
            }
        )

    OUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=to_jsonable),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=to_jsonable))


if __name__ == "__main__":
    main()
