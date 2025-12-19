from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from psa.processing.pipeline import build_feature_matrix


def load_artifacts(model_dir: str = "/opt/ml/model") -> dict[str, Any]:
    path = f"{model_dir}/iforest.joblib"
    return joblib.load(path)


def predict_from_payload(artifacts: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    batch = payload.get("sensor_batch")
    if not isinstance(batch, list):
        raise ValueError("payload must contain sensor_batch as a list")

    df = pd.DataFrame(batch)
    feature_cols = artifacts.get("feature_cols") or [c for c in ["pressure", "force", "acceleration", "temperature"] if c in df.columns]

    out = build_feature_matrix(
        df,
        source_sampling_ms=10,
        target_sampling_ms=50,
        window_ms=1000,
        step_ms=250,
        feature_cols=feature_cols,
        use_butterworth=True,
        use_kalman=False,
        return_window_meta=True,
    )

    X, rows, meta = out

    if X.shape[0] == 0:
        return {
            "model": "isolation_forest",
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "details": {"reason": "insufficient_data", "windows": 0},
        }

    wrapped = artifacts["model"]
    scores = wrapped.score(X)
    threshold = float(artifacts.get("threshold", getattr(wrapped, "threshold", 0.5)))

    top_k = min(3, int(scores.size))
    topk_mean = float(np.mean(np.sort(scores)[-top_k:])) if top_k > 0 else 0.0
    anomaly_score = topk_mean

    top_idx = np.argsort(scores)[::-1][: min(5, int(scores.size))]
    top_windows = [
        {
            "rank": int(i + 1),
            "score": float(scores[idx]),
            "start_ts_ms": int(meta[idx]["start_ts_ms"]),
            "end_ts_ms": int(meta[idx]["end_ts_ms"]),
        }
        for i, idx in enumerate(top_idx.tolist())
    ]

    explain = _explain(rows, int(top_idx[0]) if top_idx.size else 0)

    return {
        "model": getattr(wrapped, "name", "isolation_forest"),
        "anomaly_score": anomaly_score,
        "is_anomaly": bool(anomaly_score >= threshold),
        "details": {
            "windows": int(X.shape[0]),
            "per_window_scores": scores[-10:].tolist(),
            "top_windows": top_windows,
            "feature_sample": rows[-1] if rows else {},
            "explain": explain,
            "threshold": threshold,
        },
    }


def _explain(rows: list[dict[str, float]], idx: int) -> dict[str, Any]:
    if not rows:
        return {"top_feature_deviations": []}

    keys = sorted(rows[0].keys())
    M = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    mu = np.mean(M, axis=0)
    sigma = np.std(M, axis=0) + 1e-9
    z = (M[idx] - mu) / sigma
    order = np.argsort(np.abs(z))[::-1][:5]

    return {
        "window_index": int(idx),
        "top_feature_deviations": [
            {"feature": keys[int(j)], "z": float(z[int(j)]), "value": float(M[idx, int(j)])} for j in order.tolist()
        ],
    }
