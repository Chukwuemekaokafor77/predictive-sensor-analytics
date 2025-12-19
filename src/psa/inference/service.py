from __future__ import annotations

import collections
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from cachetools import TTLCache

from psa.api.schemas import SensorSample
from psa.db.models import Base
from psa.db.repo import insert_inference
from psa.db.session import create_engine, create_session_factory
from psa.models.iforest import IsolationForestAnomalyModel
from psa.processing.pipeline import build_feature_matrix


class InferenceService:
    def __init__(self) -> None:
        self._cache = TTLCache(maxsize=1024, ttl=30)
        self._engine = create_engine()
        self._session_factory = create_session_factory(self._engine)

        self._model = IsolationForestAnomalyModel()
        self._is_fitted = False

        self._recent_batch_scores: collections.deque[float] = collections.deque(maxlen=500)

    async def start(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def stop(self) -> None:
        await self._engine.dispose()

    async def predict(self, batch: list[SensorSample]) -> dict[str, Any]:
        key = self._request_hash(batch)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        df = pd.DataFrame([s.model_dump() for s in batch])
        feature_cols = [c for c in ["pressure", "force", "acceleration", "temperature"] if c in df.columns]

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
            result = {
                "model": self._model.name,
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "details": {"reason": "insufficient_data", "windows": 0},
            }
            self._cache[key] = result
            return result

        self._ensure_fitted(n_features=int(X.shape[1]))

        scores = self._model.score(X)

        top_k = min(3, int(scores.size))
        topk_mean = float(np.mean(np.sort(scores)[-top_k:])) if top_k > 0 else 0.0
        p95 = float(np.percentile(scores, 95))
        batch_score = topk_mean

        last_m = min(10, int(scores.size))
        n_over = int(np.sum(scores[-last_m:] >= float(self._model.threshold))) if last_m > 0 else 0
        is_anomaly = bool((batch_score >= float(self._model.threshold)) and (n_over >= 2))

        drift_stats = self._update_and_get_drift(batch_score)

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

        explain = self._explain_window(rows, int(top_idx[0]) if top_idx.size else 0)

        result = {
            "model": self._model.name,
            "anomaly_score": batch_score,
            "is_anomaly": is_anomaly,
            "details": {
                "windows": int(X.shape[0]),
                "aggregation": {
                    "type": "topk_mean",
                    "top_k": int(top_k),
                    "topk_mean": float(topk_mean),
                    "p95": float(p95),
                    "n_over_threshold_last_m": int(n_over),
                    "m": int(last_m),
                },
                "per_window_scores": scores[-10:].tolist(),
                "top_windows": top_windows,
                "feature_sample": rows[-1] if rows else {},
                "explain": explain,
                "threshold": float(self._model.threshold),
                "drift": drift_stats,
            },
        }

        async with self._session_factory() as session:
            await insert_inference(
                session,
                model=self._model.name,
                anomaly_score=batch_score,
                is_anomaly=is_anomaly,
                request_hash=key,
                details=result["details"],
            )

        self._cache[key] = result
        return result

    def _request_hash(self, batch: list[SensorSample]) -> str:
        payload = json.dumps([s.model_dump() for s in batch], separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_fitted(self, *, n_features: int) -> None:
        if self._is_fitted:
            return
        rng = np.random.default_rng(7)
        n_windows = 800
        base = rng.normal(0.0, 1.0, size=(n_windows, n_features))
        drift = rng.normal(0.0, 0.05, size=(n_windows, 1))
        X_train = base + drift
        self._model.fit(X_train)
        self._is_fitted = True

    def _update_and_get_drift(self, batch_score: float) -> dict[str, float | int]:
        prev = np.array(list(self._recent_batch_scores), dtype=float)
        prev_n = int(prev.size)
        prev_mean = float(np.mean(prev)) if prev_n else 0.0
        prev_std = float(np.std(prev)) if prev_n else 0.0

        self._recent_batch_scores.append(float(batch_score))
        cur = np.array(list(self._recent_batch_scores), dtype=float)
        cur_n = int(cur.size)
        cur_mean = float(np.mean(cur)) if cur_n else 0.0
        cur_std = float(np.std(cur)) if cur_n else 0.0

        return {
            "baseline_n": prev_n,
            "baseline_mean": prev_mean,
            "baseline_std": prev_std,
            "current_n": cur_n,
            "current_mean": cur_mean,
            "current_std": cur_std,
            "delta_mean": float(cur_mean - prev_mean) if prev_n else 0.0,
        }

    def _explain_window(self, rows: list[dict[str, float]], idx: int) -> dict[str, Any]:
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
