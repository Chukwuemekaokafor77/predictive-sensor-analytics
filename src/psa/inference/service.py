from __future__ import annotations

import collections
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from cachetools import TTLCache

from psa.api.schemas import SensorSample
from psa.config import settings
from psa.db.models import Base
from psa.db.repo import insert_inference
from psa.db.session import create_engine, create_session_factory
from psa.models.autoencoder import AutoencoderAnomalyModel
from psa.models.iforest import IsolationForestAnomalyModel
from psa.models.lstm_reconstruction import LSTMReconstructionAnomalyModel
from psa.processing.pipeline import build_feature_matrix, build_sequence_tensor


class InferenceService:
    def __init__(self) -> None:
        self._cache = TTLCache(maxsize=1024, ttl=30)

        self._engine = None
        self._session_factory = None
        if settings.enable_db:
            self._engine = create_engine()
            self._session_factory = create_session_factory(self._engine)

        self._model_family = (settings.model_family or "iforest").strip().lower()
        if self._model_family not in {"iforest", "autoencoder", "lstm"}:
            self._model_family = "iforest"

        self._model_iforest = IsolationForestAnomalyModel()
        self._model_ae = AutoencoderAnomalyModel()
        self._model_lstm = LSTMReconstructionAnomalyModel()
        self._is_fitted = False

        self._feature_names: list[str] | None = None
        self._feature_cols_default = [c.strip() for c in (settings.feature_cols or "").split(",") if c.strip()]
        if not self._feature_cols_default:
            self._feature_cols_default = ["pressure", "force", "acceleration", "temperature"]

        self._recent_batch_scores: collections.deque[float] = collections.deque(maxlen=500)

    async def start(self) -> None:
        if self._engine is not None:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        await self._try_load_artifacts()

    async def stop(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()

    async def predict(self, batch: list[SensorSample]) -> dict[str, Any]:
        key = self._request_hash(batch)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        df = pd.DataFrame([s.model_dump() for s in batch])
        df = df.sort_values("ts_ms") if "ts_ms" in df.columns else df
        feature_cols = [c for c in self._feature_cols_default if c in df.columns]
        if not feature_cols:
            feature_cols = [c for c in ["pressure", "force", "acceleration", "temperature"] if c in df.columns]

        X = np.zeros((0, 0), dtype=float)
        rows: list[dict[str, float]] = []
        meta: list[dict[str, int]] = []
        feature_names: list[str] = []

        if self._model_family in {"iforest", "autoencoder"}:
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
            X, rows, meta, feature_names = out
        else:
            # LSTM expects per-window sequences: (n_windows, seq_len, n_features)
            out_seq = build_sequence_tensor(
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
            X_seq, meta, _seq_cols = out_seq
            # Keep `X` empty in this branch; scoring uses `X_seq`
            X = np.zeros((int(X_seq.shape[0]), 0), dtype=float)
            feature_names = []

        if X.shape[0] == 0:
            result = {
                "model": self._active_model_name(),
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "details": {"reason": "insufficient_data", "windows": 0},
            }
            self._cache[key] = result
            return result

        if self._feature_names is None and feature_names:
            self._feature_names = feature_names

        if self._feature_names is not None and feature_names and self._feature_names != feature_names:
            raise ValueError("Feature names mismatch (expected training feature_names)")

        self._ensure_fitted(n_features=int(X.shape[1]))

        if self._model_family == "iforest":
            scores = self._model_iforest.score(X)
            threshold = float(self._model_iforest.threshold)
        elif self._model_family == "autoencoder":
            scores = self._model_ae.score(X)
            threshold = float(self._model_ae.threshold)
        else:
            # Recompute for LSTM branch (X is placeholder)
            out_seq = build_sequence_tensor(
                df,
                source_sampling_ms=10,
                target_sampling_ms=50,
                window_ms=1000,
                step_ms=250,
                feature_cols=feature_cols,
                use_butterworth=True,
                use_kalman=False,
                return_window_meta=False,
            )
            X_seq, _ = out_seq
            scores = self._model_lstm.score(X_seq)
            threshold = float(self._model_lstm.threshold)

        top_k = min(3, int(scores.size))
        topk_mean = float(np.mean(np.sort(scores)[-top_k:])) if top_k > 0 else 0.0
        p95 = float(np.percentile(scores, 95))
        batch_score = topk_mean

        last_m = min(10, int(scores.size))
        n_over = int(np.sum(scores[-last_m:] >= float(threshold))) if last_m > 0 else 0
        is_anomaly = bool((batch_score >= float(threshold)) and (n_over >= 2))

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
            "model": self._active_model_name(),
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
                "threshold": float(threshold),
                "feature_names": self._feature_names or feature_names,
                "drift": drift_stats,
            },
        }

        if self._session_factory is not None:
            async with self._session_factory() as session:
                await insert_inference(
                    session,
                    model=self._active_model_name(),
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
        if self._model_family in {"iforest", "autoencoder"}:
            n_windows = 800
            base = rng.normal(0.0, 1.0, size=(n_windows, n_features))
            drift = rng.normal(0.0, 0.05, size=(n_windows, 1))
            X_train = base + drift
            if self._model_family == "iforest":
                self._model_iforest.fit(X_train)
            else:
                self._model_ae.fit(X_train)
        else:
            # LSTM expects sequences: (n_windows, seq_len, n_features)
            n_windows = 400
            seq_len = 20
            X_seq_train = rng.normal(0.0, 1.0, size=(n_windows, seq_len, max(1, int(len(self._feature_cols_default)))))
            self._model_lstm.fit(X_seq_train)

        self._is_fitted = True

    def _active_model_name(self) -> str:
        if self._model_family == "iforest":
            return self._model_iforest.name
        if self._model_family == "autoencoder":
            return self._model_ae.name
        return self._model_lstm.name

    async def _try_load_artifacts(self) -> None:
        path = settings.model_artifact_path
        if not path:
            return

        lower = str(path).lower()
        if lower.endswith(".pt") or lower.endswith(".pth"):
            import torch

            artifacts = torch.load(path, map_location="cpu")
        else:
            # Keep this as a lazy optional dependency; joblib is already in requirements.
            import joblib

            artifacts = joblib.load(path)

        fam = str(artifacts.get("model_family") or artifacts.get("model") or "").strip().lower()
        if fam in {"iforest", "isolation_forest"}:
            self._model_family = "iforest"
            m = artifacts.get("model")
            if isinstance(m, IsolationForestAnomalyModel):
                self._model_iforest = m
            elif m is not None and hasattr(m, "score_samples"):
                # allow storing the raw sklearn model
                self._model_iforest._model = m
            if "threshold" in artifacts:
                self._model_iforest.threshold = float(artifacts["threshold"])
            if "score_mean" in artifacts:
                self._model_iforest.score_mean = float(artifacts["score_mean"])
            if "score_std" in artifacts:
                self._model_iforest.score_std = float(artifacts["score_std"])
            self._feature_names = list(artifacts.get("feature_names") or []) or None
            self._is_fitted = True
        elif fam in {"autoencoder"}:
            self._model_family = "autoencoder"
            m = artifacts.get("model")
            if isinstance(m, AutoencoderAnomalyModel):
                self._model_ae = m
            elif "state_dict" in artifacts:
                # torch artifact
                from psa.models.autoencoder import TabularAutoencoder

                n_features = int(artifacts.get("n_features") or 0)
                if n_features <= 0:
                    raise ValueError("Invalid autoencoder artifact: missing n_features")
                self._model_ae.model = TabularAutoencoder(n_features).to(self._model_ae.device)
                self._model_ae.model.load_state_dict(artifacts["state_dict"])
            if "threshold" in artifacts:
                self._model_ae.threshold = float(artifacts["threshold"])
            self._feature_names = list(artifacts.get("feature_names") or []) or None
            self._is_fitted = True
        elif fam in {"lstm", "lstm_reconstruction"}:
            self._model_family = "lstm"
            m = artifacts.get("model")
            if isinstance(m, LSTMReconstructionAnomalyModel):
                self._model_lstm = m
            elif "state_dict" in artifacts:
                from psa.models.lstm_reconstruction import LSTMAutoencoder

                n_features = int(artifacts.get("n_features") or 0)
                if n_features <= 0:
                    raise ValueError("Invalid LSTM artifact: missing n_features")
                self._model_lstm.model = LSTMAutoencoder(n_features).to(self._model_lstm.device)
                self._model_lstm.model.load_state_dict(artifacts["state_dict"])
            if "threshold" in artifacts:
                self._model_lstm.threshold = float(artifacts["threshold"])
            # LSTM works on raw sensor sequences, so feature_names are not required.
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
