from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestAnomalyModel:
    name = "isolation_forest"

    def __init__(self, *, contamination: float = 0.02, random_state: int = 7) -> None:
        self._model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        self.threshold = 0.0
        self.score_mean = 0.0
        self.score_std = 1.0

    def fit(self, X: np.ndarray) -> None:
        if X.size == 0:
            return
        self._model.fit(X)

        scores = self.score_raw(X)
        if scores.size:
            self.score_mean = float(np.mean(scores))
            self.score_std = float(np.std(scores) + 1e-9)
            self.threshold = float(np.quantile(scores, 0.99))

    def score_raw(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0,), dtype=float)

        # IsolationForest.score_samples returns higher scores for inliers.
        # We invert so that higher means "more anomalous".
        return (-self._model.score_samples(X)).astype(float)

    def score(self, X: np.ndarray) -> np.ndarray:
        # Keep the public method name `score` for call sites; return stable raw scores.
        return self.score_raw(X)

    def score_z(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_raw(X)
        if scores.size == 0:
            return scores
        return (scores - float(self.score_mean)) / (float(self.score_std) + 1e-9)
