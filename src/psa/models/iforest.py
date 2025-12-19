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
        self.threshold = 0.65

    def fit(self, X: np.ndarray) -> None:
        if X.size == 0:
            return
        self._model.fit(X)

        scores = self.score(X)
        if scores.size:
            self.threshold = float(np.quantile(scores, 0.99))

    def score(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0,), dtype=float)
        raw = -self._model.score_samples(X)
        raw = (raw - float(np.min(raw))) / (float(np.ptp(raw)) + 1e-9)
        return raw
