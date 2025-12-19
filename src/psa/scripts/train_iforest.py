from __future__ import annotations

import argparse

import joblib
import numpy as np

from psa.models.iforest import IsolationForestAnomalyModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="iforest.joblib")
    p.add_argument("--n-features", type=int, default=24)
    args = p.parse_args()

    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(5000, args.n_features))

    m = IsolationForestAnomalyModel()
    m.fit(X)

    joblib.dump({"model": m, "threshold": m.threshold}, args.out)


if __name__ == "__main__":
    main()
