from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from psa.models.iforest import IsolationForestAnomalyModel
from psa.processing.pipeline import build_feature_matrix


def _load_training_df(training_dir: Path) -> pd.DataFrame:
    # Expected: one or more CSV files containing columns:
    # ts_ms, pressure, force, acceleration, temperature
    csvs = sorted(training_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {training_dir}")

    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    if "ts_ms" not in df.columns:
        raise ValueError("Training CSV must contain ts_ms column")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--training-dir", default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = p.parse_args()

    training_dir = Path(args.training_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = _load_training_df(training_dir)
    feature_cols = [c for c in ["pressure", "force", "acceleration", "temperature"] if c in df.columns]

    X, _ = build_feature_matrix(
        df,
        source_sampling_ms=10,
        target_sampling_ms=50,
        window_ms=1000,
        step_ms=250,
        feature_cols=feature_cols,
        use_butterworth=True,
        use_kalman=False,
    )

    if X.shape[0] == 0:
        raise RuntimeError("No windows produced from training data; check sampling/window settings")

    model = IsolationForestAnomalyModel()
    model.fit(X)

    artifact_path = model_dir / "iforest.joblib"
    joblib.dump({"model": model, "threshold": model.threshold, "feature_cols": feature_cols}, artifact_path)


if __name__ == "__main__":
    main()
