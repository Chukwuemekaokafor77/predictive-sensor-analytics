import pandas as pd

from psa.processing.pipeline import build_feature_matrix


def test_build_feature_matrix_returns_window_meta_when_enabled() -> None:
    df = pd.DataFrame(
        {
            "ts_ms": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "pressure": [1.0] * 11,
            "force": [2.0] * 11,
            "acceleration": [0.1] * 11,
            "temperature": [30.0] * 11,
        }
    )

    out = build_feature_matrix(
        df,
        source_sampling_ms=10,
        target_sampling_ms=10,
        window_ms=50,
        step_ms=20,
        feature_cols=["pressure", "force", "acceleration", "temperature"],
        return_window_meta=True,
    )

    X, rows, meta, feature_names = out
    assert X.shape[0] == len(rows) == len(meta)
    assert all("start_ts_ms" in m and "end_ts_ms" in m for m in meta)
    assert len(feature_names) == X.shape[1]
