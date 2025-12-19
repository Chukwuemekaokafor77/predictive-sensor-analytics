from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def butterworth_lowpass(x: np.ndarray, *, fs_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if x.size == 0:
        return x
    nyq = 0.5 * fs_hz
    norm = cutoff_hz / nyq
    b, a = signal.butter(order, norm, btype="low", analog=False)
    
    # SciPy's filtfilt requires a minimum length (padlen). For short windows,
    # skip filtering rather than failing.
    min_len = 3 * max(len(a), len(b))
    if x.size <= min_len:
        return x
    return signal.filtfilt(b, a, x)


def simple_kalman_1d(z: np.ndarray, *, q: float = 1e-5, r: float = 1e-2) -> np.ndarray:
    if z.size == 0:
        return z
    x_hat = np.zeros_like(z, dtype=float)
    p = 1.0
    x = float(z[0])

    for i, zi in enumerate(z):
        p = p + q
        k = p / (p + r)
        x = x + k * (float(zi) - x)
        p = (1 - k) * p
        x_hat[i] = x

    return x_hat


def resample_df(df: pd.DataFrame, *, target_ms: int) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts_ms"], unit="ms", utc=True)
    out = out.set_index("ts").drop(columns=["ts_ms"]).sort_index()

    rule = f"{target_ms}ms"
    out = out.resample(rule).mean().interpolate("time")

    out["ts_ms"] = (out.index.view("int64") // 1_000_000).astype("int64")
    out = out.reset_index(drop=True)
    return out


def sliding_windows(
    df: pd.DataFrame,
    *,
    window_ms: int,
    step_ms: int,
    target_ms: int,
    feature_cols: list[str],
) -> list[pd.DataFrame]:
    if df.empty:
        return []

    samples_per_window = max(1, int(round(window_ms / target_ms)))
    samples_per_step = max(1, int(round(step_ms / target_ms)))

    windows: list[pd.DataFrame] = []
    n = len(df)
    for start in range(0, n - samples_per_window + 1, samples_per_step):
        w = df.iloc[start : start + samples_per_window]
        windows.append(w[["ts_ms", *feature_cols]].reset_index(drop=True))

    return windows


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    m = float(np.mean(x))
    v = float(np.var(x))
    if v == 0.0:
        return 0.0
    return float(np.mean(((x - m) ** 4)) / (v**2))


def _zero_crossing_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1])))


def _spectral_energy(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    spec = np.fft.rfft(x)
    return float(np.sum(np.abs(spec) ** 2) / x.size)


def extract_features(window_df: pd.DataFrame, *, feature_cols: list[str]) -> dict[str, float]:
    feats: dict[str, float] = {}
    for col in feature_cols:
        x = window_df[col].to_numpy(dtype=float)
        feats[f"{col}_rms"] = _rms(x)
        feats[f"{col}_kurtosis"] = _kurtosis(x)
        feats[f"{col}_zcr"] = _zero_crossing_rate(x)
        feats[f"{col}_spectral_energy"] = _spectral_energy(x)
        feats[f"{col}_moving_avg"] = float(np.mean(x)) if x.size else 0.0
        feats[f"{col}_ptp"] = float(np.ptp(x)) if x.size else 0.0
    return feats


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    source_sampling_ms: int,
    target_sampling_ms: int,
    window_ms: int,
    step_ms: int,
    feature_cols: list[str],
    use_butterworth: bool = True,
    butter_cutoff_hz: float = 30.0,
    use_kalman: bool = False,
    return_window_meta: bool = False,
) -> tuple[np.ndarray, list[dict[str, float]]] | tuple[np.ndarray, list[dict[str, float]], list[dict[str, int]]]:
    if df.empty:
        return np.zeros((0, 0), dtype=float), []

    proc = df.copy()
    for col in feature_cols:
        if col not in proc.columns:
            proc[col] = np.nan
        proc[col] = proc[col].astype(float).interpolate().bfill().ffill()

    if use_butterworth:
        fs_hz = 1000.0 / float(source_sampling_ms)
        for col in feature_cols:
            proc[col] = butterworth_lowpass(proc[col].to_numpy(dtype=float), fs_hz=fs_hz, cutoff_hz=butter_cutoff_hz)

    if use_kalman:
        for col in feature_cols:
            proc[col] = simple_kalman_1d(proc[col].to_numpy(dtype=float))

    proc = resample_df(proc, target_ms=target_sampling_ms)

    windows = sliding_windows(
        proc,
        window_ms=window_ms,
        step_ms=step_ms,
        target_ms=target_sampling_ms,
        feature_cols=feature_cols,
    )

    rows: list[dict[str, float]] = [extract_features(w, feature_cols=feature_cols) for w in windows]
    if not rows:
        return np.zeros((0, 0), dtype=float), []

    keys = sorted(rows[0].keys())
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)

    if not return_window_meta:
        return X, rows

    meta: list[dict[str, int]] = []
    for w in windows:
        ts = w["ts_ms"].to_numpy(dtype="int64")
        if ts.size == 0:
            meta.append({"start_ts_ms": 0, "end_ts_ms": 0})
        else:
            meta.append({"start_ts_ms": int(ts[0]), "end_ts_ms": int(ts[-1])})

    return X, rows, meta
