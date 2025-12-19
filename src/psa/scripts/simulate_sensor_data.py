from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Sample:
    ts_ms: int
    pressure: float
    force: float
    acceleration: float
    temperature: float


def generate(*, seconds: int = 10, sampling_ms: int = 10, seed: int = 7, with_anomaly: bool = True) -> list[Sample]:
    rng = np.random.default_rng(seed)
    n = int(seconds * 1000 / sampling_ms)
    t = np.arange(n) * (sampling_ms / 1000.0)

    pressure = 4.0 + 0.2 * np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 0.03, size=n)
    force = 10.0 + 0.8 * np.sin(2 * np.pi * 0.6 * t + 0.4) + rng.normal(0, 0.08, size=n)
    acceleration = 0.3 * np.sin(2 * np.pi * 15 * t) + rng.normal(0, 0.04, size=n)
    temperature = 35.0 + 0.01 * t + rng.normal(0, 0.02, size=n)

    if with_anomaly:
        idx = int(n * 0.7)
        pressure[idx : idx + 10] += 1.2
        acceleration[idx : idx + 40] += 0.6 * np.sin(2 * np.pi * 50 * t[idx : idx + 40])

    out: list[Sample] = []
    start_ts = 1730000000000
    for i in range(n):
        out.append(
            Sample(
                ts_ms=start_ts + i * sampling_ms,
                pressure=float(pressure[i]),
                force=float(force[i]),
                acceleration=float(acceleration[i]),
                temperature=float(temperature[i]),
            )
        )

    return out


if __name__ == "__main__":
    samples = generate(seconds=10, sampling_ms=10, with_anomaly=True)
    payload = {"sensor_batch": [asdict(s) for s in samples]}
    print(json.dumps(payload))
