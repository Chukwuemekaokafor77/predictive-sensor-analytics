# Predictive Sensor Analytics & Anomaly Detection (Manufacturing)

End-to-end ML system for high-frequency manufacturing sensor streams (pressure/force/acceleration/temperature):

- Signal processing: filtering, resampling, sliding windows
- Feature extraction: RMS, kurtosis, spectral energy, zero-crossing rate, moving average, peak-to-peak
- Models: Isolation Forest (baseline), Autoencoder, LSTM reconstruction
- Serving: FastAPI `/predict` + caching + persistence (PostgreSQL)
- Cloud hooks: AWS S3 + SageMaker Runtime invoker adapters

## Architecture (high-level)

```
            +-------------------+
            |  Sensor Streams   |
            | (10ms sampling)   |
            +---------+---------+
                      |
                      v
            +-------------------+
            | Preprocessing     |
            | - Butterworth LPF |
            | - (opt) Kalman    |
            | - Resample 50ms   |
            +---------+---------+
                      |
                      v
            +-------------------+
            | Windowing         |
            | - 1s windows      |
            | - 250ms stride    |
            +---------+---------+
                      |
                      v
            +-------------------+
            | Feature Extract   |
            | - RMS/Kurtosis    |
            | - Spectral energy |
            | - ZCR/MAvg/PTP    |
            +---------+---------+
                      |
                      v
            +-------------------+             +--------------------+
            | Anomaly Model     |             | PostgreSQL         |
            | - IsolationForest |<----------->| inference_results  |
            | - AE / LSTM (opt) |             +--------------------+
            +---------+---------+
                      |
                      v
            +-------------------+
            | FastAPI Service   |
            | POST /predict     |
            +---------+---------+
                      |
                      v
            +-------------------+
            | Client / MES / UI |
            +-------------------+

(Optional AWS)

  +--------+        +------------------+        +------------------+
  |  S3    |<------>| SageMaker Train  |------->| SageMaker Endpoint|
  +--------+        +------------------+        +------------------+
                              ^                          |
                              |                          v
                         model.tar.gz              FastAPI/Lambda wrapper
```

## Local run (one command)

First-time setup (so `psa` is importable from the `src/` layout):

```bash
pip install -e .
```

### API only (no Postgres)

From the project root:

```bash
python -m uvicorn psa.api.main:app --reload --port 8000
```

Then:

- `GET http://localhost:8000/health`

## Docker run (API + Postgres)

1) Ensure Docker Desktop is running.
2) `.env` must exist (this repo includes one; you can edit it if needed).

```bash
docker compose up --build
```

## Generate sample payload

This generates a realistic 10-second batch (10ms sampling) and prints JSON to stdout:

```bash
python -m psa.scripts.simulate_sensor_data > payload.json
```

## Call `/predict` (curl)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data-binary @payload.json
```

## Output

The API returns:

- `anomaly_score`: 0..1 (normalized)
- `is_anomaly`: boolean
- `details`: per-window scores + threshold + a sample feature vector

## SageMaker-ready container layout

A SageMaker-compatible training + inference layout is provided in `sagemaker/`.

### Model artifact format (what gets deployed)

During training, write your model artifacts to:

- `/opt/ml/model/`

SageMaker will package that folder as:

- `model.tar.gz`

This project writes:

- `/opt/ml/model/iforest.joblib` (joblib dump containing the fitted model + threshold)

### Training entrypoint

- `sagemaker/train.py`

### Inference entrypoint

- `sagemaker/serve.py`

It serves SageMaker-standard endpoints:

- `GET /ping`
- `POST /invocations`

The inference payload is compatible with the same request structure used by this repo’s FastAPI service:

```json
{ "sensor_batch": [ {"ts_ms": 0, "pressure": 1.0, ...}, ... ] }
```
