from pydantic import BaseModel, Field


class SensorSample(BaseModel):
    ts_ms: int = Field(..., description="Timestamp in milliseconds")
    pressure: float | None = None
    force: float | None = None
    acceleration: float | None = None
    temperature: float | None = None


class PredictRequest(BaseModel):
    sensor_batch: list[SensorSample]


class PredictResponse(BaseModel):
    model: str
    anomaly_score: float
    is_anomaly: bool
    details: dict
