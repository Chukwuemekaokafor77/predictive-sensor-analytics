from fastapi import FastAPI, HTTPException

from psa.api.schemas import PredictRequest, PredictResponse
from psa.inference.service import InferenceService


app = FastAPI(title="Predictive Sensor Analytics", version="0.1.0")

_service: InferenceService | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _service
    _service = InferenceService()
    await _service.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _service
    if _service is not None:
        await _service.stop()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    result = await _service.predict(req.sensor_batch)
    return PredictResponse(**result)
