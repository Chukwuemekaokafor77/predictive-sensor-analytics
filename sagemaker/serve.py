from __future__ import annotations

import json
from typing import Any

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from sagemaker.inference import load_artifacts, predict_from_payload


app = FastAPI(title="psa-sagemaker-inference")

_artifacts: dict[str, Any] | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _artifacts
    _artifacts = load_artifacts("/opt/ml/model")


@app.get("/ping")
async def ping() -> dict:
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(req: Request) -> Response:
    global _artifacts
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        body = await req.body()
        payload = json.loads(body.decode("utf-8")) if body else {}
        result = predict_from_payload(_artifacts, payload)
        return Response(content=orjson.dumps(result), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
