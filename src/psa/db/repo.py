from sqlalchemy.ext.asyncio import AsyncSession

from psa.db.models import InferenceResult


async def insert_inference(
    session: AsyncSession,
    *,
    model: str,
    anomaly_score: float,
    is_anomaly: bool,
    request_hash: str,
    details: dict,
) -> None:
    session.add(
        InferenceResult(
            model=model,
            anomaly_score=float(anomaly_score),
            is_anomaly=bool(is_anomaly),
            request_hash=request_hash,
            details=details,
        )
    )
    await session.commit()
