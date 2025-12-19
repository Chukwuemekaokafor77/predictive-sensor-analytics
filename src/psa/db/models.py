from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    model = Column(String(64), nullable=False)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False)

    request_hash = Column(String(64), nullable=False, index=True)
    details = Column(JSON, nullable=False)
