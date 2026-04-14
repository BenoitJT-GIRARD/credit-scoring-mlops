from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
        nullable=False,
    )

    request_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)

    sk_id_curr: Mapped[int | None] = mapped_column(Integer, nullable=True)

    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    proba_default: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    input_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    output_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
