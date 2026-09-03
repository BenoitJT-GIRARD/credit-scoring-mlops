"""The prediction log: one row per scored request, never updated.

The score, the decision, the threshold and the model version travel together, because a
probability without the threshold that turned it into a decision records nothing.
"""

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
    # Nullable, because a failed request has no score. Zero is a score; the absence of
    # one is not, and storing 0.0 for a failure would corrupt every rate computed from
    # this table.
    proba_default: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: One of ``FailureKind``, or null on success. Closed vocabulary, so failures can be
    #: counted rather than read.
    failure_kind: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)

    input_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    output_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
