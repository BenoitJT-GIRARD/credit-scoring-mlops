from __future__ import annotations

from credexp.db.models import Base
from credexp.db.session import engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
