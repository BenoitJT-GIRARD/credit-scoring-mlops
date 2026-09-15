"""The engine and session factory, built from the configured database URL.

**Connecting has a deadline.** Writing each prediction to PostgreSQL is best-effort: the
API serves with no database behind it, and says so in three places. That promise was true
of the code and false of the wait. With nothing listening, `psycopg` retried until the
operating system gave up, and `init_db()` took a little over two minutes to fail — long
enough for a container to be declared unhealthy and restarted before it ever answered, and
long enough for a reader running `uvicorn` without Docker to think the command had hung.
`tests/system/test_service_end_to_end.py` is what found it, because a test client never
starts the application and so never runs this.

Three seconds is the whole budget. A database on the same compose network answers in
milliseconds; one that has not answered in three seconds is not going to answer this
request either.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from credexp.config import settings

#: Seconds `psycopg` may spend opening a connection before it gives up. Overridable because
#: a remote managed database on a cold start is slower than a container two hops away.
CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "3"))

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": CONNECT_TIMEOUT},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
