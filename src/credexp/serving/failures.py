"""One closed vocabulary for the ways a prediction can fail.

The structured log, the database row and the Prometheus counter all name the same thing,
so an incident can be traced from one to the next. Three vocabularies for one incident is
the guarantee that nothing will ever be cross-referenced.

The enumeration is closed on purpose. A free-text category is an error message again, and
you do not count messages — you count categories, and you only get a rate out of something
that repeats.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["FailureKind"]


class FailureKind(StrEnum):
    """Why a request did not produce a score."""

    #: The pipeline raised while scoring a well-formed request.
    INFERENCE_ERROR = "inference_error"
    #: No model was loaded, or the registry could not be reached and the fallback also
    #: failed. The request never reached the pipeline.
    MODEL_UNAVAILABLE = "model_unavailable"
    #: The payload validated as JSON but did not match what the model expects.
    SCHEMA_MISMATCH = "schema_mismatch"
    #: The prediction succeeded and could not be persisted. The client is unaffected;
    #: the audit trail is not.
    DATABASE_ERROR = "database_error"
