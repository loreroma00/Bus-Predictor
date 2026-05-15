"""Compatibility shim for live validation.

Validation logic now lives in ``application.services.validator``.
"""

from application.services.validator import (
    LiveValidationSession,
    LiveValidationStatus,
    LiveValidator,
    TripValidationResult,
)

__all__ = [
    "LiveValidationSession",
    "LiveValidationStatus",
    "LiveValidator",
    "TripValidationResult",
]
