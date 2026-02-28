# Domain Package
# Re-export all public symbols for backward compatibility

from .internal_events import domain_events, DIARY_FINISHED
from .time_utils import to_unix_time, to_readable_time, get_seconds_since_midnight
from .static_data import Route, Trip, Stop, Shape
from .live_data import Autobus, GPSData, Schedule, Update
from .virtual_entities import Observatory
from .observers import Observer, Diary, Measurement
from .cities import City
from .verification_strategies import (
    TripVerificationStrategy,
    BasicTripVerification,
    ScaledMeasurementVerification,
)

__all__ = [
    # Time utilities
    "to_unix_time",
    "to_readable_time",
    "get_seconds_since_midnight",
    # Static data
    "Route",
    "Trip",
    "Stop",
    "Shape",
    # Live data
    "Autobus",
    "GPSData",
    "Schedule",
    "Update",
    # Virtual entities
    "Observatory",
    # Observers
    "Observer",
    "Diary",
    "Measurement",
    # Cities
    "City",
    # Verification strategies
    "TripVerificationStrategy",
    "BasicTripVerification",
    "ScaledMeasurementVerification",
    # Events
    "domain_events",
    "DIARY_FINISHED",
]
