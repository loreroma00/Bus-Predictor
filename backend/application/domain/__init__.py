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
from .ledgers import (
    TopologyLedger,
    ScheduleLedger,
    HistoricalLedger,
    PredictedLedger,
    VehicleLedger,
    MeasurementRecord,
    StopArrival,           # backward-compat alias for MeasurementRecord
    StopPredictionRecord,
    VehicleTripRecord,
    extract_measurements_from_diary,
    project_diary_to_stops,  # backward-compat wrapper
    summarize_diary_for_vehicle,
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
    # Ledgers
    "TopologyLedger",
    "ScheduleLedger",
    "HistoricalLedger",
    "PredictedLedger",
    "VehicleLedger",
    "MeasurementRecord",
    "StopArrival",
    "StopPredictionRecord",
    "VehicleTripRecord",
    "extract_measurements_from_diary",
    "project_diary_to_stops",
    "summarize_diary_for_vehicle",
    # Events
    "domain_events",
    "DIARY_FINISHED",
]
