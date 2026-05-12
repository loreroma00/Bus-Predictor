# Domain Package
# Re-export all public symbols for backward compatibility

from .internal_events import (
    ConsoleEventBus,
    DomainEventBus,
    InternalEvent,
    InternalEventBus,
    LIVE_TRIP_FINISHED,
    LiveTripFinishedEvent,
    SERVICES_START,
    SERVICES_STOP,
    SHUTDOWN_REQUESTED,
    ServicesStartEvent,
    ServicesStopEvent,
    ShutdownRequestedEvent,
    console_events,
    domain_events,
)
from .time_utils import to_unix_time, to_readable_time, get_seconds_since_midnight
from .static_data import (
    Route,
    Shape,
    Stop,
    Trip,
    Vehicle,
    VehicleHistoryLedger,
    VehicleTripRecord,
    summarize_live_trip_for_vehicle,
)
from .live_data import GPSData, LiveTrip, Measurement, Schedule, Update
from .virtual_entities import (
    HistoricalLedger,
    MeasurementRecord,
    Observatory,
    ScheduleLedger,
    StopArrival,
    TopologyLedger,
    extract_measurements_from_live_trip,
    project_live_trip_to_measurements,
)
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
    "Vehicle",
    # Live data
    "GPSData",
    "LiveTrip",
    "Measurement",
    "Schedule",
    "Update",
    # Virtual entities
    "Observatory",
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
    "VehicleHistoryLedger",
    "MeasurementRecord",
    "StopArrival",
    "VehicleTripRecord",
    "extract_measurements_from_live_trip",
    "project_live_trip_to_measurements",
    "summarize_live_trip_for_vehicle",
    # Events
    "ConsoleEventBus",
    "domain_events",
    "console_events",
    "DomainEventBus",
    "InternalEvent",
    "InternalEventBus",
    "LIVE_TRIP_FINISHED",
    "LiveTripFinishedEvent",
    "SERVICES_START",
    "SERVICES_STOP",
    "SHUTDOWN_REQUESTED",
    "ServicesStartEvent",
    "ServicesStopEvent",
    "ShutdownRequestedEvent",
]
