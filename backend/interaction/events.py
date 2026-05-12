"""Compatibility exports for console/internal events.

Event definitions and bus implementations live in
``application.domain.internal_events``. This module remains as the stable
interaction import path for console code and tests.
"""

from application.domain.internal_events import (
    ConsoleEventBus,
    InternalEvent,
    InternalEventBus,
    ServicesStartEvent,
    ServicesStopEvent,
    ShutdownRequestedEvent,
    SERVICES_START,
    SERVICES_STOP,
    SHUTDOWN_REQUESTED,
    console_events,
)

__all__ = [
    "ConsoleEventBus",
    "InternalEvent",
    "InternalEventBus",
    "ServicesStartEvent",
    "ServicesStopEvent",
    "ShutdownRequestedEvent",
    "SERVICES_START",
    "SERVICES_STOP",
    "SHUTDOWN_REQUESTED",
    "console_events",
]
