"""Internal application events and pub/sub infrastructure."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type

from .interfaces import EventBus


class InternalEvent(ABC):
    """Base class for events exchanged through the internal event bus."""

    event_name: ClassVar[str]

    @property
    def name(self) -> str:
        """Return the event name used by the bus."""
        return self.event_name

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return the payload delivered to subscribers."""
        ...


@dataclass(frozen=True)
class LiveTripFinishedEvent(InternalEvent):
    """A live trip has ended and its measurements are ready for downstream use."""

    event_name: ClassVar[str] = "live_trip_finished"

    live_trip: Any
    route_id: str
    vehicle_type_name: str = "Unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "live_trip": self.live_trip,
            "route_id": self.route_id,
            "vehicle_type_name": self.vehicle_type_name,
        }


@dataclass(frozen=True)
class ShutdownRequestedEvent(InternalEvent):
    """The console/user requested process shutdown."""

    event_name: ClassVar[str] = "shutdown_requested"

    def as_dict(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class ServicesStartEvent(InternalEvent):
    """The console/user requested collection service startup."""

    event_name: ClassVar[str] = "services_start"

    def as_dict(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class ServicesStopEvent(InternalEvent):
    """The console/user requested collection service shutdown."""

    event_name: ClassVar[str] = "services_stop"

    def as_dict(self) -> dict[str, Any]:
        return {}


LIVE_TRIP_FINISHED = LiveTripFinishedEvent.event_name
SHUTDOWN_REQUESTED = ShutdownRequestedEvent.event_name
SERVICES_START = ServicesStartEvent.event_name
SERVICES_STOP = ServicesStopEvent.event_name


EventKey = str | InternalEvent | Type[InternalEvent]


class InternalEventBus(EventBus):
    """Synchronous pub/sub bus for named and typed internal events."""

    def __init__(self, label: str = "event"):
        """Initialize with an empty subscriber map."""
        self.label = label
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event: EventKey, handler: Callable):
        """Subscribe a handler to a string event name or event class."""
        event_name = self._event_name(event)
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)

    def unsubscribe(self, event: EventKey, handler: Callable):
        """Remove a handler from an event."""
        event_name = self._event_name(event)
        if event_name in self._subscribers and handler in self._subscribers[event_name]:
            self._subscribers[event_name].remove(handler)

    def emit(self, event: EventKey, data: Optional[dict] = None):
        """Emit an event. Handlers receive a payload dictionary."""
        event_name, payload = self._event_payload(event, data)
        if event_name not in self._subscribers:
            return

        for handler in list(self._subscribers[event_name]):
            try:
                handler(payload)
            except Exception as e:
                logging.error(
                    "Error in %s handler for '%s': %s",
                    self.label,
                    event_name,
                    e,
                )

    def _event_payload(
        self,
        event: EventKey,
        data: Optional[dict],
    ) -> tuple[str, dict[str, Any]]:
        if isinstance(event, InternalEvent):
            return event.name, event.as_dict()
        return self._event_name(event), data or {}

    def _event_name(self, event: EventKey) -> str:
        if isinstance(event, str):
            return event
        if isinstance(event, InternalEvent):
            return event.name
        if isinstance(event, type) and issubclass(event, InternalEvent):
            return event.event_name
        raise TypeError(f"Unsupported event key: {event!r}")


class DomainEventBus(InternalEventBus):
    """Event bus for domain-initiated events."""

    def __init__(self):
        super().__init__(label="domain event")


class ConsoleEventBus(InternalEventBus):
    """Event bus for console/user-initiated events."""

    def __init__(self):
        super().__init__(label="console event")


domain_events = DomainEventBus()
console_events = ConsoleEventBus()

__all__ = [
    "ConsoleEventBus",
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
    "console_events",
    "domain_events",
]
