"""
Domain Interfaces - Defines what the domain NEEDS from outer layers.

Uses Protocol for structural subtyping — implementations don't need to
inherit from these, they just need to have matching method signatures.
"""

from typing import Protocol, runtime_checkable, Callable, Optional, Dict, List, Any


@runtime_checkable
class EventBus(Protocol):
    """
    Interface for Event Bus implementations (Pub/Sub).
    """

    def subscribe(self, event_name: Any, handler: Callable):
        """Subscribe a handler to an event."""
        ...

    def unsubscribe(self, event_name: Any, handler: Callable):
        """Unsubscribe a handler from an event."""
        ...

    def emit(self, event_name: Any, data: Optional[dict] = None):
        """Emit an event to subscribers."""
        ...


@runtime_checkable
class Pipeline(Protocol):
    """Interface for data cleaning pipelines."""

    def clean(self) -> List[Any]:
        """Run the pipeline and return cleaned domain objects."""
        ...


@runtime_checkable
class CacheStrategy(Protocol):
    """
    Interface for ledger cache operations.

    Implementations should provide load/save methods for
    TopologyLedger and ScheduleLedger.
    No inheritance required — just implement the methods.
    """

    def load_topology(self, expected_md5: str = None):
        """Load TopologyLedger from cache. Returns None if unavailable."""
        ...

    def save_topology(self, topology, source_md5: str = None) -> None:
        """Save TopologyLedger to cache."""
        ...

    def load_schedule(self, expected_md5: str = None):
        """Load ScheduleLedger from cache. Returns None if unavailable."""
        ...

    def save_schedule(self, schedule_ledger, source_md5: str = None) -> None:
        """Save ScheduleLedger to cache."""
        ...


@runtime_checkable
class GeocodingStrategy(Protocol):
    """
    Interface for geocoding operations.

    Implementations handle async geocoding with rate limiting.
    """

    def enqueue(self, lat: float, lon: float, hex_id: str) -> None:
        """Enqueue a coordinate for background geocoding."""
        ...

    def get_street(self, lat: float, lon: float) -> str | None:
        """Get cached street name, or None if not yet resolved."""
        ...

    def process_one(self) -> bool:
        """Process one item from queue. Returns False if empty."""
        ...


@runtime_checkable
class TrafficStrategy(Protocol):
    """
    Interface for traffic data operations.

    Implementations fetch traffic data and update hexagon speeds.
    """

    def update_traffic(self) -> int:
        """Update traffic for all hexagons. Returns count of updated hexagons."""
        ...
