"""Runtime context for the backend process.

This module intentionally holds data only.  Bootstrapping lives in
``bootstrapper.py``; domain behavior stays on ``Observatory`` and the ledgers.
"""

from dataclasses import dataclass, field
from typing import Any

from config import Config


@dataclass
class ApplicationContext:
    """Container for runtime objects wired during startup."""

    config: Config | dict[str, Any] | None = None
    observatory: Any = None
    city: Any = None
    cache_strategy: Any = None
    persistence_gateway: Any = None
    geocoding_service: Any = None
    traffic_service: Any = None
    feed_fetcher: Any = None
    stop_event: Any = None
    shutdown_event: Any = None
    threads: dict[str, Any] = field(default_factory=dict)
    thread_loader: Any = None
    state_interface: Any = None
    predictor: Any = None
    bus_type_predictor: Any = None
    validation_controller: Any = None

    def __post_init__(self):
        """Normalize callers onto the typed backend configuration object."""
        self.config = Config.coerce(self.config)
