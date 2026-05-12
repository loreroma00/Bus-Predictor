"""Runtime context for the backend process.

This module intentionally holds data only.  Bootstrapping lives in
``bootstrapper.py``; domain behavior stays on ``Observatory`` and the ledgers.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ApplicationContext:
    """Container for runtime objects wired during startup."""

    config: dict[str, Any]
    observatory: Any
    city: Any
    cache_strategy: Any = None
    geocoding_service: Any = None
    traffic_service: Any = None
    state_interface: Any = None
    predictor: Any = None
    bus_type_predictor: Any = None

