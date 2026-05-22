"""Application-facing persistence gateway contract.

Application code depends on this port instead of importing ``persistence``
modules directly.  The concrete adapter lives in the persistence layer.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from config import Config


class PersistenceGateway(Protocol):
    """Facade contract for persistence operations used by the application."""

    def ensure_database(self, config: Config | None = None) -> Any:
        """Initialize or return the default database connection."""
        ...

    def shutdown_database(self) -> None:
        """Close database resources owned by the persistence layer."""
        ...

    def save_completed_measurements(
        self,
        measurements: list[dict],
        filename: str | None = None,
    ) -> None:
        """Persist completed live-trip measurement dictionaries."""
        ...

    def write_historical_records(
        self,
        connection_string: str,
        table_name: str,
        records: list[dict],
    ) -> None:
        """Write historical measurement records."""
        ...

    def read_historical_records(
        self,
        connection_string: str,
        table_name: str,
        trip_id: str | None = None,
        route_id: str | None = None,
        date_start: float | None = None,
        date_end: float | None = None,
    ) -> pd.DataFrame:
        """Read historical measurement records."""
        ...

    def write_prediction_records(
        self,
        connection_string: str,
        table_name: str,
        predictions: list[dict],
    ) -> None:
        """Write prediction records."""
        ...

    def read_prediction_records(
        self,
        connection_string: str,
        table_name: str,
        route_id: str | None = None,
        direction_id: int | None = None,
        trip_date: str | None = None,
        scheduled_start: str | None = None,
    ) -> pd.DataFrame:
        """Read prediction records."""
        ...

    def write_vehicle_trip_records(
        self,
        connection_string: str,
        table_name: str,
        records: list[dict],
    ) -> None:
        """Write vehicle trip records."""
        ...

    def read_vehicle_trip_records(
        self,
        connection_string: str,
        table_name: str,
        vehicle_id: str | None = None,
        route_id: str | None = None,
        fuel_type: int | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> pd.DataFrame:
        """Read vehicle trip records."""
        ...

    async def fetch_validation_rows_by_trip_ids(
        self,
        trip_ids: set[str],
    ) -> dict[str, list[Any]]:
        """Fetch validation fallback rows grouped by trip id."""
        ...

    def read_historical_training_rows(
        self,
        start_date: str | None = None,
    ) -> pd.DataFrame:
        """Read historical rows used to build the prediction training dataset."""
        ...

    def read_historical_traffic_rows(self) -> pd.DataFrame:
        """Read historical rows used to compute traffic averages."""
        ...


class NoopPersistenceGateway:
    """Safe default for tests and isolated domain objects."""

    def ensure_database(self, config: Config | None = None) -> None:
        return None

    def shutdown_database(self) -> None:
        return None

    def save_completed_measurements(
        self,
        measurements: list[dict],
        filename: str | None = None,
    ) -> None:
        return None

    def write_historical_records(
        self,
        connection_string: str,
        table_name: str,
        records: list[dict],
    ) -> None:
        return None

    def read_historical_records(
        self,
        connection_string: str,
        table_name: str,
        trip_id: str | None = None,
        route_id: str | None = None,
        date_start: float | None = None,
        date_end: float | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def write_prediction_records(
        self,
        connection_string: str,
        table_name: str,
        predictions: list[dict],
    ) -> None:
        return None

    def read_prediction_records(
        self,
        connection_string: str,
        table_name: str,
        route_id: str | None = None,
        direction_id: int | None = None,
        trip_date: str | None = None,
        scheduled_start: str | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def write_vehicle_trip_records(
        self,
        connection_string: str,
        table_name: str,
        records: list[dict],
    ) -> None:
        return None

    def read_vehicle_trip_records(
        self,
        connection_string: str,
        table_name: str,
        vehicle_id: str | None = None,
        route_id: str | None = None,
        fuel_type: int | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    async def fetch_validation_rows_by_trip_ids(
        self,
        trip_ids: set[str],
    ) -> dict[str, list[Any]]:
        return {}

    def read_historical_training_rows(
        self,
        start_date: str | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def read_historical_traffic_rows(self) -> pd.DataFrame:
        return pd.DataFrame()


_default_gateway: PersistenceGateway = NoopPersistenceGateway()


def configure_persistence_gateway(
    gateway: PersistenceGateway | None,
) -> PersistenceGateway:
    """Install the process-wide persistence gateway used by default."""
    global _default_gateway
    _default_gateway = gateway or NoopPersistenceGateway()
    return _default_gateway


def get_persistence_gateway() -> PersistenceGateway:
    """Return the configured persistence gateway."""
    return _default_gateway
