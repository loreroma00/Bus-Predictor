"""Concrete persistence facade.

This adapter intentionally contains no persistence logic.  It delegates to the
existing persistence modules while presenting one application-facing object.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import Config


class PersistenceFacade:
    """Thin adapter over the persistence layer."""

    def __init__(self, config: Config | None = None):
        """Store the runtime config used for default persistence destinations."""
        self.config = Config.coerce(config)

    def configure(self, config: Config) -> None:
        """Replace the runtime config used by this facade."""
        self.config = Config.coerce(config)

    def ensure_database(self, config: Config | None = None) -> Any:
        from .database import get_db_connection

        if config is not None:
            self.configure(config)
        return get_db_connection(
            connection_string=self.config.ledger.db_connection,
            table_name=self.config.ledger.historical_table,
        )

    def shutdown_database(self) -> None:
        from .database import shutdown_database

        shutdown_database()

    def save_completed_measurements(
        self,
        measurements: list[dict],
        filename: str | None = None,
    ) -> None:
        from .measurements import save_measurements_incremental

        save_measurements_incremental(
            measurements,
            filename=filename,
            measurements_path=self.config.paths.measurements_path,
            measurements_file=self.config.paths.measurements_file,
        )

    def write_historical_records(
        self,
        connection_string: str,
        table_name: str,
        records: list[dict],
    ) -> None:
        from .database import write_historical

        if not connection_string or not table_name:
            return
        write_historical(connection_string, table_name, records)

    def read_historical_records(
        self,
        connection_string: str,
        table_name: str,
        trip_id: str | None = None,
        route_id: str | None = None,
        date_start: float | None = None,
        date_end: float | None = None,
    ) -> pd.DataFrame:
        from .database import read_historical

        if not connection_string or not table_name:
            return pd.DataFrame()
        return read_historical(
            connection_string,
            table_name,
            trip_id=trip_id,
            route_id=route_id,
            date_start=date_start,
            date_end=date_end,
        )

    def write_prediction_records(
        self,
        connection_string: str,
        table_name: str,
        predictions: list[dict],
    ) -> None:
        from .database import write_predicted

        if not connection_string or not table_name:
            return
        write_predicted(connection_string, table_name, predictions)

    def read_prediction_records(
        self,
        connection_string: str,
        table_name: str,
        route_id: str | None = None,
        direction_id: int | None = None,
        trip_date: str | None = None,
        scheduled_start: str | None = None,
    ) -> pd.DataFrame:
        from .database import read_predicted

        if not connection_string or not table_name:
            return pd.DataFrame()
        return read_predicted(
            connection_string,
            table_name,
            route_id=route_id,
            direction_id=direction_id,
            trip_date=trip_date,
            scheduled_start=scheduled_start,
        )

    def write_vehicle_trip_records(
        self,
        connection_string: str,
        table_name: str,
        records: list[dict],
    ) -> None:
        from .database import write_vehicle_trips

        if not connection_string or not table_name:
            return
        write_vehicle_trips(connection_string, table_name, records)

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
        from .database import read_vehicle_trips

        if not connection_string or not table_name:
            return pd.DataFrame()
        return read_vehicle_trips(
            connection_string,
            table_name,
            vehicle_id=vehicle_id,
            route_id=route_id,
            fuel_type=fuel_type,
            date_start=date_start,
            date_end=date_end,
        )

    async def fetch_validation_rows_by_trip_ids(
        self,
        trip_ids: set[str],
    ) -> dict[str, list[Any]]:
        from .database import fetch_validation_rows_by_trip_ids

        return await fetch_validation_rows_by_trip_ids(
            trip_ids,
            connection_string=self.config.ledger.db_connection,
            table_name=self.config.ledger.historical_table,
        )

    def read_historical_training_rows(
        self,
        start_date: str | None = None,
    ) -> pd.DataFrame:
        from .database import read_historical_training_rows

        return read_historical_training_rows(
            connection_string=self.config.ledger.db_connection,
            table_name=self.config.ledger.historical_table,
            start_date=start_date,
        )

    def read_historical_traffic_rows(self) -> pd.DataFrame:
        from .database import read_historical_traffic_rows

        return read_historical_traffic_rows(
            connection_string=self.config.ledger.db_connection,
            table_name=self.config.ledger.historical_table,
        )


def create_persistence_gateway(config: Config | None = None) -> PersistenceFacade:
    """Create the process persistence gateway."""
    return PersistenceFacade(config)
