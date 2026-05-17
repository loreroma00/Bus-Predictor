"""Database access layer for TimescaleDB persistence."""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
import uuid

import pandas as pd

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

from config import Ledger

# Configure logger
logger = logging.getLogger(__name__)


# Shared connection pools keyed by connection_string.
# All tables on the same server share one pool.
_shared_pools: dict[str, Any] = {}


class TimescaleDBConnection:
    """
    Async connection pool for TimescaleDB.

    Pools are shared across all instances with the same connection_string.
    Each instance manages its own write queue for a specific table.
    """

    def __init__(self, connection_string: str, table_name: str):
        """
        Initialize with specific connection details.
        """
        self.connection_string = connection_string
        self.table_name = table_name

        self.pool = None

    async def connect(self):
        """Establish or reuse a shared connection pool."""
        if not ASYNCPG_AVAILABLE:
            return False

        # Reuse existing shared pool if available
        if self.connection_string in _shared_pools:
            self.pool = _shared_pools[self.connection_string]
            logger.info(f"Reusing shared pool for {self.table_name}")
            return True

        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=10,
                command_timeout=60,
            )
            _shared_pools[self.connection_string] = self.pool
            logger.info(f"Connected to TimescaleDB (shared pool, first table: {self.table_name})")
            return True
        except Exception as e:
            logger.error(f"TimescaleDB connection failed for {self.table_name}: {e}")
            return False

    async def get_valid_pool(self):
        """Ensure the pool is valid for the current loop."""
        if self.pool:
            try:
                current_loop = asyncio.get_running_loop()
                if getattr(self.pool, "_loop", None) != current_loop:
                    # Loop changed — discard shared pool reference and
                    # terminate the old pool to prevent connection leaks.
                    old_pool = self.pool
                    _shared_pools.pop(self.connection_string, None)
                    self.pool = None
                    try:
                        old_pool.terminate()
                        logger.warning(
                            f"Terminated stale pool for {self.table_name} (loop mismatch)"
                        )
                    except Exception:
                        pass
            except Exception:
                self.pool = None

        if not self.pool:
            await self.connect()

        return self.pool

    async def close(self):
        """Close connection pool."""
        if self.pool:
            # Only close if this is the last reference and loop matches
            try:
                current_loop = asyncio.get_running_loop()
                if getattr(self.pool, "_loop", None) == current_loop:
                    # Remove from shared pools first
                    _shared_pools.pop(self.connection_string, None)
                    await self.pool.close()
            except Exception:
                pass
            self.pool = None
            logger.info(f"TimescaleDB connection closed for {self.table_name}.")

    async def ensure_table_exists(self):
        """
        Tables should be created manually or via migration scripts.
        """
        if not self.pool:
            return False
        # Simplified: Assume tables exist.
        return True


# Singleton instances for global access (keyed by connection_string, table_name)
_db_instances: dict[tuple[str, str], "TimescaleDBConnection"] = {}


def get_db_connection(
    config: dict = None, connection_string: str = None, table_name: str = None
) -> TimescaleDBConnection:
    """
    Get or create a database connection for the given string and table.
    """
    global _db_instances

    if not connection_string:
        connection_string = Ledger.DB_CONNECTION
    if not table_name:
        table_name = Ledger.HISTORICAL_TABLE

    key = (connection_string, table_name)

    if key not in _db_instances:
        instance = TimescaleDBConnection(connection_string, table_name)
        _db_instances[key] = instance

    return _db_instances[key]


def _log_connection_target(uri: str, health_logger: logging.Logger):
    """Log connection details without leaking credentials."""
    try:
        parsed = urlparse(uri)
        health_logger.info(
            "   Target: %s://%s:***@%s:%s%s",
            parsed.scheme,
            parsed.username,
            parsed.hostname,
            parsed.port,
            parsed.path,
        )
    except Exception:
        health_logger.info("   Target: [Could not parse connection string]")


def _mock_insert_statement(table_type: str, table_name: str, mock_id: str, ts: datetime):
    """Return health-check INSERT and DELETE statements for a ledger table."""
    if table_type == "historical":
        query = f"""
            INSERT INTO {table_name} (
                trip_id, route_id, direction_id, vehicle_id,
                latitude, longitude, hexagon_id,
                stop_sequence, shape_dist_travelled, distance_to_next_stop,
                is_in_preferential,
                measurement_time, actual_start_time,
                schedule_adherence, scheduled_start_time, delay_genuine,
                current_speed, speed_ratio, current_traffic_speed,
                temperature, apparent_temperature, humidity,
                precipitation, wind_speed, weather_code,
                bus_type, door_number, occupancy_status,
                deposits
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
                $21,$22,$23,$24,$25,$26,$27,$28,$29
            )
        """
        test_trip_id = f"TEST_TRIP_{mock_id}"
        values = (
            test_trip_id,
            "TEST_ROUTE",
            0,
            f"TEST_VEHICLE_{mock_id}",
            41.9,
            12.5,
            "TEST_HEX",
            0,
            0.0,
            0.0,
            False,
            ts,
            ts,
            0.0,
            "00:00:00",
            0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            "[]",
        )
        delete_query = f"DELETE FROM {table_name} WHERE trip_id = $1"
        delete_values = (test_trip_id,)
    elif table_type == "predicted":
        query = f"""
            INSERT INTO {table_name} (
                route_id, direction_id, trip_date, scheduled_start,
                stop_id, stop_sequence,
                predicted_arrival, predicted_delay_sec, predicted_crowd_level,
                prediction_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        test_route_id = f"TEST_ROUTE_{mock_id}"
        values = (
            test_route_id,
            0,
            ts.date(),
            "00:00:00",
            "TEST_STOP",
            0,
            "00:00:00",
            0.0,
            0,
            ts,
        )
        delete_query = f"DELETE FROM {table_name} WHERE route_id = $1"
        delete_values = (test_route_id,)
    elif table_type == "vehicle":
        query = f"""
            INSERT INTO {table_name} (
                vehicle_id, trip_id, route_id, direction_id,
                vehicle_type_name, fuel_type, euro_class, capacity_total,
                trip_date, scheduled_start, actual_start_time,
                trip_end_time, trip_duration_sec,
                mean_delay_sec, median_delay_sec, max_delay_sec,
                min_delay_sec, std_delay_sec,
                mean_occupancy, max_occupancy,
                measurement_count, preferential_ratio, recorded_at
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
                $21,$22,$23
            )
        """
        test_trip_id = f"TEST_TRIP_{mock_id}"
        values = (
            f"TEST_VEHICLE_{mock_id}",
            test_trip_id,
            "TEST_ROUTE",
            0,
            "TEST_TYPE",
            0,
            0,
            0,
            ts.date(),
            "00:00:00",
            ts,
            ts,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            1,
            0.0,
            ts,
        )
        delete_query = f"DELETE FROM {table_name} WHERE trip_id = $1"
        delete_values = (test_trip_id,)
    else:
        raise ValueError(f"Unknown table type {table_type}")

    return query, values, delete_query, delete_values


async def _test_table_operations(
    db_conn: TimescaleDBConnection,
    table_type: str,
    table_name: str,
    health_logger: logging.Logger,
):
    """Insert and delete a sanitized mock row in one table."""
    if not await db_conn.connect():
        health_logger.error("Failed to connect for %s", table_name)
        return

    health_logger.info("Connection successful for %s", table_name)
    pool = await db_conn.get_valid_pool()
    if not pool:
        health_logger.error("Failed to get pool for %s", table_name)
        await db_conn.close()
        return

    try:
        async with pool.acquire() as conn:
            mock_id = str(uuid.uuid4())
            ts = datetime.now()

            try:
                query, values, delete_query, delete_values = _mock_insert_statement(
                    table_type,
                    table_name,
                    mock_id,
                    ts,
                )
            except ValueError as e:
                health_logger.warning(str(e))
                return

            try:
                await conn.execute(query, *values)
                health_logger.info("   Mock insertion successful")
            except Exception as e:
                health_logger.error("   Insertion failed: %s", e)
                return

            try:
                await conn.execute(delete_query, *delete_values)
                health_logger.info("   Mock deletion successful")
            except Exception as e:
                health_logger.error("   Deletion failed: %s", e)

    except Exception as e:
        health_logger.error("Unexpected error during test op: %s", e)
    finally:
        await db_conn.close()


async def test_database_connection():
    """Test configured TimescaleDB connections and table write/delete operations."""
    from config import _CONFIG_PATH, load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    health_logger = logging.getLogger("test_db")

    if not _CONFIG_PATH.exists():
        health_logger.warning("Config file not found at: %s", _CONFIG_PATH)
    else:
        health_logger.info("Loading config from: %s", _CONFIG_PATH)

    load_config()
    conn_str = Ledger.DB_CONNECTION
    health_logger.info("\nTesting Ledger Database...")
    _log_connection_target(conn_str, health_logger)

    if not conn_str:
        health_logger.warning("Ledger DB connection not configured.")
        return

    tables = [
        ("Historical", Ledger.HISTORICAL_TABLE, "historical"),
        ("Predicted", Ledger.PREDICTED_TABLE, "predicted"),
        ("Vehicle", Ledger.VEHICLE_TABLE, "vehicle"),
    ]
    for label, table_name, table_type in tables:
        if not table_name:
            health_logger.warning("%s ledger table not defined", label)
            continue
        health_logger.info("   %s Table: '%s'", label, table_name)
        db_conn = TimescaleDBConnection(conn_str, table_name)
        await _test_table_operations(db_conn, table_type, table_name, health_logger)


class LedgerDBWriter:
    """Batched async writer for a single ledger table."""

    def __init__(self, connection_string: str, table_name: str, batch_size: int = 50):
        self._conn_str = connection_string
        self._table_name = table_name
        self._batch_size = batch_size
        self._pool: Any = None

    async def _ensure_pool(self):
        """Create or reuse the writer pool."""
        if self._pool is not None:
            return self._pool
        if not ASYNCPG_AVAILABLE:
            logger.error("asyncpg not available - ledger writes disabled")
            return None
        try:
            self._pool = await asyncpg.create_pool(
                self._conn_str,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        except Exception as e:
            logger.error("Failed to create pool for %s: %s", self._table_name, e)
        return self._pool

    async def insert_rows(self, query: str, rows: list[tuple]):
        """Execute an INSERT VALUES statement for a batch of rows."""
        pool = await self._ensure_pool()
        if not pool:
            return
        try:
            async with pool.acquire() as conn:
                await conn.executemany(query, rows)
            logger.debug("Inserted %s rows into %s", len(rows), self._table_name)
        except Exception as e:
            logger.error("Ledger insert failed (%s): %s", self._table_name, e)

    async def close(self):
        """Close the writer pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


def _ts(unix: float) -> datetime:
    """Convert a Unix timestamp to an aware UTC datetime."""
    return datetime.fromtimestamp(float(unix), tz=timezone.utc)


_ledger_writers: dict[str, LedgerDBWriter] = {}
_sync_engines: dict[str, Any] = {}


def _get_ledger_writer(connection_string: str, table_name: str) -> LedgerDBWriter:
    """Get or create a ledger writer for a table."""
    key = f"{connection_string}:{table_name}"
    if key not in _ledger_writers:
        _ledger_writers[key] = LedgerDBWriter(connection_string, table_name)
    return _ledger_writers[key]


def _get_db_loop():
    """Borrow the shared DB event loop from persistence.strategy."""
    from .strategy import _get_db_loop as get_loop

    return get_loop()


def write_historical(connection_string: str, table_name: str, records: list[dict]):
    """Write MeasurementRecord rows to the database."""
    if not records:
        return
    writer = _get_ledger_writer(connection_string, table_name)
    query = f"""
        INSERT INTO {table_name} (
            trip_id, route_id, direction_id, vehicle_id,
            latitude, longitude, hexagon_id,
            stop_sequence, shape_dist_travelled, distance_to_next_stop,
            is_in_preferential,
            measurement_time, actual_start_time,
            schedule_adherence, scheduled_start_time, delay_genuine,
            current_speed, speed_ratio, current_traffic_speed,
            temperature, apparent_temperature, humidity,
            precipitation, wind_speed, weather_code,
            bus_type, door_number, occupancy_status,
            deposits
        ) VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
            $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
            $21,$22,$23,$24,$25,$26,$27,$28,$29
        )
        ON CONFLICT DO NOTHING
    """
    rows = [
        (
            r["trip_id"],
            r["route_id"],
            int(r["direction_id"]),
            str(r.get("vehicle_id", "")),
            float(r["latitude"]),
            float(r["longitude"]),
            str(r.get("hexagon_id", "")),
            int(r["stop_sequence"]),
            float(r["shape_dist_travelled"]),
            float(r["distance_to_next_stop"]),
            bool(r.get("is_in_preferential", False)),
            _ts(r["measurement_time"]),
            _ts(r.get("actual_start_time") or r["measurement_time"]),
            float(r["schedule_adherence"]),
            str(r.get("scheduled_start_time", "")),
            int(r.get("delay_genuine", 0)),
            float(r["current_speed"]),
            float(r["speed_ratio"]),
            float(r["current_traffic_speed"]),
            float(r.get("temperature", 0.0)),
            float(r.get("apparent_temperature", 0.0)),
            float(r.get("humidity", 0.0)),
            float(r["precipitation"]),
            float(r.get("wind_speed", 0.0)),
            int(r["weather_code"]),
            int(r["bus_type"]),
            int(r["door_number"]),
            int(r["occupancy_status"]),
            str(r.get("deposits", "[]")),
        )
        for r in records
    ]
    asyncio.run_coroutine_threadsafe(writer.insert_rows(query, rows), _get_db_loop())


def write_predicted(connection_string: str, table_name: str, predictions: list[dict]):
    """Write StopPredictionRecord rows to the database."""
    if not predictions:
        return
    writer = _get_ledger_writer(connection_string, table_name)
    query = f"""
        INSERT INTO {table_name} (
            route_id, direction_id, trip_date, scheduled_start,
            stop_id, stop_sequence,
            predicted_arrival, predicted_delay_sec, predicted_crowd_level,
            prediction_timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT DO NOTHING
    """
    rows = [
        (
            p["route_id"],
            int(p["direction_id"]),
            datetime.strptime(p["trip_date"], "%d-%m-%Y").date()
            if p.get("trip_date")
            else None,
            p["scheduled_start"],
            p["stop_id"],
            int(p["stop_sequence"]),
            p["predicted_arrival"],
            float(p["predicted_delay_sec"]),
            int(p["predicted_crowd_level"]),
            _ts(p["prediction_timestamp"]),
        )
        for p in predictions
    ]
    asyncio.run_coroutine_threadsafe(writer.insert_rows(query, rows), _get_db_loop())


def write_vehicle_trips(connection_string: str, table_name: str, records: list[dict]):
    """Write VehicleTripRecord rows to the database."""
    if not records:
        return
    writer = _get_ledger_writer(connection_string, table_name)
    query = f"""
        INSERT INTO {table_name} (
            vehicle_id, trip_id, route_id, direction_id,
            vehicle_type_name, fuel_type, euro_class, capacity_total,
            trip_date, scheduled_start, actual_start_time,
            trip_end_time, trip_duration_sec,
            mean_delay_sec, median_delay_sec, max_delay_sec,
            min_delay_sec, std_delay_sec,
            mean_occupancy, max_occupancy,
            measurement_count, preferential_ratio, recorded_at
        ) VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
            $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
            $21,$22,$23
        )
        ON CONFLICT DO NOTHING
    """
    rows = [
        (
            r["vehicle_id"],
            r["trip_id"],
            r["route_id"],
            int(r["direction_id"]),
            r["vehicle_type_name"],
            int(r["fuel_type"]),
            int(r["euro_class"]),
            int(r["capacity_total"]),
            datetime.strptime(r["trip_date"], "%Y-%m-%d").date()
            if r.get("trip_date")
            else None,
            r["scheduled_start"],
            _ts(r["actual_start_time"]),
            _ts(r["trip_end_time"]),
            float(r["trip_duration_sec"]),
            float(r["mean_delay_sec"]),
            float(r["median_delay_sec"]),
            float(r["max_delay_sec"]),
            float(r["min_delay_sec"]),
            float(r["std_delay_sec"]),
            float(r["mean_occupancy"]),
            int(r["max_occupancy"]),
            int(r["measurement_count"]),
            float(r["preferential_ratio"]),
            _ts(r["recorded_at"]),
        )
        for r in records
    ]
    asyncio.run_coroutine_threadsafe(writer.insert_rows(query, rows), _get_db_loop())


def _get_cached_sync_engine(connection_string: str):
    """Get or create a cached sync SQLAlchemy engine."""
    if connection_string in _sync_engines:
        return _sync_engines[connection_string]
    engine = get_sync_engine(connection_string)
    if engine is not None:
        _sync_engines[connection_string] = engine
    return engine


def read_historical(
    connection_string: str,
    table_name: str,
    trip_id: str = None,
    route_id: str = None,
    date_start: float = None,
    date_end: float = None,
) -> pd.DataFrame:
    """Read historical measurements."""
    engine = _get_cached_sync_engine(connection_string)
    if engine is None:
        return pd.DataFrame()

    where = []
    if trip_id:
        where.append(f"trip_id = '{trip_id}'")
    if route_id:
        where.append(f"route_id = '{route_id}'")
    if date_start is not None:
        where.append(f"measurement_time >= to_timestamp({date_start})")
    if date_end is not None:
        where.append(f"measurement_time < to_timestamp({date_end})")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    query = f"SELECT * FROM {table_name}{clause}"
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error("Failed to read %s: %s", table_name, e)
        return pd.DataFrame()


def read_predicted(
    connection_string: str,
    table_name: str,
    route_id: str = None,
    direction_id: int = None,
    trip_date: str = None,
    scheduled_start: str = None,
) -> pd.DataFrame:
    """Read predicted arrivals."""
    engine = _get_cached_sync_engine(connection_string)
    if engine is None:
        return pd.DataFrame()

    where = []
    if route_id:
        where.append(f"route_id = '{route_id}'")
    if direction_id is not None:
        where.append(f"direction_id = {int(direction_id)}")
    if trip_date:
        where.append(f"trip_date = '{trip_date}'")
    if scheduled_start:
        prefix = scheduled_start[:5]
        where.append(f"scheduled_start LIKE '{prefix}%'")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    query = f"SELECT * FROM {table_name}{clause}"
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error("Failed to read %s: %s", table_name, e)
        return pd.DataFrame()


def read_vehicle_trips(
    connection_string: str,
    table_name: str,
    vehicle_id: str = None,
    route_id: str = None,
    fuel_type: int = None,
    date_start: str = None,
    date_end: str = None,
) -> pd.DataFrame:
    """Read vehicle trip records."""
    engine = _get_cached_sync_engine(connection_string)
    if engine is None:
        return pd.DataFrame()

    where = []
    if vehicle_id:
        where.append(f"vehicle_id = '{vehicle_id}'")
    if route_id:
        where.append(f"route_id = '{route_id}'")
    if fuel_type is not None:
        where.append(f"fuel_type = {fuel_type}")
    if date_start:
        where.append(f"trip_date >= '{date_start}'")
    if date_end:
        where.append(f"trip_date <= '{date_end}'")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    query = f"SELECT * FROM {table_name}{clause}"
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error("Failed to read %s: %s", table_name, e)
        return pd.DataFrame()


def read_prediction_training_rows(start_date: str = None) -> pd.DataFrame:
    """Read historical measurements for dataset extraction."""
    engine = get_sync_engine(Ledger.DB_CONNECTION)
    if engine is None:
        return pd.DataFrame()

    query = f"""
    SELECT
        *,
        measurement_time AS ts
    FROM {Ledger.HISTORICAL_TABLE}
    """
    if start_date:
        query += f"\n    WHERE measurement_time >= '{start_date} 00:00:00'"

    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error("Error querying historical training rows: %s", e)
        return pd.DataFrame()


def read_traffic_training_rows() -> pd.DataFrame:
    """Read traffic-related historical measurements for canonical averages."""
    engine = get_sync_engine(Ledger.DB_CONNECTION)
    if engine is None:
        return pd.DataFrame()

    query = f"""
    SELECT
        hexagon_id AS h3_index,
        measurement_time AS ts,
        speed_ratio,
        current_traffic_speed
    FROM {Ledger.HISTORICAL_TABLE}
    WHERE hexagon_id IS NOT NULL
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return df
        ts = pd.to_datetime(df["ts"])
        df["day_type"] = ts.dt.weekday.map(lambda weekday: 1 if weekday == 5 else 2 if weekday == 6 else 0)
        return df
    except Exception as e:
        logger.error("Error querying historical traffic rows: %s", e)
        return pd.DataFrame()


async def fetch_validation_rows_by_trip_ids(trip_ids: set[str]) -> dict[str, list[Any]]:
    """Fetch validation fallback rows grouped by trip id."""
    if not trip_ids:
        return {}
    if not ASYNCPG_AVAILABLE:
        logger.warning("asyncpg not available: DB fallback disabled")
        return {}

    conn_str = Ledger.DB_CONNECTION
    if not conn_str:
        logger.warning("Ledger DB connection not configured: DB fallback disabled")
        return {}

    rows_by_trip: dict[str, list[Any]] = {}
    conn = None
    try:
        conn = await asyncpg.connect(conn_str, timeout=30)
        query = f"""
            SELECT trip_id, stop_sequence, schedule_adherence, occupancy_status
            FROM {Ledger.HISTORICAL_TABLE}
            WHERE trip_id = ANY($1)
            ORDER BY trip_id, measurement_time ASC
        """
        rows = await conn.fetch(query, list(trip_ids))
        for row in rows:
            tid = row["trip_id"]
            rows_by_trip.setdefault(tid, []).append(row)
    except Exception:
        logger.exception("DB fallback query failed")
    finally:
        if conn is not None:
            await conn.close()
    return rows_by_trip


async def close_ledger_writers():
    """Close all ledger writer pools."""
    for writer in _ledger_writers.values():
        await writer.close()
    _ledger_writers.clear()


async def init_database(config: dict = None) -> bool:
    """Initialize the configured ledger database connections."""

    instances = [
        get_db_connection(
            connection_string=Ledger.DB_CONNECTION,
            table_name=Ledger.HISTORICAL_TABLE,
        ),
        get_db_connection(
            connection_string=Ledger.DB_CONNECTION,
            table_name=Ledger.PREDICTED_TABLE,
        ),
        get_db_connection(
            connection_string=Ledger.DB_CONNECTION,
            table_name=Ledger.VEHICLE_TABLE,
        ),
    ]

    results = await asyncio.gather(*[inst.connect() for inst in instances])
    return all(results)


async def close_database():
    """Close all database connections and shared pools."""
    global _db_instances
    for conn in _db_instances.values():
        await conn.close()
    await close_ledger_writers()
    _db_instances.clear()
    _shared_pools.clear()
    _sync_engines.clear()


def shutdown_database():
    """
    Synchronous helper: close all DB pools on the shared event loop,
    then shut down the loop. Safe to call from any thread.
    """
    from .strategy import _get_db_loop, shutdown_db_loop

    try:
        loop = _get_db_loop()
        future = asyncio.run_coroutine_threadsafe(close_database(), loop)
        future.result(timeout=10)
    except Exception as e:
        logger.error(f"Error closing database pools: {e}")
    finally:
        shutdown_db_loop()


def get_sync_engine(connection_string: str = None):
    """
    Get a synchronous SQLAlchemy engine for batch operations.

    Converts postgresql:// URLs to postgresql+psycopg2:// for SQLAlchemy.

    Args:
        connection_string: PostgreSQL connection string. Defaults to Ledger.DB_CONNECTION.

    Returns:
        SQLAlchemy Engine or None if not configured.
    """
    import sqlalchemy
    from urllib.parse import urlparse

    if connection_string is None:
        connection_string = Ledger.DB_CONNECTION

    if not connection_string:
        return None

    if "user:password" in connection_string:
        return None

    try:
        parsed = urlparse(connection_string)
        port = parsed.port or 5432
        sqlalchemy_url = f"postgresql+psycopg2://{parsed.username}:{parsed.password}@{parsed.hostname}:{port}{parsed.path}"
        return sqlalchemy.create_engine(sqlalchemy_url)
    except Exception as e:
        logger.error(f"Failed to create sync engine: {e}")
        return None


def get_sync_engine_for_pipeline(pipeline: str = "prediction"):
    """
    Convenience getter for the configured ledger database.

    Args:
        pipeline: Ignored. Kept for callers that still pass a dataset name.

    Returns:
        SQLAlchemy Engine or None if not configured.
    """
    return get_sync_engine(Ledger.DB_CONNECTION)
