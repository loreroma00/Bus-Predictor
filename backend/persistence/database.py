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

from config import Config

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


HISTORICAL_COLUMNS = (
    "trip_id",
    "route_id",
    "direction_id",
    "vehicle_id",
    "latitude",
    "longitude",
    "hexagon_id",
    "stop_sequence",
    "shape_dist_travelled",
    "distance_to_next_stop",
    "is_in_preferential",
    "measurement_time",
    "actual_start_time",
    "schedule_adherence",
    "scheduled_start_time",
    "delay_genuine",
    "current_speed",
    "speed_ratio",
    "current_traffic_speed",
    "temperature",
    "apparent_temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "weather_code",
    "bus_type",
    "door_number",
    "occupancy_status",
    "deposits",
)

PREDICTED_COLUMNS = (
    "route_id",
    "direction_id",
    "trip_date",
    "scheduled_start",
    "stop_id",
    "stop_sequence",
    "predicted_arrival",
    "predicted_delay_sec",
    "predicted_crowd_level",
    "prediction_timestamp",
)

VEHICLE_TRIP_COLUMNS = (
    "vehicle_id",
    "trip_id",
    "route_id",
    "direction_id",
    "vehicle_type_name",
    "fuel_type",
    "euro_class",
    "capacity_total",
    "trip_date",
    "scheduled_start",
    "actual_start_time",
    "trip_end_time",
    "trip_duration_sec",
    "mean_delay_sec",
    "median_delay_sec",
    "max_delay_sec",
    "min_delay_sec",
    "std_delay_sec",
    "mean_occupancy",
    "max_occupancy",
    "measurement_count",
    "preferential_ratio",
    "recorded_at",
)


def get_db_connection(
    connection_string: str,
    table_name: str,
) -> TimescaleDBConnection:
    """
    Get or create a database connection for the given string and table.
    """
    global _db_instances

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


def _insert_statement(
    table_name: str,
    columns: tuple[str, ...],
    on_conflict: str | None = "ON CONFLICT DO NOTHING",
) -> str:
    """Build an asyncpg positional INSERT statement for a ledger table."""
    column_sql = ", ".join(columns)
    placeholder_sql = ", ".join(f"${idx}" for idx in range(1, len(columns) + 1))
    conflict_sql = f"\n        {on_conflict}" if on_conflict else ""
    return f"""
        INSERT INTO {table_name} (
            {column_sql}
        ) VALUES (
            {placeholder_sql}
        ){conflict_sql}
    """


def _mock_insert_statement(table_type: str, table_name: str, mock_id: str, ts: datetime):
    """Return health-check INSERT and DELETE statements for a ledger table."""
    if table_type == "historical":
        query = _insert_statement(table_name, HISTORICAL_COLUMNS, on_conflict=None)
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
        query = _insert_statement(table_name, PREDICTED_COLUMNS, on_conflict=None)
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
        query = _insert_statement(table_name, VEHICLE_TRIP_COLUMNS, on_conflict=None)
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


async def test_database_connection(config: Config | None = None):
    """Test configured TimescaleDB connections and table write/delete operations."""
    config = Config.coerce(config)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    health_logger = logging.getLogger("test_db")

    if not config.source_path.exists():
        health_logger.warning("Config file not found at: %s", config.source_path)
    else:
        health_logger.info("Loading config from: %s", config.source_path)

    conn_str = config.ledger.db_connection
    health_logger.info("\nTesting Ledger Database...")
    _log_connection_target(conn_str, health_logger)

    if not conn_str:
        health_logger.warning("Ledger DB connection not configured.")
        return

    tables = [
        ("Historical", config.ledger.historical_table, "historical"),
        ("Predicted", config.ledger.predicted_table, "predicted"),
        ("Vehicle", config.ledger.vehicle_table, "vehicle"),
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

    async def insert_rows(self, insert_query: str, rows: list[tuple]):
        """Execute an INSERT VALUES statement for a batch of rows."""
        pool = await self._ensure_pool()
        if not pool:
            return
        try:
            async with pool.acquire() as conn:
                await conn.executemany(insert_query, rows)
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


def _prediction_date(value: str | None):
    """Parse API-facing prediction dates."""
    return datetime.strptime(value, "%d-%m-%Y").date() if value else None


def _trip_date(value: str | None):
    """Parse ledger-facing vehicle trip dates."""
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


def _historical_values(record: dict) -> tuple:
    """Normalize one historical measurement dict to DB column order."""
    return (
        record["trip_id"],
        record["route_id"],
        int(record["direction_id"]),
        str(record.get("vehicle_id", "")),
        float(record["latitude"]),
        float(record["longitude"]),
        str(record.get("hexagon_id", "")),
        int(record["stop_sequence"]),
        float(record["shape_dist_travelled"]),
        float(record["distance_to_next_stop"]),
        bool(record.get("is_in_preferential", False)),
        _ts(record["measurement_time"]),
        _ts(record.get("actual_start_time") or record["measurement_time"]),
        float(record["schedule_adherence"]),
        str(record.get("scheduled_start_time", "")),
        int(record.get("delay_genuine", 0)),
        float(record["current_speed"]),
        float(record["speed_ratio"]),
        float(record["current_traffic_speed"]),
        float(record.get("temperature", 0.0)),
        float(record.get("apparent_temperature", 0.0)),
        float(record.get("humidity", 0.0)),
        float(record["precipitation"]),
        float(record.get("wind_speed", 0.0)),
        int(record["weather_code"]),
        int(record["bus_type"]),
        int(record["door_number"]),
        int(record["occupancy_status"]),
        str(record.get("deposits", "[]")),
    )


def _predicted_values(prediction: dict) -> tuple:
    """Normalize one stop prediction dict to DB column order."""
    return (
        prediction["route_id"],
        int(prediction["direction_id"]),
        _prediction_date(prediction.get("trip_date")),
        prediction["scheduled_start"],
        prediction["stop_id"],
        int(prediction["stop_sequence"]),
        prediction["predicted_arrival"],
        float(prediction["predicted_delay_sec"]),
        int(prediction["predicted_crowd_level"]),
        _ts(prediction["prediction_timestamp"]),
    )


def _vehicle_trip_values(record: dict) -> tuple:
    """Normalize one vehicle trip dict to DB column order."""
    return (
        record["vehicle_id"],
        record["trip_id"],
        record["route_id"],
        int(record["direction_id"]),
        record["vehicle_type_name"],
        int(record["fuel_type"]),
        int(record["euro_class"]),
        int(record["capacity_total"]),
        _trip_date(record.get("trip_date")),
        record["scheduled_start"],
        _ts(record["actual_start_time"]),
        _ts(record["trip_end_time"]),
        float(record["trip_duration_sec"]),
        float(record["mean_delay_sec"]),
        float(record["median_delay_sec"]),
        float(record["max_delay_sec"]),
        float(record["min_delay_sec"]),
        float(record["std_delay_sec"]),
        float(record["mean_occupancy"]),
        int(record["max_occupancy"]),
        int(record["measurement_count"]),
        float(record["preferential_ratio"]),
        _ts(record["recorded_at"]),
    )


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


def _submit_insert(
    connection_string: str,
    table_name: str,
    columns: tuple[str, ...],
    rows: list[tuple],
) -> None:
    """Submit an async batched insert on the shared DB event loop."""
    if not rows:
        return
    writer = _get_ledger_writer(connection_string, table_name)
    insert_query = _insert_statement(table_name, columns)
    asyncio.run_coroutine_threadsafe(
        writer.insert_rows(insert_query, rows),
        _get_db_loop(),
    )


def write_historical(connection_string: str, table_name: str, records: list[dict]):
    """Write historical measurement rows to the database."""
    if not records:
        return
    _submit_insert(
        connection_string,
        table_name,
        HISTORICAL_COLUMNS,
        [_historical_values(record) for record in records],
    )


def write_predicted(connection_string: str, table_name: str, predictions: list[dict]):
    """Write stop prediction rows to the database."""
    if not predictions:
        return
    _submit_insert(
        connection_string,
        table_name,
        PREDICTED_COLUMNS,
        [_predicted_values(prediction) for prediction in predictions],
    )


def write_vehicle_trips(connection_string: str, table_name: str, records: list[dict]):
    """Write vehicle trip rows to the database."""
    if not records:
        return
    _submit_insert(
        connection_string,
        table_name,
        VEHICLE_TRIP_COLUMNS,
        [_vehicle_trip_values(record) for record in records],
    )


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


def read_historical_training_rows(
    connection_string: str,
    table_name: str,
    start_date: str = None,
) -> pd.DataFrame:
    """Read historical measurements for dataset extraction."""
    engine = get_sync_engine(connection_string)
    if engine is None:
        return pd.DataFrame()

    query = f"""
    SELECT
        *,
        measurement_time AS ts
    FROM {table_name}
    """
    if start_date:
        query += f"\n    WHERE measurement_time >= '{start_date} 00:00:00'"

    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error("Error querying historical training rows: %s", e)
        return pd.DataFrame()


def read_historical_traffic_rows(
    connection_string: str,
    table_name: str,
) -> pd.DataFrame:
    """Read traffic-related historical measurements for canonical averages."""
    engine = get_sync_engine(connection_string)
    if engine is None:
        return pd.DataFrame()

    query = f"""
    SELECT
        hexagon_id AS h3_index,
        measurement_time AS ts,
        speed_ratio,
        current_traffic_speed
    FROM {table_name}
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


async def fetch_validation_rows_by_trip_ids(
    trip_ids: set[str],
    connection_string: str,
    table_name: str,
) -> dict[str, list[Any]]:
    """Fetch validation fallback rows grouped by trip id."""
    if not trip_ids:
        return {}
    if not ASYNCPG_AVAILABLE:
        logger.warning("asyncpg not available: DB fallback disabled")
        return {}

    conn_str = connection_string
    if not conn_str:
        logger.warning("Ledger DB connection not configured: DB fallback disabled")
        return {}

    rows_by_trip: dict[str, list[Any]] = {}
    conn = None
    try:
        conn = await asyncpg.connect(conn_str, timeout=30)
        query = f"""
            SELECT trip_id, stop_sequence, schedule_adherence, occupancy_status
            FROM {table_name}
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


async def init_database(config: Config | None = None) -> bool:
    """Initialize the configured ledger database connections."""
    config = Config.coerce(config)

    instances = [
        get_db_connection(
            connection_string=config.ledger.db_connection,
            table_name=config.ledger.historical_table,
        ),
        get_db_connection(
            connection_string=config.ledger.db_connection,
            table_name=config.ledger.predicted_table,
        ),
        get_db_connection(
            connection_string=config.ledger.db_connection,
            table_name=config.ledger.vehicle_table,
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


def get_sync_engine(connection_string: str):
    """
    Get a synchronous SQLAlchemy engine for batch operations.

    Converts postgresql:// URLs to postgresql+psycopg2:// for SQLAlchemy.

    Args:
        connection_string: PostgreSQL connection string.

    Returns:
        SQLAlchemy Engine or None if not configured.
    """
    import sqlalchemy
    from urllib.parse import urlparse

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
