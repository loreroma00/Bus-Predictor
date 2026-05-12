"""
Database Access Layer - TimescaleDB persistence for Vector data.

Uses asyncpg for async connection pooling and batched writes.
"""

import logging
import os
import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List
from urllib.parse import urlparse
import uuid

import pandas as pd

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

from application.post_processing.vectorization import (
    PredictionVector,
    PredictionLabel,
    TrafficVector,
    TrafficLabel,
    VehicleVector,
    VehicleLabel,
)
from application.domain.time_utils import get_seconds_since_midnight
from config import Prediction, Traffic, Vehicle

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
        self._write_queue = []
        self._queue_lock = asyncio.Lock()
        self._batch_size = 100

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
        """Close connection pool and flush remaining writes."""
        if self._write_queue:
            await self._flush_queue()
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

    async def insert_items(self, items: List[Any]):
        """
        Queue items (Vectors or Labels) for async batch insert.
        """
        if not items:
            return

        async with self._queue_lock:
            self._write_queue.extend(items)
            if len(self._write_queue) >= self._batch_size:
                await self._flush_queue()

    async def _flush_queue(self):
        """Flush the write queue to database."""
        if not self._write_queue:
            return

        items = self._write_queue[:]
        self._write_queue.clear()

        if not items:
            return

        pool = await self.get_valid_pool()
        if not pool:
            # Failed to connect, re-queue items?
            # For now, log error and drop to prevent memory leak if DB is down forever
            logger.error(
                f"❌ Failed to get connection pool for {self.table_name}. Dropping {len(items)} items."
            )
            return

        # Determine type based on first item
        first_item = items[0]
        query = ""
        rows = []

        try:
            if isinstance(first_item, PredictionVector):
                query, rows = self._prepare_prediction_vectors(items)
            elif isinstance(first_item, PredictionLabel):
                query, rows = self._prepare_prediction_labels(items)
            elif isinstance(first_item, TrafficVector):
                query, rows = self._prepare_traffic_vectors(items)
            elif isinstance(first_item, TrafficLabel):
                query, rows = self._prepare_traffic_labels(items)
            elif isinstance(first_item, VehicleVector):
                query, rows = self._prepare_vehicle_vectors(items)
            elif isinstance(first_item, VehicleLabel):
                query, rows = self._prepare_vehicle_labels(items)
            else:
                logger.warning(f"⚠️ Unknown item type for DB insert: {type(first_item)}")
                return

            if not query:
                return

            async with pool.acquire() as conn:
                await conn.executemany(query, rows)

            logger.info(f"📊 Inserted {len(rows)} rows into {self.table_name}")

        except Exception as e:
            logger.error(f"❌ TimescaleDB insert failed ({self.table_name}): {e}")
            # Optional: Re-queue failed rows or handle error strategy
            # self._write_queue.extend(items)

    def _prepare_prediction_vectors(self, items: List[PredictionVector]):
        """Prepare prediction vectors."""
        query = f"""
            INSERT INTO {self.table_name} (
                id, ts, route_id, direction_id, stop_sequence,
                shape_dist_travelled, distance_to_next_stop, far_status,
                day_type, rush_hour_status, time_feat, time_sin, time_cos,
                schedule_adherence, speed_ratio, current_traffic_speed, 
                current_speed, precipitation, weather_code, bus_type, door_number,
                deposit_grottarossa, deposit_magliana, deposit_tor_sapienza, deposit_portonaccio,
                deposit_monte_sacro, deposit_tor_pagnotta, deposit_tor_cervara, deposit_maglianella,
                deposit_costi, deposit_trastevere, deposit_acilia, deposit_tor_vergata,
                deposit_porta_maggiore,
                served_ratio,
                trip_id,
                sch_starting_time_cos, sch_starting_time_sin,
                starting_time_cos, starting_time_sin,
                delay_genuine
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36,
                $37, $38, $39, $40, $41
            )
            ON CONFLICT DO NOTHING
        """
        rows = []
        for v in items:
            ts = (
                datetime.fromtimestamp(v.time) if hasattr(v, "time") else datetime.now()
            )
            # Calculate time_feat
            seconds = get_seconds_since_midnight(v.time) if hasattr(v, "time") else 0
            time_feat = seconds / 86400.0

            rows.append(
                (
                    v.id,
                    ts,
                    str(v.route_id),
                    int(v.direction_id),
                    int(v.stop_sequence),
                    float(v.shape_dist_travelled),
                    float(v.distance_to_next_stop),
                    bool(v.far_status),
                    int(v.day_type),
                    bool(v.rush_hour_status),
                    float(time_feat),
                    float(v.time_encoding[0]),
                    float(v.time_encoding[1]),
                    float(v.schedule_adherence),
                    float(v.speed_ratio),
                    float(v.current_traffic_speed),
                    float(v.current_speed),
                    float(v.precipitation),
                    int(v.weather_code),
                    int(v.bus_type),
                    int(v.door_number),
                    int(v.deposit_grottarossa),
                    int(v.deposit_magliana),
                    int(v.deposit_tor_sapienza),
                    int(v.deposit_portonaccio),
                    int(v.deposit_monte_sacro),
                    int(v.deposit_tor_pagnotta),
                    int(v.deposit_tor_cervara),
                    int(v.deposit_maglianella),
                    int(v.deposit_costi),
                    int(v.deposit_trastevere),
                    int(v.deposit_acilia),
                    int(v.deposit_tor_vergata),
                    int(v.deposit_porta_maggiore),
                    float(v.served_ratio),
                    str(v.trip_id),
                    float(v.sch_starting_time_cos),
                    float(v.sch_starting_time_sin),
                    float(v.starting_time_cos),
                    float(v.starting_time_sin),
                    int(v.delay_genuine),
                )
            )
        return query, rows

    def _prepare_prediction_labels(self, items: List[PredictionLabel]):
        """Prepare prediction labels."""
        query = f"""
            INSERT INTO {self.table_name} (
                id, ts, time_seconds, occupancy_status
            ) VALUES (
                $1, $2, $3, $4
            )
            ON CONFLICT DO NOTHING
        """
        rows = []
        for l in items:
            ts = (
                datetime.fromtimestamp(l.time) if hasattr(l, "time") else datetime.now()
            )
            rows.append(
                (
                    l.id,
                    ts,
                    int(l.time_seconds),
                    int(l.occupancy_status),
                )
            )
        return query, rows

    def _prepare_traffic_vectors(self, items: List[TrafficVector]):
        """Prepare traffic vectors."""
        query = f"""
            INSERT INTO {self.table_name} (
                id, ts, day_type, rush_hour_status, time_sin, time_cos, hexagon_id, trip_id
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8
            )
            ON CONFLICT DO NOTHING
        """
        rows = []
        for v in items:
            ts = (
                datetime.fromtimestamp(v.time) if hasattr(v, "time") else datetime.now()
            )
            rows.append(
                (
                    v.id,
                    ts,
                    int(v.day_type),
                    bool(v.rush_hour_status),
                    float(v.time_encoding[0]),
                    float(v.time_encoding[1]),
                    str(v.hexagon_id) if v.hexagon_id else None,
                    str(v.trip_id),
                )
            )
        return query, rows

    def _prepare_traffic_labels(self, items: List[TrafficLabel]):
        """Prepare traffic labels."""
        query = f"""
            INSERT INTO {self.table_name} (
                id, ts, speed_ratio, current_traffic_speed
            ) VALUES (
                $1, $2, $3, $4
            )
            ON CONFLICT DO NOTHING
        """
        rows = []
        for l in items:
            ts = (
                datetime.fromtimestamp(l.time) if hasattr(l, "time") else datetime.now()
            )
            rows.append(
                (
                    l.id,
                    ts,
                    float(l.speed_ratio),
                    float(l.current_traffic_speed),
                )
            )
        return query, rows

    def _prepare_vehicle_vectors(self, items: List[VehicleVector]):
        """Prepare vehicle vectors."""
        query = f"""
            INSERT INTO {self.table_name} (
                id, ts, route_id, trip_id, direction_id
            ) VALUES (
                $1, $2, $3, $4, $5
            )
            ON CONFLICT DO NOTHING
        """
        rows = []
        for v in items:
            ts = (
                datetime.fromtimestamp(v.timestamp)
                if hasattr(v, "timestamp")
                else datetime.now()
            )
            rows.append(
                (
                    v.id,
                    ts,
                    str(v.route_id),
                    str(v.trip_id),
                    int(v.direction_id),
                )
            )
        return query, rows

    def _prepare_vehicle_labels(self, items: List[VehicleLabel]):
        """Prepare vehicle labels."""
        query = f"""
            INSERT INTO {self.table_name} (
                id, ts, vehicle_type
            ) VALUES (
                $1, $2, $3
            )
            ON CONFLICT DO NOTHING
        """
        rows = []
        for l in items:
            ts = (
                datetime.fromtimestamp(l.time) if hasattr(l, "time") else datetime.now()
            )
            rows.append(
                (
                    l.id,
                    ts,
                    str(l.vehicle_type),
                )
            )
        return query, rows

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
        connection_string = Prediction.VECTOR_DB_CONNECTION
    if not table_name:
        table_name = Prediction.VECTOR_TABLE

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
    """Return the health-check INSERT statement and values for a configured table."""
    if table_type == "pred_vec":
        query = f"""
            INSERT INTO {table_name} (
                id, ts, route_id, direction_id, stop_sequence,
                shape_dist_travelled, distance_to_next_stop, far_status,
                day_type, rush_hour_status, time_feat, time_sin, time_cos,
                schedule_adherence, speed_ratio, current_traffic_speed,
                current_speed, precipitation, weather_code, bus_type, door_number,
                deposit_grottarossa, deposit_magliana, deposit_tor_sapienza, deposit_portonaccio,
                deposit_monte_sacro, deposit_tor_pagnotta, deposit_tor_cervara, deposit_maglianella,
                deposit_costi, deposit_trastevere, deposit_acilia, deposit_tor_vergata,
                deposit_porta_maggiore,
                served_ratio,
                trip_id,
                sch_starting_time_cos, sch_starting_time_sin,
                starting_time_cos, starting_time_sin,
                delay_genuine
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35,
                $36, $37, $38, $39, $40, $41
            )
        """
        values = (
            mock_id,
            ts,
            "TEST",
            0,
            0,
            0.0,
            0.0,
            False,
            0,
            False,
            0.0,
            0.0,
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
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            "TEST_TRIP",
            0.0,
            0.0,
            0.0,
            0.0,
            0,
        )
    elif table_type == "pred_lbl":
        query = f"""
            INSERT INTO {table_name} (id, ts, time_seconds, occupancy_status)
            VALUES ($1, $2, $3, $4)
        """
        values = (mock_id, ts, 0, 0)
    elif table_type == "traf_vec":
        query = f"""
            INSERT INTO {table_name} (id, ts, day_type, rush_hour_status, time_sin, time_cos, hexagon_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        values = (mock_id, ts, 0, False, 0.0, 0.0, "TEST_HEX")
    elif table_type == "traf_lbl":
        query = f"""
            INSERT INTO {table_name} (id, ts, speed_ratio, current_traffic_speed)
            VALUES ($1, $2, $3, $4)
        """
        values = (mock_id, ts, 0.0, 0.0)
    elif table_type == "veh_vec":
        query = f"""
            INSERT INTO {table_name} (id, ts, route_id, trip_id, direction_id)
            VALUES ($1, $2, $3, $4, $5)
        """
        values = (mock_id, ts, "TEST_ROUTE", "TEST_TRIP", 0)
    elif table_type == "veh_lbl":
        query = f"""
            INSERT INTO {table_name} (id, ts, vehicle_type)
            VALUES ($1, $2, $3)
        """
        values = (mock_id, ts, "TEST_TYPE")
    else:
        raise ValueError(f"Unknown table type {table_type}")

    return query, values


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
                query, values = _mock_insert_statement(table_type, table_name, mock_id, ts)
            except ValueError as e:
                health_logger.warning(str(e))
                return

            try:
                await conn.execute(query, *values)
                health_logger.info("   Mock insertion successful")
            except Exception as e:
                health_logger.error("   Insertion failed: %s", e)
                return

            delete_query = f"DELETE FROM {table_name} WHERE id = $1"
            try:
                await conn.execute(delete_query, mock_id)
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

    config = load_config()

    if Prediction.ENABLED:
        health_logger.info("\nTesting Prediction Database...")
        pred_section = config.get("prediction", {})
        health_logger.info("   Loaded Keys: %s", list(pred_section.keys()))
        if "vector_db_connection" in pred_section:
            _log_connection_target(pred_section["vector_db_connection"], health_logger)
        else:
            health_logger.warning("   'vector_db_connection' NOT found in [prediction] section!")

        conn_str = Prediction.VECTOR_DB_CONNECTION
        table_vec = Prediction.VECTOR_TABLE
        table_lbl = Prediction.LABEL_TABLE

        health_logger.info("   Resolved Connection String:")
        _log_connection_target(conn_str, health_logger)

        if conn_str:
            if table_vec:
                health_logger.info("   Vector Table: '%s'", table_vec)
                db_vec = TimescaleDBConnection(conn_str, table_vec)
                await _test_table_operations(db_vec, "pred_vec", table_vec, health_logger)
            else:
                health_logger.warning("Prediction vector table not defined")

            if table_lbl:
                health_logger.info("   Label Table: '%s'", table_lbl)
                db_lbl = TimescaleDBConnection(conn_str, table_lbl)
                await _test_table_operations(db_lbl, "pred_lbl", table_lbl, health_logger)
            else:
                health_logger.warning("Prediction label table not defined")
    else:
        health_logger.info("\nPrediction pipeline disabled.")

    if Traffic.ENABLED:
        health_logger.info("\nTesting Traffic Database...")
        conn_str = Traffic.VECTOR_DB_CONNECTION
        table_vec = Traffic.VECTOR_TABLE
        table_lbl = Traffic.LABEL_TABLE
        _log_connection_target(conn_str, health_logger)

        if conn_str:
            if table_vec:
                health_logger.info("   Vector Table: '%s'", table_vec)
                db_vec = TimescaleDBConnection(conn_str, table_vec)
                await _test_table_operations(db_vec, "traf_vec", table_vec, health_logger)
            else:
                health_logger.warning("Traffic vector table not defined")

            if table_lbl:
                health_logger.info("   Label Table: '%s'", table_lbl)
                db_lbl = TimescaleDBConnection(conn_str, table_lbl)
                await _test_table_operations(db_lbl, "traf_lbl", table_lbl, health_logger)
            else:
                health_logger.warning("Traffic label table not defined")
    else:
        health_logger.info("\nTraffic pipeline disabled.")

    if Vehicle.ENABLED:
        health_logger.info("\nTesting Vehicle Database...")
        conn_str = Vehicle.VECTOR_DB_CONNECTION
        table_vec = Vehicle.VECTOR_TABLE
        table_lbl = Vehicle.LABEL_TABLE
        _log_connection_target(conn_str, health_logger)

        if conn_str:
            if table_vec:
                health_logger.info("   Vector Table: '%s'", table_vec)
                db_vec = TimescaleDBConnection(conn_str, table_vec)
                await _test_table_operations(db_vec, "veh_vec", table_vec, health_logger)
            else:
                health_logger.warning("Vehicle vector table not defined")

            if table_lbl:
                health_logger.info("   Label Table: '%s'", table_lbl)
                db_lbl = TimescaleDBConnection(conn_str, table_lbl)
                await _test_table_operations(db_lbl, "veh_lbl", table_lbl, health_logger)
            else:
                health_logger.warning("Vehicle label table not defined")
    else:
        health_logger.info("\nVehicle pipeline disabled.")


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
    """Read joined prediction vectors/labels for dataset extraction."""
    engine = get_sync_engine_for_pipeline("prediction")
    if engine is None:
        return pd.DataFrame()

    query = f"""
    SELECT
        v.id,
        v.ts,
        v.trip_id,
        v.route_id,
        v.direction_id,
        v.stop_sequence,
        v.shape_dist_travelled,
        v.distance_to_next_stop,
        v.far_status,
        v.day_type,
        v.rush_hour_status,
        v.time_feat,
        v.time_sin,
        v.time_cos,
        v.schedule_adherence,
        v.speed_ratio,
        v.current_traffic_speed,
        v.current_speed,
        v.precipitation,
        v.weather_code,
        v.bus_type,
        v.door_number,
        v.deposit_grottarossa,
        v.deposit_magliana,
        v.deposit_tor_sapienza,
        v.deposit_portonaccio,
        v.deposit_monte_sacro,
        v.deposit_tor_pagnotta,
        v.deposit_tor_cervara,
        v.deposit_maglianella,
        v.deposit_costi,
        v.deposit_trastevere,
        v.deposit_acilia,
        v.deposit_tor_vergata,
        v.deposit_porta_maggiore,
        v.served_ratio,
        v.starting_time_sin,
        v.starting_time_cos,
        v.sch_starting_time_sin,
        v.sch_starting_time_cos,
        l.time_seconds,
        l.occupancy_status
    FROM {Prediction.VECTOR_TABLE} v
    JOIN {Prediction.LABEL_TABLE} l ON v.id = l.id
    """
    if start_date:
        query += f"\n    WHERE v.ts >= '{start_date} 00:00:00'"

    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error("Error querying prediction training rows: %s", e)
        return pd.DataFrame()


def read_traffic_training_rows() -> pd.DataFrame:
    """Read joined traffic vectors/labels for canonical traffic averages."""
    engine = get_sync_engine_for_pipeline("traffic")
    if engine is None:
        return pd.DataFrame()

    query = f"""
    SELECT
        v.hexagon_id AS h3_index,
        v.day_type,
        v.ts,
        l.speed_ratio,
        l.current_traffic_speed
    FROM {Traffic.VECTOR_TABLE} v
    JOIN {Traffic.LABEL_TABLE} l ON v.id::text = l.id::text AND v.ts = l.ts
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error("Error querying traffic training rows: %s", e)
        return pd.DataFrame()


async def fetch_validation_rows_by_trip_ids(trip_ids: set[str]) -> dict[str, list[Any]]:
    """Fetch validation fallback rows grouped by trip id."""
    if not trip_ids:
        return {}
    if not ASYNCPG_AVAILABLE:
        logger.warning("asyncpg not available: DB fallback disabled")
        return {}

    conn_str = Prediction.VECTOR_DB_CONNECTION
    if not conn_str:
        logger.warning("Prediction DB connection not configured: DB fallback disabled")
        return {}

    rows_by_trip: dict[str, list[Any]] = {}
    conn = None
    try:
        conn = await asyncpg.connect(conn_str, timeout=30)
        query = f"""
            SELECT v.trip_id, v.stop_sequence, v.schedule_adherence, l.occupancy_status
            FROM {Prediction.VECTOR_TABLE} v
            LEFT JOIN {Prediction.LABEL_TABLE} l ON v.id = l.id
            WHERE v.trip_id = ANY($1)
            ORDER BY v.trip_id, v.ts ASC
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
    """Initialize the default database connections (shared pools)."""

    # Collect all instances that need connecting
    instances = [
        get_db_connection(
            connection_string=Prediction.VECTOR_DB_CONNECTION,
            table_name=Prediction.VECTOR_TABLE,
        ),
        get_db_connection(
            connection_string=Prediction.VECTOR_DB_CONNECTION,
            table_name=Prediction.LABEL_TABLE,
        ),
    ]

    if Traffic.ENABLED:
        instances.append(get_db_connection(
            connection_string=Traffic.VECTOR_DB_CONNECTION,
            table_name=Traffic.VECTOR_TABLE,
        ))
        instances.append(get_db_connection(
            connection_string=Traffic.VECTOR_DB_CONNECTION,
            table_name=Traffic.LABEL_TABLE,
        ))

    if Vehicle.ENABLED:
        instances.append(get_db_connection(
            connection_string=Vehicle.VECTOR_DB_CONNECTION,
            table_name=Vehicle.VECTOR_TABLE,
        ))
        instances.append(get_db_connection(
            connection_string=Vehicle.VECTOR_DB_CONNECTION,
            table_name=Vehicle.LABEL_TABLE,
        ))

    # connect() reuses shared pools, so only the first call per
    # connection_string actually creates a pool.
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
        connection_string: PostgreSQL connection string. Defaults to Prediction.VECTOR_DB_CONNECTION.

    Returns:
        SQLAlchemy Engine or None if not configured.
    """
    import sqlalchemy
    from urllib.parse import urlparse

    if connection_string is None:
        connection_string = Prediction.VECTOR_DB_CONNECTION

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
    Convenience getter for specific pipeline databases.

    Args:
        pipeline: One of "prediction", "traffic", or "vehicle"

    Returns:
        SQLAlchemy Engine or None if not configured.
    """
    pipeline_map = {
        "prediction": Prediction.VECTOR_DB_CONNECTION,
        "traffic": Traffic.VECTOR_DB_CONNECTION,
        "vehicle": Vehicle.VECTOR_DB_CONNECTION,
    }

    conn_str = pipeline_map.get(pipeline)
    if not conn_str:
        return None

    return get_sync_engine(conn_str)
