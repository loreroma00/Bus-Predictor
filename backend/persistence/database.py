"""
Database Access Layer - TimescaleDB persistence for Vector data.

Uses asyncpg for async connection pooling and batched writes.
"""

import logging
import os
import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, List

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
    _db_instances.clear()
    _shared_pools.clear()


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
