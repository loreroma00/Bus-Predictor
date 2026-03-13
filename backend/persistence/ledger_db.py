"""
Ledger Database Layer — async writes and sync reads for ledger tables.

Tables:
  historical_arrivals  — observed stop arrivals (from diaries)
  predicted_arrivals   — model-predicted stop arrivals
  vehicle_trips        — per-vehicle trip performance summaries
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, List, Optional

import pandas as pd

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None
    ASYNCPG_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
#  Async writer (used from sync callers via shared event loop)
# ============================================================

class LedgerDBWriter:
    """Batched async writer for a single ledger table."""

    def __init__(self, connection_string: str, table_name: str, batch_size: int = 50):
        self._conn_str = connection_string
        self._table_name = table_name
        self._batch_size = batch_size
        self._pool: Optional[Any] = None

    async def _ensure_pool(self):
        if self._pool is not None:
            return self._pool
        if not ASYNCPG_AVAILABLE:
            logger.error("asyncpg not available — ledger writes disabled")
            return None
        try:
            self._pool = await asyncpg.create_pool(
                self._conn_str, min_size=1, max_size=5, command_timeout=30,
            )
        except Exception as e:
            logger.error(f"Failed to create pool for {self._table_name}: {e}")
        return self._pool

    async def insert_rows(self, query: str, rows: list[tuple]):
        """Execute an INSERT … VALUES for a batch of rows."""
        pool = await self._ensure_pool()
        if not pool:
            return
        try:
            async with pool.acquire() as conn:
                await conn.executemany(query, rows)
            logger.debug(f"Inserted {len(rows)} rows into {self._table_name}")
        except Exception as e:
            logger.error(f"Ledger insert failed ({self._table_name}): {e}")

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None


# Singleton writers keyed by table name
_writers: dict[str, LedgerDBWriter] = {}


def _get_writer(connection_string: str, table_name: str) -> LedgerDBWriter:
    if table_name not in _writers:
        _writers[table_name] = LedgerDBWriter(connection_string, table_name)
    return _writers[table_name]


def _get_loop():
    """Borrow the shared DB event loop from persistence.strategy."""
    from .strategy import _get_db_loop
    return _get_db_loop()


# ============================================================
#  Sync façade — fire-and-forget write from any thread
# ============================================================

def write_historical(connection_string: str, table_name: str, arrivals: list[dict]):
    """Write StopArrival records to the database (sync, thread-safe)."""
    if not arrivals:
        return
    writer = _get_writer(connection_string, table_name)
    query = f"""
        INSERT INTO {table_name} (
            trip_id, stop_id, stop_sequence,
            actual_arrival_time, schedule_adherence, occupancy_status,
            vehicle_id
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT DO NOTHING
    """
    rows = [
        (
            a["trip_id"],
            a["stop_id"],
            int(a["stop_sequence"]),
            float(a["actual_arrival_time"]),
            float(a["schedule_adherence"]),
            int(a["occupancy_status"]),
            str(a.get("vehicle_id", "")),
        )
        for a in arrivals
    ]
    loop = _get_loop()
    asyncio.run_coroutine_threadsafe(writer.insert_rows(query, rows), loop)


def write_predicted(connection_string: str, table_name: str, predictions: list[dict]):
    """Write StopPredictionRecord records to the database (sync, thread-safe)."""
    if not predictions:
        return
    writer = _get_writer(connection_string, table_name)
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
            p["trip_date"],
            p["scheduled_start"],
            p["stop_id"],
            int(p["stop_sequence"]),
            p["predicted_arrival"],
            float(p["predicted_delay_sec"]),
            int(p["predicted_crowd_level"]),
            float(p["prediction_timestamp"]),
        )
        for p in predictions
    ]
    loop = _get_loop()
    asyncio.run_coroutine_threadsafe(writer.insert_rows(query, rows), loop)


def write_vehicle_trips(connection_string: str, table_name: str, records: list[dict]):
    """Write VehicleTripRecord records to the database (sync, thread-safe)."""
    if not records:
        return
    writer = _get_writer(connection_string, table_name)
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
            r["vehicle_id"], r["trip_id"], r["route_id"], int(r["direction_id"]),
            r["vehicle_type_name"], int(r["fuel_type"]), int(r["euro_class"]),
            int(r["capacity_total"]),
            r["trip_date"], r["scheduled_start"],
            float(r["actual_start_time"]), float(r["trip_end_time"]),
            float(r["trip_duration_sec"]),
            float(r["mean_delay_sec"]), float(r["median_delay_sec"]),
            float(r["max_delay_sec"]), float(r["min_delay_sec"]),
            float(r["std_delay_sec"]),
            float(r["mean_occupancy"]), int(r["max_occupancy"]),
            int(r["measurement_count"]), float(r["preferential_ratio"]),
            float(r["recorded_at"]),
        )
        for r in records
    ]
    loop = _get_loop()
    asyncio.run_coroutine_threadsafe(writer.insert_rows(query, rows), loop)


# ============================================================
#  Sync readers (for preprocessing / analytics)
# ============================================================

def read_historical(
    connection_string: str,
    table_name: str,
    trip_id: str = None,
    date_start: float = None,
    date_end: float = None,
) -> pd.DataFrame:
    """Read historical arrivals via SQLAlchemy (sync)."""
    engine = _get_sync_engine(connection_string)
    if engine is None:
        return pd.DataFrame()

    where = []
    if trip_id:
        where.append(f"trip_id = '{trip_id}'")
    if date_start is not None:
        where.append(f"actual_arrival_time >= {date_start}")
    if date_end is not None:
        where.append(f"actual_arrival_time < {date_end}")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    query = f"SELECT * FROM {table_name}{clause}"
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"Failed to read {table_name}: {e}")
        return pd.DataFrame()


def read_predicted(
    connection_string: str,
    table_name: str,
    route_id: str = None,
    trip_date: str = None,
) -> pd.DataFrame:
    """Read predicted arrivals via SQLAlchemy (sync)."""
    engine = _get_sync_engine(connection_string)
    if engine is None:
        return pd.DataFrame()

    where = []
    if route_id:
        where.append(f"route_id = '{route_id}'")
    if trip_date:
        where.append(f"trip_date = '{trip_date}'")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    query = f"SELECT * FROM {table_name}{clause}"
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"Failed to read {table_name}: {e}")
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
    """Read vehicle trip records via SQLAlchemy (sync)."""
    engine = _get_sync_engine(connection_string)
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
        logger.error(f"Failed to read {table_name}: {e}")
        return pd.DataFrame()


# ============================================================
#  Internal helpers
# ============================================================

_engines: dict[str, Any] = {}


def _get_sync_engine(connection_string: str):
    """Get or create a sync SQLAlchemy engine."""
    if connection_string in _engines:
        return _engines[connection_string]
    from .database import get_sync_engine
    engine = get_sync_engine(connection_string)
    if engine is not None:
        _engines[connection_string] = engine
    return engine


async def close_ledger_writers():
    """Close all ledger writer pools."""
    for writer in _writers.values():
        await writer.close()
    _writers.clear()
