"""
Persistence Strategies - Abstract strategy patterns for persistence operations.
Contains cache strategies and saving strategies.
"""

import logging
import asyncio
import os
import pickle
import threading
from abc import ABC, abstractmethod
from typing import Optional

TOPOLOGY_CACHE_FILE = "topology_cache.pkl"
SCHEDULE_CACHE_FILE = "schedule_cache.pkl"
CITY_CACHE_FILE = "city_cache.pkl"


# ============================================================
# Cache Strategy Pattern (Ledger)
# ============================================================
class BaseCacheStrategy(ABC):
    """
    Base class for cache strategies.
    Subclasses must define strategy_name for discovery.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Unique name for this strategy (used for lookup)."""
        pass

    @abstractmethod
    def load_topology(self, expected_md5: Optional[str] = None):
        """Load TopologyLedger from cache. Returns None if not available."""
        pass

    @abstractmethod
    def save_topology(self, topology, source_md5: Optional[str] = None):
        """Save TopologyLedger to cache."""
        pass

    @abstractmethod
    def load_schedule(self, expected_md5: Optional[str] = None):
        """Load ScheduleLedger from cache. Returns None if not available."""
        pass

    @abstractmethod
    def save_schedule(self, schedule_ledger, source_md5: Optional[str] = None):
        """Save ScheduleLedger to cache."""
        pass


class FileCacheStrategy(BaseCacheStrategy):
    """File-based cache using pickle."""

    strategy_name = "file"

    def __init__(self, cache_dir: str = "."):
        """Initialize the instance."""
        self._cache_dir = cache_dir
        self._topology_path = os.path.join(cache_dir, TOPOLOGY_CACHE_FILE)
        self._schedule_path = os.path.join(cache_dir, SCHEDULE_CACHE_FILE)

    # ---------- Topology ----------

    def load_topology(self, expected_md5: Optional[str] = None):
        """Load TopologyLedger from pickle cache."""
        return self._load(self._topology_path, expected_md5, "topology")

    def save_topology(self, topology, source_md5: Optional[str] = None):
        """Save TopologyLedger to pickle cache."""
        if source_md5:
            topology.source_md5 = source_md5
        self._save(self._topology_path, topology, "topology")

    # ---------- Schedule ----------

    def load_schedule(self, expected_md5: Optional[str] = None):
        """Load ScheduleLedger from pickle cache."""
        return self._load(self._schedule_path, expected_md5, "schedule")

    def save_schedule(self, schedule_ledger, source_md5: Optional[str] = None):
        """Save ScheduleLedger to pickle cache."""
        if source_md5:
            schedule_ledger.source_md5 = source_md5
        self._save(self._schedule_path, schedule_ledger, "schedule")

    # ---------- Internal ----------

    def _load(self, path: str, expected_md5: Optional[str], label: str):
        """Load."""
        if not os.path.exists(path):
            return None
        try:
            logging.info(f"Loading {label} from cache ({path})...")
            with open(path, "rb") as f:
                obj = pickle.load(f)

            if expected_md5 and hasattr(obj, "source_md5"):
                if obj.source_md5 != expected_md5:
                    logging.info(
                        f"Cache Outdated ({label}: expected {expected_md5}, "
                        f"found {obj.source_md5}). Invalidating..."
                    )
                    os.remove(path)
                    return None

            return obj
        except Exception as e:
            logging.error(f"Failed to load {label} cache: {e}. Rebuilding...")
            return None

    def _save(self, path: str, obj, label: str):
        """Save."""
        try:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        except Exception as e:
            logging.error(f"Could not save {label} cache: {e}")

    # ---------- City Cache (unchanged) ----------

    def save_city_cache(self, city, cache_path=None):
        """
        Save city hexagons with street caches.
        Only saves the streets dict for each hexagon to minimize size.
        """
        if cache_path is None:
            cache_path = os.path.join(self._cache_dir, CITY_CACHE_FILE)
        try:
            data = {
                hex_id: dict(hexagon.streets)
                for hex_id, hexagon in city.hexagons.items()
                if hexagon.streets  # Only save hexagons that have street data
            }
            if data:
                with open(cache_path, "wb") as f:
                    pickle.dump(data, f)
                logging.info(f"Saved city cache: {len(data)} hexagons with street data")
        except Exception as e:
            logging.error(f"Could not save city cache: {e}")

    def load_city_cache(self, city, cache_path=None):
        """
        Restore street caches to existing hexagons.
        Creates hexagons if they don't exist yet.
        """
        if cache_path is None:
            cache_path = os.path.join(self._cache_dir, CITY_CACHE_FILE)
        if not os.path.exists(cache_path):
            logging.info(f"No city cache found at {cache_path}")
            return 0
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            restored = 0
            for hex_id, streets in data.items():
                if hex_id not in city.hexagons:
                    # Import locally to avoid circular dependency
                    from application.domain.cities import Hexagon

                    city.hexagons[hex_id] = Hexagon(hex_id)
                city.hexagons[hex_id].streets = streets
                restored += len(streets)

            logging.info(
                f"Restored city cache: {len(data)} hexagons, {restored} street entries"
            )

            # DEBUG: Verify a sample hexagon was populated correctly
            if city.hexagons:
                sample_hex = next(iter(city.hexagons.values()))
                logging.debug(
                    f"  [CACHE DEBUG] Sample hex: {sample_hex.hex_id}, streets: {len(sample_hex.streets)}"
                )
                if sample_hex.streets:
                    sample_key = next(iter(sample_hex.streets.keys()))
                    logging.debug(
                        f"  [CACHE DEBUG] Sample key: {sample_key} -> {sample_hex.streets[sample_key]}"
                    )

            return restored
        except Exception as e:
            logging.error(f"Could not load city cache: {e}")
            return 0

    def get_city_cache_size(self, city) -> int:
        """Get total number of cached street entries for dirty checking."""
        return sum(len(h.streets) for h in city.hexagons.values())


class NoCacheStrategy(BaseCacheStrategy):
    """No caching - always rebuilds from source."""

    strategy_name = "none"

    def load_topology(self, expected_md5: Optional[str] = None):
        """Load the topology."""
        return None

    def save_topology(self, topology, source_md5: Optional[str] = None):
        """Save the topology."""
        pass

    def load_schedule(self, expected_md5: Optional[str] = None):
        """Load the schedule."""
        return None

    def save_schedule(self, schedule_ledger, source_md5: Optional[str] = None):
        """Save the schedule."""
        pass


def get_cache_strategy(name: str) -> BaseCacheStrategy:
    """
    Factory: Get a cache strategy by name.
    Discovers all subclasses of BaseCacheStrategy.
    """
    for subclass in BaseCacheStrategy.__subclasses__():
        if hasattr(subclass, "strategy_name") and subclass.strategy_name == name:
            return subclass()

    logging.warning(f"Unknown cache strategy '{name}'. Available: {get_available_strategies()}")
    logging.info("Defaulting to 'file'")
    return FileCacheStrategy()


def get_available_strategies() -> list:
    """List all available cache strategy names."""
    return [
        s.strategy_name
        for s in BaseCacheStrategy.__subclasses__()
        if hasattr(s, "strategy_name")
    ]


# ============================================================
# Shared DB Event Loop (for async DB operations from threads)
# ============================================================
_db_loop: asyncio.AbstractEventLoop | None = None
_db_loop_thread: threading.Thread | None = None
_db_loop_lock = threading.Lock()


def _run_db_loop(loop: asyncio.AbstractEventLoop):
    """Target for the DB event loop thread."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        loop.close()


def _get_db_loop() -> asyncio.AbstractEventLoop:
    """Get or create the shared event loop for all DB operations."""
    global _db_loop, _db_loop_thread
    if _db_loop is not None and _db_loop.is_running():
        return _db_loop
    with _db_loop_lock:
        # Double-check after acquiring lock
        if _db_loop is not None and _db_loop.is_running():
            return _db_loop
        _db_loop = asyncio.new_event_loop()
        _db_loop_thread = threading.Thread(
            target=_run_db_loop,
            args=(_db_loop,),
            daemon=True,
            name="db-event-loop",
        )
        _db_loop_thread.start()
        return _db_loop


def shutdown_db_loop():
    """Shutdown the shared DB event loop. Call during application exit."""
    global _db_loop, _db_loop_thread
    if _db_loop is None:
        return
    if _db_loop.is_running():
        _db_loop.call_soon_threadsafe(_db_loop.stop)
    if _db_loop_thread is not None:
        _db_loop_thread.join(timeout=5)
    _db_loop = None
    _db_loop_thread = None


# ============================================================
# Saving Strategy Pattern (Diaries)
# ============================================================
class saving_strategy(ABC):
    """Abstract base for diary saving strategies."""

    @abstractmethod
    def execute(self, diaries_list, filename):
        """Execute."""
        pass


class saving_parquet(saving_strategy):
    """Saves diaries to parquet format."""

    def execute(self, diaries_list, filename):
        # Import here to avoid circular dependency
        """Execute."""
        from .diaries import DIARIES_PATH, updateParquet, writeParquet
        import pandas as pd

        df = pd.DataFrame(diaries_list)

        # Ensure stop_sequence columns are integer
        for col in ["sequence", "stop_sequence"]:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
                )

        # Ensure directory exists
        os.makedirs(DIARIES_PATH, exist_ok=True)
        full_path = os.path.join(DIARIES_PATH, filename)

        if os.path.exists(full_path):
            appended = updateParquet(df, full_path)
            if appended:
                logging.info(f"Appended {len(df)} records to {full_path}")
        else:
            writeParquet(df, full_path)
            logging.info(f"Saved {len(df)} records to {full_path}")


class saving_database(saving_strategy):
    """
    Saves vectors to TimescaleDB.

    This strategy expects the diaries_list to contain (Vector, Label, time) tuples.
    """

    def __init__(self, connection_string: str = None, table_name: str = None):
        """Initialize the instance."""
        self.connection_string = connection_string
        self.table_name = table_name

    def execute(self, vectors_list, filename=None):
        """
        Execute database insert for vectors.

        Submits the async insert to a shared persistent event loop,
        avoiding the creation of throwaway event loops that leak
        connection pools.

        Args:
            vectors_list: List of (Vector, Label, measurement_time) tuples
            filename: Ignored for database strategy
        """
        if not vectors_list:
            return

        loop = _get_db_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._async_execute(vectors_list), loop
        )
        try:
            future.result(timeout=60)
        except Exception as e:
            logging.error(f"Database save failed: {e}")

    async def _async_execute(self, vectors_list):
        """Async execution of database insert."""
        from .database import get_db_connection
        from config import Prediction, Traffic
        from application.post_processing.vectorization import (
            PredictionVector,
            PredictionLabel,
            TrafficVector,
            TrafficLabel,
        )

        # Buckets for different types: key is (connection_string, table_name)
        buckets = {
            (Prediction.VECTOR_DB_CONNECTION, Prediction.VECTOR_TABLE): [],
            (Prediction.VECTOR_DB_CONNECTION, Prediction.LABEL_TABLE): [],
            (Traffic.VECTOR_DB_CONNECTION, Traffic.VECTOR_TABLE): [],
            (Traffic.VECTOR_DB_CONNECTION, Traffic.LABEL_TABLE): [],
        }

        for item in vectors_list:
            # item is (vector, label, time)
            if isinstance(item, tuple) and len(item) == 3:
                vec, lbl, _ = item  # time is expected to be in the objects now

                if isinstance(vec, PredictionVector):
                    buckets[(Prediction.VECTOR_DB_CONNECTION, Prediction.VECTOR_TABLE)].append(
                        vec
                    )
                elif isinstance(vec, TrafficVector):
                    buckets[(Traffic.VECTOR_DB_CONNECTION, Traffic.VECTOR_TABLE)].append(
                        vec
                    )

                if isinstance(lbl, PredictionLabel):
                    buckets[(Prediction.VECTOR_DB_CONNECTION, Prediction.LABEL_TABLE)].append(
                        lbl
                    )
                elif isinstance(lbl, TrafficLabel):
                    buckets[(Traffic.VECTOR_DB_CONNECTION, Traffic.LABEL_TABLE)].append(
                        lbl
                    )

        # Process each bucket
        for (conn_str, table), items in buckets.items():
            if not items:
                continue

            # Skip if config is missing (e.g. Traffic disabled or not configured)
            if not conn_str or not table:
                continue

            db = get_db_connection(connection_string=conn_str, table_name=table)

            # Ensure connected via get_valid_pool (handles loop checks)
            pool = await db.get_valid_pool()
            if not pool:
                logging.error(f"Failed to connect to {table}")
                continue

            await db.insert_items(items)
