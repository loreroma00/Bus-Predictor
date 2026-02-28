"""
Persistence Strategies - Abstract strategy patterns for persistence operations.
Contains cache strategies and saving strategies.
"""

import logging
import asyncio
import os
import pickle
from abc import ABC, abstractmethod
from typing import Optional

CACHE_FILE = "ledger_cache.pkl"
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
    def load(self, expected_md5: Optional[str] = None) -> dict | None:
        """Load ledger from cache. Returns None if not available."""
        pass

    @abstractmethod
    def save(self, ledger: dict, source_md5: Optional[str] = None):
        """Save ledger to cache."""
        pass


class FileCacheStrategy(BaseCacheStrategy):
    """File-based cache using pickle."""

    strategy_name = "file"

    def __init__(self, cache_path=CACHE_FILE):
        self.cache_path = cache_path

    def load(self, expected_md5: Optional[str] = None) -> dict | None:
        """Load ledger from pickle cache."""
        if not os.path.exists(self.cache_path):
            return None

        try:
            logging.info(f"Loading ledger from cache ({self.cache_path})...")
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)

            # MD5 Verification Logic
            if expected_md5:
                cached_md5 = cache.get("__metadata__", {}).get("md5")
                if cached_md5 != expected_md5:
                    logging.info(
                        f"Cache Outdated (Expected {expected_md5}, Found {cached_md5}). Invalidating..."
                    )
                    os.remove(self.cache_path)
                    return None

            return cache
        except Exception as e:
            logging.error(f"Failed to load cache: {e}. Rebuilding...")
            return None

    def save(self, ledger: dict, source_md5: Optional[str] = None):
        """Save ledger to pickle cache."""
        try:
            if source_md5:
                ledger["__metadata__"] = {"md5": source_md5}

            with open(self.cache_path, "wb") as f:
                pickle.dump(ledger, f)
        except Exception as e:
            logging.error(f"Could not save cache: {e}")

    def save_city_cache(self, city, cache_path=None):
        """
        Save city hexagons with street caches.
        Only saves the streets dict for each hexagon to minimize size.
        """
        if cache_path is None:
            cache_path = CITY_CACHE_FILE
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
            cache_path = CITY_CACHE_FILE
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

    def load(self, expected_md5: Optional[str] = None) -> dict | None:
        """Always returns None - no cache."""
        return None

    def save(self, ledger: dict, source_md5: Optional[str] = None):
        """Does nothing - no persistence."""
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
# Saving Strategy Pattern (Diaries)
# ============================================================
class saving_strategy(ABC):
    """Abstract base for diary saving strategies."""

    @abstractmethod
    def execute(self, diaries_list, filename):
        pass


class saving_parquet(saving_strategy):
    """Saves diaries to parquet format."""

    def execute(self, diaries_list, filename):
        # Import here to avoid circular dependency
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
        self.connection_string = connection_string
        self.table_name = table_name

    def execute(self, vectors_list, filename=None):
        """
        Execute database insert for vectors.

        Args:
            vectors_list: List of (Vector, Label, measurement_time) tuples
            filename: Ignored for database strategy
        """

        if not vectors_list:
            return

        # Use nest_asyncio to allow running from within existing event loops
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass  # Not installed, may fail if called from async context

        # Run async insert in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, create task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._async_execute(vectors_list)
                    )
                    future.result()  # Wait for completion
            else:
                loop.run_until_complete(self._async_execute(vectors_list))
        except RuntimeError:
            # No event loop exists, create one
            asyncio.run(self._async_execute(vectors_list))

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

            # Ensure connected
            if not db.pool:
                await db.connect()

            await db.insert_items(items)
