"""
Shared State - Singleton instances shared across API and Ingestor.

Both API and Ingestor access the same Observatory and City instances,
ensuring a single source of truth for GTFS data and hexagon weather.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

_observatory = None
_city = None
_city_name = "Rome"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_BUS_LANES_PATH = PROJECT_ROOT / "static_bus_lanes_roma.pkl"


def get_observatory(config: dict = None):
    """
    Get or create the singleton Observatory instance.

    If already created, returns existing instance.
    Otherwise, creates new instance with file cache.
    """
    global _observatory

    if _observatory is not None:
        return _observatory

    from application.domain.virtual_entities import Observatory
    from persistence.strategy import FileCacheStrategy

    logging.info("Initializing Observatory singleton...")
    cache_strategy = FileCacheStrategy(
        cache_dir=str(PROJECT_ROOT)
    )
    _observatory = Observatory(cache_strategy=cache_strategy, config=config)
    _observatory.get_topology()
    logging.info("Observatory initialized.")

    return _observatory


def get_city():
    """
    Get or create the singleton City instance.

    If already created, returns existing instance.
    Otherwise, creates new City("Rome") with static bus lanes.
    """
    global _city, _city_name

    if _city is not None:
        return _city

    from application.domain.cities import City

    logging.info("Initializing City singleton...")

    static_bus_lanes = {}
    if STATIC_BUS_LANES_PATH.exists():
        try:
            with open(STATIC_BUS_LANES_PATH, "rb") as f:
                static_bus_lanes = pickle.load(f)
            logging.info(f"Loaded {len(static_bus_lanes)} static bus lanes")
        except Exception as e:
            logging.warning(f"Could not load static bus lanes: {e}")

    _city = City(_city_name, static_bus_lanes)
    logging.info(f"City '{_city_name}' initialized.")

    return _city


def check_for_updates() -> bool:
    """
    Check for new GTFS data version.

    Returns True if ledger was updated.
    """
    global _observatory

    if _observatory is None:
        return False

    return _observatory.check_and_reload_ledger()


def reset():
    """
    Reset all singletons (useful for testing).
    """
    global _observatory, _city
    _observatory = None
    _city = None
