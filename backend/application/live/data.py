"""Compatibility runtime state for live ingestion.

Thread creation and GTFS-RT application no longer live here.  New code should
use ``ApplicationContext`` plus ``thread_loader.ThreadLoader``; this module only
keeps stable imports for console commands and older tests.
"""

from __future__ import annotations

import logging
import threading

from .. import domain as t
from .feed_fetcher import LiveFeedFetcher

SHUTDOWN_EVENT = threading.Event()
STOP_COLLECTION_EVENT = threading.Event()

OBSERVATORY: t.Observatory = None
CACHE_STRATEGY = None
CONFIG: dict = None

_feed_fetcher: LiveFeedFetcher | None = None


def initialize(
    observatory: t.Observatory,
    cache_strategy=None,
    config: dict = None,
    feed_fetcher: LiveFeedFetcher = None,
    stop_event: threading.Event = None,
    shutdown_event: threading.Event = None,
):
    """Bind legacy module-level references to the current runtime context."""
    global OBSERVATORY, CACHE_STRATEGY, CONFIG, _feed_fetcher
    global STOP_COLLECTION_EVENT, SHUTDOWN_EVENT

    OBSERVATORY = observatory
    CACHE_STRATEGY = cache_strategy
    CONFIG = config or {}
    if stop_event is not None:
        STOP_COLLECTION_EVENT = stop_event
    if shutdown_event is not None:
        SHUTDOWN_EVENT = shutdown_event

    _feed_fetcher = feed_fetcher or _build_feed_fetcher(CONFIG)
    logging.info("Live data compatibility state initialized.")


def get_feed_fetcher() -> LiveFeedFetcher:
    """Return the configured live feed fetcher, building one if needed."""
    global _feed_fetcher
    if _feed_fetcher is None:
        _feed_fetcher = _build_feed_fetcher(CONFIG or {})
    return _feed_fetcher


def get_realtime_updates(city_name: str = "Rome"):
    """Compatibility wrapper: fetch records and let Observatory ingest them."""
    if OBSERVATORY is None:
        return []
    records = get_feed_fetcher().fetch()
    return OBSERVATORY.ingest_live_feed(records, city_name=city_name)


def print_tracking_summary():
    """Compatibility no-op retained for old imports."""
    logging.info("Tracking summaries are emitted by ThreadLoader.")


def run_collection_loop():
    """Compatibility target for old callers."""
    context = _compat_context()
    if context is None:
        return
    from thread_loader import ThreadLoader

    ThreadLoader(context)._run_collection_loop()


def run_weather_loop(update_time: int = 900):
    """Compatibility target for old callers."""
    context = _compat_context()
    if context is None:
        return
    context.config.setdefault("timings", {})["update_interval"] = update_time
    from thread_loader import ThreadLoader

    ThreadLoader(context)._run_weather_loop()


def run_geocoding_loop():
    """Compatibility target for old callers."""
    context = _compat_context()
    if context is None:
        return
    from thread_loader import ThreadLoader

    ThreadLoader(context)._run_geocoding_loop()


def run_traffic_loop(update_interval: int = 300):
    """Compatibility target for old callers."""
    context = _compat_context()
    if context is None:
        return
    context.config.setdefault("timings", {})["traffic_update_interval"] = update_interval
    from thread_loader import ThreadLoader

    ThreadLoader(context)._run_traffic_loop()


def _build_feed_fetcher(config: dict) -> LiveFeedFetcher:
    """Build a LiveFeedFetcher from config."""
    urls = config.get("urls", {}) if config else {}
    return LiveFeedFetcher(
        vehicles_url=urls.get("rtgtfs_vehicles"),
        trips_url=urls.get("rtgtfs_trip_updates"),
    )


def _compat_context():
    """Build a minimal runtime context for old loop callers."""
    if OBSERVATORY is None:
        return None

    from application.runtime import ApplicationContext

    city = OBSERVATORY.get_city("Rome")
    return ApplicationContext(
        config=CONFIG or {},
        observatory=OBSERVATORY,
        city=city,
        cache_strategy=CACHE_STRATEGY,
        geocoding_service=getattr(OBSERVATORY, "_geocoding", None),
        traffic_service=getattr(city, "traffic_service", None) if city else None,
        feed_fetcher=get_feed_fetcher(),
        stop_event=STOP_COLLECTION_EVENT,
        shutdown_event=SHUTDOWN_EVENT,
    )
