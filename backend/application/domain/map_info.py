"""Geocoding services — sync and async Nominatim wrappers with per-hex street caching."""

import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from functools import lru_cache
import queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cities import City


# User agent is required by Nominatim usage policy
# CUSTOM CONFIGURATION
NOMINATIM_DOMAIN = "192.168.1.96:1213"
NOMINATIM_SCHEME = "http"

geolocator = Nominatim(
    user_agent="thesis_project_real_bus_tracker",
    domain=NOMINATIM_DOMAIN,
    scheme=NOMINATIM_SCHEME,
)

# Create a rate-limited wrapper (1 request per second) ONLY for official server
if "nominatim.openstreetmap.org" in NOMINATIM_DOMAIN:
    geocode_throttled = RateLimiter(geolocator.reverse, min_delay_seconds=1.0)
else:
    # No rate limit for local/custom server
    geocode_throttled = geolocator.reverse


# Coordinates are rounded to 5 decimal places (~1m precision) for better cache hits.
# This prevents near-duplicate requests from GPS jitter.
@lru_cache(maxsize=1000)
def _get_street_name_cached(lat: float, lon: float) -> str:
    """
    Internal cached function - expects already-rounded coordinates.
    """
    try:
        # RateLimiter handles the 1 req/s delay automatically (if active).
        # This is CRITICAL to avoid getting banned by official Nominatim.
        location = geocode_throttled((lat, lon), exactly_one=True, addressdetails=True)

        if location and location.raw and "address" in location.raw:
            address = location.raw["address"]
            # Try to find the most relevant "street" field
            return (
                address.get("road")
                or address.get("pedestrian")
                or address.get("suburb")
                or "Unknown Street"
            )

        return "Unknown Location"
    except Exception as e:
        logging.error(f"Geocoding error: {e}")
        return "Unknown"


def get_street_name(lat: float, lon: float) -> str:
    """
    Returns the street name for a given latitude and longitude.

    Coordinates are rounded to 5 decimal places (~1m precision)
    to maximize cache hits from GPS jitter.
    """
    # Round to ensure consistent cache keys
    rounded_lat = round(lat, 5)
    rounded_lon = round(lon, 5)
    return _get_street_name_cached(rounded_lat, rounded_lon)


class SyncGeocodingService:
    """
    Synchronous geocoding service for fast/local servers.

    Implements GeocodingStrategy Protocol for dependency injection.
    Resolves streets immediately inline (no queue, no background thread).
    Use this when your Nominatim server has no rate limits.
    """

    def __init__(self, city: "City"):
        """Bind the service to the ``City`` whose hexagons cache the resolved streets."""
        self._city = city

    def enqueue(self, lat: float, lon: float, hex_id: str) -> None:
        """
        Immediately resolve and cache the street name (synchronous).
        Called 'enqueue' for interface compatibility, but doesn't actually queue.
        """
        key = (round(lat, 5), round(lon, 5))

        # Skip if already cached
        hexagon = self._city.get_hexagon(hex_id)
        if hexagon and key in hexagon.streets:
            return

        # Resolve immediately
        street = get_street_name(lat, lon)
        if hexagon:
            hexagon.add_street(lat, lon, street)

    def get_street(self, lat: float, lon: float) -> str | None:
        """Get cached street name from hexagon."""
        key = (round(lat, 5), round(lon, 5))
        hex_id = self._city.get_hex_id(lat, lon)
        hexagon = self._city.get_hexagon(hex_id)
        return hexagon.get_street_by_coords(*key) if hexagon else None

    def process_one(self) -> bool:
        """No-op for sync service (nothing to process)."""
        return False

    def get_and_reset_resolved_count(self) -> int:
        """No tracking needed for sync service."""
        return 0

    def get_queue_size(self) -> int:
        """Always 0 for sync service."""
        return 0


class AsyncGeocodingService:
    """
    Background geocoding service with queue for rate-limited servers.

    Implements GeocodingStrategy Protocol for dependency injection.
    Uses a queue to process geocoding requests at a controlled rate.
    Use this when connecting to official Nominatim (1 req/s limit).
    """

    def __init__(self, city: "City"):
        """Bind the service to a ``City`` and initialise the async resolution queue."""
        self._city = city
        self._queue: queue.Queue[tuple[float, float, str]] = queue.Queue()
        self._pending: set[tuple[float, float]] = set()
        self._resolved_since_last_report = 0

    def enqueue(self, lat: float, lon: float, hex_id: str) -> None:
        """Add coords to resolution queue (deduplicated)."""
        key = (round(lat, 5), round(lon, 5))

        # Skip if already pending or cached
        if key in self._pending:
            return
        hexagon = self._city.get_hexagon(hex_id)
        if hexagon and key in hexagon.streets:
            return

        self._pending.add(key)
        self._queue.put((key[0], key[1], hex_id))

    def get_street(self, lat: float, lon: float) -> str | None:
        """Get cached street name, or None if not yet resolved."""
        key = (round(lat, 5), round(lon, 5))
        hex_id = self._city.get_hex_id(lat, lon)
        hexagon = self._city.get_hexagon(hex_id)
        return hexagon.get_street_by_coords(*key) if hexagon else None

    def process_one(self) -> bool:
        """Process one item from queue. Returns False if empty."""
        lat, lon, hex_id = None, None, None
        try:
            lat, lon, hex_id = self._queue.get_nowait()
            street = get_street_name(lat, lon)
            hexagon = self._city.get_hexagon(hex_id)
            if hexagon:
                hexagon.add_street(lat, lon, street)
                self._resolved_since_last_report += 1
            return True
        except queue.Empty:
            return False
        finally:
            if lat is not None and lon is not None:
                self._pending.discard((lat, lon))

    def get_and_reset_resolved_count(self) -> int:
        """Get resolved count since last call and reset."""
        count = self._resolved_since_last_report
        self._resolved_since_last_report = 0
        return count

    def get_queue_size(self) -> int:
        """Returns current queue size."""
        return self._queue.qsize()


def create_geocoding_service(city: "City"):
    """
    Factory function to create appropriate geocoding service based on server config.

    Returns SyncGeocodingService for local/fast servers (no rate limit).
    Returns AsyncGeocodingService for official Nominatim (rate limited).
    """
    if "nominatim.openstreetmap.org" in NOMINATIM_DOMAIN:
        logging.info("📍 Using ASYNC geocoding (rate-limited for official Nominatim)")
        return AsyncGeocodingService(city)
    else:
        logging.info(f"📍 Using SYNC geocoding (fast mode for {NOMINATIM_DOMAIN})")
        return SyncGeocodingService(city)
