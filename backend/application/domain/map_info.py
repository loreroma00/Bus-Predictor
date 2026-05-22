"""Geocoding services: sync and async Nominatim wrappers with per-hex caching."""

from __future__ import annotations

import logging
import queue
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

if TYPE_CHECKING:
    from .cities import City

DEFAULT_NOMINATIM_SCHEME = "https"


def _parse_nominatim_url(nominatim_url: str) -> tuple[str, str]:
    """Return ``(scheme, domain)`` for geopy from a configured URL."""
    parsed = urlparse(nominatim_url)
    if not parsed.scheme:
        parsed = urlparse(f"{DEFAULT_NOMINATIM_SCHEME}://{nominatim_url}")

    domain = parsed.netloc or parsed.path
    if not domain:
        raise ValueError("Nominatim URL must include a host")

    return parsed.scheme, domain.rstrip("/")


def _street_from_location(location) -> str:
    """Extract the best street-like label from a Nominatim result."""
    if location and location.raw and "address" in location.raw:
        address = location.raw["address"]
        return (
            address.get("road")
            or address.get("pedestrian")
            or address.get("suburb")
            or "Unknown Street"
        )
    return "Unknown Location"


class StreetNameResolver:
    """Small cached reverse-geocoder configured at runtime."""

    def __init__(self, nominatim_url: str, user_agent: str):
        """Create the geopy client and local coordinate cache."""
        scheme, domain = _parse_nominatim_url(nominatim_url)
        self.domain = domain
        geolocator = Nominatim(
            user_agent=user_agent,
            domain=domain,
            scheme=scheme,
        )
        if "nominatim.openstreetmap.org" in domain:
            self._reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1.0)
        else:
            self._reverse = geolocator.reverse
        self._cache: dict[tuple[float, float], str] = {}

    @property
    def is_rate_limited(self) -> bool:
        """Return whether this resolver targets the official Nominatim service."""
        return "nominatim.openstreetmap.org" in self.domain

    def get_street_name(self, lat: float, lon: float) -> str:
        """Resolve a rounded coordinate pair into a street-like label."""
        key = (round(lat, 5), round(lon, 5))
        if key in self._cache:
            return self._cache[key]

        try:
            location = self._reverse(key, exactly_one=True, addressdetails=True)
            street = _street_from_location(location)
        except Exception as exc:
            logging.error("Geocoding error: %s", exc)
            street = "Unknown"

        self._cache[key] = street
        return street


class SyncGeocodingService:
    """
    Synchronous geocoding service for fast/local servers.

    Implements GeocodingStrategy Protocol for dependency injection.
    Resolves streets immediately inline (no queue, no background thread).
    """

    def __init__(self, city: "City", resolver: StreetNameResolver):
        """Bind the service to the city whose hexagons cache street names."""
        self._city = city
        self._resolver = resolver

    def enqueue(self, lat: float, lon: float, hex_id: str) -> None:
        """
        Immediately resolve and cache the street name.

        The method name matches the geocoding strategy protocol.
        """
        key = (round(lat, 5), round(lon, 5))

        hexagon = self._city.get_hexagon(hex_id)
        if hexagon and key in hexagon.streets:
            return

        street = self._resolver.get_street_name(lat, lon)
        if hexagon:
            hexagon.add_street(lat, lon, street)

    def get_street(self, lat: float, lon: float) -> str | None:
        """Get cached street name from the backing hexagon."""
        key = (round(lat, 5), round(lon, 5))
        hex_id = self._city.get_hex_id(lat, lon)
        hexagon = self._city.get_hexagon(hex_id)
        return hexagon.get_street_by_coords(*key) if hexagon else None

    def process_one(self) -> bool:
        """No-op for sync service."""
        return False

    def get_and_reset_resolved_count(self) -> int:
        """No background work is tracked for sync service."""
        return 0

    def get_queue_size(self) -> int:
        """Always zero for sync service."""
        return 0


class AsyncGeocodingService:
    """
    Background geocoding service with queue for rate-limited servers.

    Implements GeocodingStrategy Protocol for dependency injection.
    """

    def __init__(self, city: "City", resolver: StreetNameResolver):
        """Bind the service to a city and initialise the async queue."""
        self._city = city
        self._resolver = resolver
        self._queue: queue.Queue[tuple[float, float, str]] = queue.Queue()
        self._pending: set[tuple[float, float]] = set()
        self._resolved_since_last_report = 0

    def enqueue(self, lat: float, lon: float, hex_id: str) -> None:
        """Add coords to the resolution queue, deduplicated by rounded coords."""
        key = (round(lat, 5), round(lon, 5))

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
        """Process one queued coordinate. Returns False if the queue is empty."""
        lat, lon, hex_id = None, None, None
        try:
            lat, lon, hex_id = self._queue.get_nowait()
            street = self._resolver.get_street_name(lat, lon)
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
        """Return and reset the number of resolved queue items."""
        count = self._resolved_since_last_report
        self._resolved_since_last_report = 0
        return count

    def get_queue_size(self) -> int:
        """Return current queue size."""
        return self._queue.qsize()


def create_geocoding_service(
    city: "City",
    nominatim_url: str,
    user_agent: str,
):
    """Create a geocoding service from runtime configuration."""
    resolver = StreetNameResolver(nominatim_url=nominatim_url, user_agent=user_agent)
    if resolver.is_rate_limited:
        logging.info("Using ASYNC geocoding for rate-limited Nominatim")
        return AsyncGeocodingService(city, resolver)

    logging.info("Using SYNC geocoding for %s", resolver.domain)
    return SyncGeocodingService(city, resolver)
