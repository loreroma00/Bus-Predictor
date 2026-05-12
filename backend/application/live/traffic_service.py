"""
Traffic Service - Projects traffic data onto H3 hexagons with directional data.

Uses h3_utils for hexagon operations and cities.py for updating hexagon data.
Designed for dependency injection following the project's existing patterns.
Aggregates traffic by 8 cardinal directions (N, NE, E, SE, S, SW, W, NW).

Smart fetching: Only fetches tiles for hexagons with live trips, respecting TTL.
"""

import logging
import math
import time
from typing import TYPE_CHECKING

import mapbox_vector_tile as mvt
import requests

from application.domain import h3_utils
from application.domain.spatial_utils import (
    get_unique_tiles_from_hexagons,
    lat_lon_to_tile,
)

if TYPE_CHECKING:
    from application.domain.cities import City

TOMTOM_BASE_URL = "https://api.tomtom.com/traffic/map/4/tile/flow"
DEFAULT_ZOOM = 12
QPS_LIMIT = 10
REQUEST_DELAY = 1.0 / QPS_LIMIT
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
TILE_TTL_SECONDS = 900


class _TomTomTrafficClient:
    """Private TomTom MVT client used only by TrafficService."""

    def __init__(self, api_key: str, zoom: int = DEFAULT_ZOOM):
        """Store API credentials and rate-limit state."""
        self.api_key = api_key
        self.zoom = zoom
        self.base_url = TOMTOM_BASE_URL
        self._last_request_time = 0.0

    def get_tiles_for_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> list[tuple[int, int]]:
        """Calculate all tile coordinates covering a bounding box."""
        x_min, y_max = lat_lon_to_tile(min_lat, min_lon, self.zoom)
        x_max, y_min = lat_lon_to_tile(max_lat, max_lon, self.zoom)
        return [
            (x, y)
            for x in range(x_min, x_max + 1)
            for y in range(y_min, y_max + 1)
        ]

    def _rate_limit(self):
        """Enforce TomTom request rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def fetch_tile(self, x: int, y: int, flow_type: str = "absolute") -> dict | None:
        """Fetch and decode one TomTom traffic tile."""
        self._rate_limit()

        url = f"{self.base_url}/{flow_type}/{self.zoom}/{x}/{y}.pbf"
        try:
            response = requests.get(url, params={"key": self.api_key}, timeout=15)
            if response.status_code == 200:
                return mvt.decode(response.content)
            if response.status_code == 403:
                logging.error("Traffic API: Invalid API key or quota exceeded")
                return None
            if response.status_code == 404:
                return None
            logging.warning(
                "Traffic API: HTTP %s for tile (%s, %s)",
                response.status_code,
                x,
                y,
            )
            return None
        except requests.RequestException as e:
            logging.error("Traffic API: Request error for tile (%s, %s): %s", x, y, e)
            return None
        except Exception as e:
            logging.error("Traffic API: Decode error for tile (%s, %s): %s", x, y, e)
            return None


class TrafficService:
    """
    Projects traffic data from road segments onto H3 hexagons by direction.

    Owns traffic retrieval, decoding, projection, and city updates.

    Smart fetching: Only fetches tiles covering hexagons with buses,
    and only when tile data has expired (TTL-based).
    """

    def __init__(
        self,
        city: "City",
        api_key: str | None = None,
        zoom: int = DEFAULT_ZOOM,
        traffic_client=None,
    ):
        """
        Initialize the traffic service.

        Args:
            city: City instance to update traffic data on
            api_key: TomTom API key
            zoom: Tile zoom level
            traffic_client: optional private test double
        """
        if (
            traffic_client is None
            and api_key is not None
            and not isinstance(api_key, str)
        ):
            traffic_client = api_key
            api_key = None

        self._city = city
        self._traffic_client = traffic_client or _TomTomTrafficClient(
            api_key=api_key,
            zoom=zoom,
        )
        client_zoom = getattr(self._traffic_client, "zoom", None)
        self._zoom = client_zoom if isinstance(client_zoom, int) else zoom
        self._tile_last_update: dict[tuple[int, int], float] = {}  # Track tile TTLs
        self._paused_until = 0.0

    def pause(self, seconds: int):
        """
        Pause traffic updates for a specified duration.
        """
        self._paused_until = time.time() + seconds
        logging.info(f"Traffic Service paused for {seconds} seconds.")

    def _is_paused(self) -> bool:
        """Return True while the pause window set by ``pause`` has not elapsed."""
        return time.time() < self._paused_until

    def _is_tile_expired(
        self, tile: tuple[int, int], ttl: int = TILE_TTL_SECONDS
    ) -> bool:
        """Check if a tile's cached data has expired."""
        last_update = self._tile_last_update.get(tile, 0)
        return (time.time() - last_update) > ttl

    def _mark_tile_updated(self, tile: tuple[int, int]):
        """Mark a tile as freshly updated."""
        self._tile_last_update[tile] = time.time()

    def get_traffic_for_hexagon(
        self,
        hexagon_or_id,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return relative ratios and absolute speeds for one hexagon."""
        hex_id = self._resolve_hex_id(hexagon_or_id)
        if not hex_id:
            return ({}, {})

        hexagon = self._resolve_city_hexagon(hexagon_or_id, hex_id)
        if self._should_refresh_hexagon_traffic(hexagon):
            self.update_traffic_info([hex_id])
            hexagon = self._resolve_city_hexagon(hexagon_or_id, hex_id)

        if hexagon is None:
            return ({}, {})

        relative = {
            direction: hexagon.get_speed_ratio(direction) for direction in DIRECTIONS
        }
        absolute = {
            direction: hexagon.get_current_speed(direction) for direction in DIRECTIONS
        }
        return (relative, absolute)

    def update_traffic_info(self, hex_ids: list[str]) -> list[str]:
        """
        Core traffic update method. Fetches tiles and updates ALL hexagons covered.

        Flow: hex_ids → unique tiles → filter expired → fetch → update ALL hexagons in each tile

        Args:
            hex_ids: List of H3 hexagon IDs that need traffic data

        Returns:
            List of ALL hex IDs that were updated (includes hexagons covered by tiles)
        """
        if self._is_paused():
            logging.debug("Traffic Service is paused. Skipping update.")
            return []

        if not hex_ids:
            return []

        abs_tiles, rel_tiles = self._fetch_traffic_tiles(hex_ids)

        if not abs_tiles:
            logging.warning("  Traffic: No absolute tile data received")
            return []

        logging.debug(
            f"  Traffic: Received {len(abs_tiles)} absolute tiles, "
            f"{len(rel_tiles)} relative tiles"
        )

        # Extract road segments from both tile sets.
        all_abs_segments = []
        for tile_data in abs_tiles:
            segments = extract_road_segments(tile_data, flow_type="absolute")
            all_abs_segments.extend(segments)

        all_rel_segments = []
        for tile_data in rel_tiles:
            segments = extract_road_segments(tile_data, flow_type="relative")
            all_rel_segments.extend(segments)

        # Merge to get accurate speed data with direction.
        merged_segments = merge_absolute_relative_segments(
            all_abs_segments, all_rel_segments
        )

        logging.debug(f"  Traffic: Merged into {len(merged_segments)} complete segments")

        if not merged_segments:
            return []

        # Project segments to hexagons and aggregate speeds by direction.
        hex_dir_speeds = self._aggregate_speeds_per_hex_direction(merged_segments)

        # Update all hexagons covered by the tiles and reset their TTLs.
        all_updated_hex_ids = set()
        for (hex_id, direction), speed_data in hex_dir_speeds.items():
            current_speed = speed_data["avg_current_speed"]
            speed_ratio = speed_data["avg_ratio"]

            try:
                self._city.update_traffic(hex_id, direction, current_speed, speed_ratio)
                # Reset the hexagon's traffic TTL
                if hex_id in self._city.hexagons:
                    self._city.hexagons[hex_id].reset_traffic_ttl()
                all_updated_hex_ids.add(hex_id)
            except KeyError:
                # Hexagon doesn't exist in city yet - skip
                pass

        logging.debug(f"  Traffic: Updated {len(all_updated_hex_ids)} unique hexagons")
        return list(all_updated_hex_ids)

    def update_traffic(self) -> int:
        """
        Global loop: updates traffic for all hexagons with active live trips.

        Returns:
            Count of ALL updated hexagons (including tile coverage propagation)
        """
        hex_ids = self._city.get_hexagons_with_live_trips()

        if not hex_ids:
            logging.debug("  Traffic: No hexagons with live trips, skipping update")
            return 0

        logging.debug(
            "  Traffic: Updating traffic for %s hexagons with live trips",
            len(hex_ids),
        )
        updated = self.update_traffic_info(hex_ids)
        return len(updated)

    def update_traffic_for_hexagon(self, hex_id: str) -> list[str]:
        """
        On-demand: updates traffic for a specific hexagon (and all hexagons in covering tiles).

        Args:
            hex_id: The hexagon ID that triggered the update

        Returns:
            List of ALL updated hex IDs (from tile coverage)
        """
        logging.debug(f"  Traffic: Immediate fetch for hexagon {hex_id[:12]}...")
        return self.update_traffic_info([hex_id])

    def _fetch_traffic_tiles(
        self,
        hex_ids: list[str],
        only_expired: bool = True,
    ) -> tuple[list[dict], list[dict]]:
        """Fetch absolute and relative traffic tiles covering the given hexagons."""
        tiles = set(get_unique_tiles_from_hexagons(hex_ids, self._zoom))
        tiles_to_fetch = tiles
        if only_expired:
            tiles_to_fetch = {tile for tile in tiles if self._is_tile_expired(tile)}

        if not tiles_to_fetch:
            logging.debug(
                "  Traffic: All %s tiles are fresh, skipping fetch",
                len(tiles),
            )
            return ([], [])

        logging.debug(
            "  Traffic: Fetching %s/%s tile pairs at zoom %s...",
            len(tiles_to_fetch),
            len(tiles),
            self._zoom,
        )
        return self._fetch_tiles_dual(list(tiles_to_fetch))

    def _fetch_tiles_dual(
        self,
        tiles: list[tuple[int, int]],
    ) -> tuple[list[dict], list[dict]]:
        """Fetch both absolute and relative TomTom traffic tiles."""
        absolute_results = []
        relative_results = []
        total = len(tiles)

        for index, (x, y) in enumerate(tiles):
            if index > 0 and index % 5 == 0:
                logging.debug("  Traffic: Fetched %s/%s tile pairs...", index, total)

            abs_data = self._traffic_client.fetch_tile(x, y, flow_type="absolute")
            if abs_data:
                abs_data["_tile_x"] = x
                abs_data["_tile_y"] = y
                abs_data["_tile_zoom"] = self._zoom
                abs_data["_flow_type"] = "absolute"
                absolute_results.append(abs_data)
                self._mark_tile_updated((x, y))

            rel_data = self._traffic_client.fetch_tile(x, y, flow_type="relative")
            if rel_data:
                rel_data["_tile_x"] = x
                rel_data["_tile_y"] = y
                rel_data["_tile_zoom"] = self._zoom
                rel_data["_flow_type"] = "relative"
                relative_results.append(rel_data)

        return absolute_results, relative_results

    @staticmethod
    def _resolve_hex_id(hexagon_or_id) -> str:
        """Accept either a Hexagon object or a raw hex id string."""
        return getattr(hexagon_or_id, "hex_id", hexagon_or_id)

    def _resolve_city_hexagon(self, hexagon_or_id, hex_id: str):
        """Return the city hexagon object backing a raw id or provided hexagon."""
        if hasattr(hexagon_or_id, "get_speed_ratio") and hasattr(
            hexagon_or_id,
            "get_current_speed",
        ):
            return hexagon_or_id
        return getattr(self._city, "hexagons", {}).get(hex_id)

    def _should_refresh_hexagon_traffic(self, hexagon) -> bool:
        """Return True when a hexagon needs a traffic refresh."""
        if self._is_paused():
            return False
        if hexagon is None:
            return True
        is_expired = getattr(hexagon, "is_traffic_expired", None)
        if is_expired is None:
            return True
        return is_expired(TILE_TTL_SECONDS)

    def _project_segment_to_hexagons(
        self, coordinates: list[tuple[float, float]]
    ) -> list[str]:
        """
        Project a road segment's coordinates to H3 hexagon IDs.

        Uses h3_utils.get_h3_index for coordinate to hex conversion.

        Args:
            coordinates: List of (lat, lon) tuples along the road segment

        Returns:
            List of unique H3 hex IDs the segment passes through
        """
        hex_ids = set()

        for lat, lon in coordinates:
            # Use h3_utils to get hexagon ID at default resolution (9)
            hex_id = h3_utils.get_h3_index(lat, lon)
            hex_ids.add(hex_id)

        return list(hex_ids)

    def _aggregate_speeds_per_hex_direction(
        self, segments: list[dict]
    ) -> dict[tuple[str, str], dict]:
        """
        Aggregate road segment speeds per hexagon AND direction.

        Multiple roads may pass through a single hexagon in the same direction.
        We average their speeds to get a representative traffic speed per direction.

        Args:
            segments: List of merged segments with current_speed, speed_ratio, and direction

        Returns:
            Dict mapping (hex_id, direction) to {
                "avg_current_speed": float,
                "avg_flow_speed": float,
                "avg_ratio": float,
                "count": int
            }
        """
        hex_dir_data: dict[tuple[str, str], dict] = {}

        for segment in segments:
            current_speed = segment.get("current_speed", 0)
            free_flow_speed = segment.get("free_flow_speed", 0)
            speed_ratio = segment.get("speed_ratio", 1.0)
            direction = segment.get("direction", "N")
            coordinates = segment.get("coordinates", [])

            if current_speed <= 0 or not coordinates:
                continue

            # Get all hexagons this segment passes through
            hex_ids = self._project_segment_to_hexagons(coordinates)

            for hex_id in hex_ids:
                key = (hex_id, direction)
                if key not in hex_dir_data:
                    hex_dir_data[key] = {
                        "total_current": 0.0,
                        "total_flow": 0.0,
                        "total_ratio": 0.0,
                        "count": 0,
                    }

                hex_dir_data[key]["total_current"] += current_speed
                hex_dir_data[key]["total_flow"] += free_flow_speed
                hex_dir_data[key]["total_ratio"] += speed_ratio
                hex_dir_data[key]["count"] += 1

        # Calculate averages
        result = {}
        for key, data in hex_dir_data.items():
            if data["count"] > 0:
                result[key] = {
                    "avg_current_speed": data["total_current"] / data["count"],
                    "avg_flow_speed": data["total_flow"] / data["count"],
                    "avg_ratio": data["total_ratio"] / data["count"],
                    "count": data["count"],
                }

        return result


def extract_road_segments(tile_data: dict, flow_type: str = "absolute") -> list[dict]:
    """Extract road segments with speed data from a decoded TomTom tile."""
    segments = []
    flow_layer = tile_data.get("Traffic flow")
    if not flow_layer:
        return segments

    tile_x = tile_data.get("_tile_x", 0)
    tile_y = tile_data.get("_tile_y", 0)
    tile_zoom = tile_data.get("_tile_zoom", DEFAULT_ZOOM)

    for feature in flow_layer.get("features", []):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        traffic_level = properties.get("traffic_level", 0)

        if geometry.get("type") != "LineString" or traffic_level <= 0:
            continue

        coords = geometry.get("coordinates", [])
        geo_coords = _tile_coords_to_latlon(coords, tile_x, tile_y, tile_zoom)
        segment_key = _make_segment_key(geo_coords)
        bearing, direction = _calculate_segment_direction(geo_coords)

        segment = {
            "segment_key": segment_key,
            "coordinates": geo_coords,
            "bearing": bearing,
            "direction": direction,
        }
        if flow_type == "absolute":
            segment["speed_kph"] = traffic_level
        else:
            segment["speed_ratio"] = traffic_level
        segments.append(segment)

    return segments


def merge_absolute_relative_segments(
    absolute_segments: list[dict],
    relative_segments: list[dict],
) -> list[dict]:
    """Merge absolute speed and relative congestion data by segment key."""
    relative_by_key = {segment["segment_key"]: segment for segment in relative_segments}

    merged = []
    for abs_segment in absolute_segments:
        key = abs_segment["segment_key"]
        rel_segment = relative_by_key.get(key)
        current_speed = abs_segment.get("speed_kph", 0)

        if rel_segment and rel_segment.get("speed_ratio", 0) > 0:
            speed_ratio = rel_segment["speed_ratio"]
            free_flow_speed = current_speed / speed_ratio
        else:
            speed_ratio = 0.8
            free_flow_speed = current_speed * 1.2

        merged.append(
            {
                "segment_key": key,
                "coordinates": abs_segment["coordinates"],
                "current_speed": current_speed,
                "free_flow_speed": free_flow_speed,
                "speed_ratio": speed_ratio,
                "direction": abs_segment.get("direction", "N"),
                "bearing": abs_segment.get("bearing", 0.0),
            }
        )

    return merged


def _calculate_segment_direction(
    coords: list[tuple[float, float]],
) -> tuple[float, str]:
    """Calculate the overall bearing and cardinal direction of a road segment."""
    from application.domain.spatial_utils import derive_bearing, get_cardinal_direction

    if len(coords) < 2:
        return (0.0, "N")

    start = coords[0]
    end = coords[-1]
    bearing = derive_bearing(start[0], start[1], end[0], end[1])
    if bearing < 0:
        bearing = 0.0

    return (bearing, get_cardinal_direction(bearing))


def _make_segment_key(coords: list[tuple[float, float]]) -> str:
    """Create a hashable key from coordinates for segment matching."""
    if not coords:
        return ""
    first = coords[0]
    last = coords[-1]
    return (
        f"{round(first[0], 5)},{round(first[1], 5)}_"
        f"{round(last[0], 5)},{round(last[1], 5)}"
    )


def _tile_coords_to_latlon(
    coords: list,
    tile_x: int,
    tile_y: int,
    zoom: int,
    extent: int = 4096,
) -> list[tuple[float, float]]:
    """Convert MVT tile-local coordinates to latitude/longitude."""
    result = []
    n = 2**zoom

    for coord in coords:
        if len(coord) < 2:
            continue

        px, py = coord[0], coord[1]
        fx = tile_x + (px / extent)
        fy = tile_y + (py / extent)
        lon = fx / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * fy / n)))
        lat = math.degrees(lat_rad)
        result.append((lat, lon))

    return result


def create_traffic_service(
    city: "City", api_key: str, zoom: int = DEFAULT_ZOOM
) -> TrafficService:
    """
    Factory function to create a TrafficService.

    Args:
        city: City instance to update
        api_key: TomTom API key
        zoom: Tile zoom level (default 12)

    Returns:
        Configured TrafficService instance
    """
    return TrafficService(city, api_key=api_key, zoom=zoom)
