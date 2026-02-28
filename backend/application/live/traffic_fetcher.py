"""
TomTom Traffic Fetcher - Fetches and decodes Traffic Flow Vector Tiles.

Uses TomTom's Traffic Flow Tiles API with MVT/PBF format.
Supports both 'absolute' (speed in kph) and 'relative' (speed as fraction of free-flow).
Rate limited to 10 QPS per TomTom API requirements.
"""

import logging
from application.domain.spatial_utils import lat_lon_to_tile

import math
import time
import requests
import mapbox_vector_tile as mvt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Constants
TOMTOM_BASE_URL = "https://api.tomtom.com/traffic/map/4/tile/flow"
DEFAULT_ZOOM = 12
QPS_LIMIT = 10
REQUEST_DELAY = 1.0 / QPS_LIMIT  # 100ms between requests


class TomTomTrafficFetcher:
    """
    Fetches and decodes TomTom Traffic Flow Vector Tiles.

    Supports both flow types:
    - 'absolute': Returns current speed in kph
    - 'relative': Returns speed as fraction of free-flow (0.0-1.0)

    Rate limited to respect TomTom's 10 QPS limit across all requests.
    """

    def __init__(self, api_key: str, zoom: int = DEFAULT_ZOOM):
        """
        Initialize the fetcher.

        Args:
            api_key: TomTom API key
            zoom: Tile zoom level (default 12 for city-level detail)
        """
        self.api_key = api_key
        self.zoom = zoom
        self.base_url = TOMTOM_BASE_URL
        self._last_request_time = 0.0

    def get_tiles_for_bbox(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> list[tuple[int, int]]:
        """
        Calculate all tile coordinates covering a bounding box.

        Args:
            min_lat, min_lon: Southwest corner
            max_lat, max_lon: Northeast corner

        Returns:
            List of (x, y) tile coordinates
        """
        # Get corner tiles
        x_min, y_max = lat_lon_to_tile(min_lat, min_lon, self.zoom)
        x_max, y_min = lat_lon_to_tile(max_lat, max_lon, self.zoom)

        # Generate all tiles in the range
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tiles.append((x, y))

        return tiles

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def fetch_tile(self, x: int, y: int, flow_type: str = "absolute") -> dict | None:
        """
        Fetch and decode a single traffic tile.

        Args:
            x, y: Tile coordinates
            flow_type: 'absolute' for speed in kph, 'relative' for fraction of free-flow

        Returns:
            Decoded MVT data as dict with layers, or None on error.
        """
        self._rate_limit()

        url = f"{self.base_url}/{flow_type}/{self.zoom}/{x}/{y}.pbf"
        params = {"key": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                # Decode MVT/PBF format
                decoded = mvt.decode(response.content)
                return decoded
            elif response.status_code == 403:
                logging.error("Traffic API: Invalid API key or quota exceeded")
                return None
            elif response.status_code == 404:
                # No data for this tile (ocean, etc.) - not an error
                return None
            else:
                logging.warning(f"Traffic API: HTTP {response.status_code} for tile ({x}, {y})")
                return None

        except requests.RequestException as e:
            logging.error(f"Traffic API: Request error for tile ({x}, {y}): {e}")
            return None
        except Exception as e:
            logging.error(f"Traffic API: Decode error for tile ({x}, {y}): {e}")
            return None

    def fetch_tiles_dual(
        self, tiles: list[tuple[int, int]]
    ) -> tuple[list[dict], list[dict]]:
        """
        Fetch both absolute and relative tiles for all coordinates.

        Fetches tiles in interleaved order to maximize data freshness:
        For each tile, fetches absolute then relative before moving to next.

        Rate limiting is shared across all requests (10 QPS total).

        Args:
            tiles: List of (x, y) tile coordinates

        Returns:
            Tuple of (absolute_tiles, relative_tiles) lists
        """
        absolute_results = []
        relative_results = []
        total = len(tiles)

        for i, (x, y) in enumerate(tiles):
            if i > 0 and i % 5 == 0:
                # Progress every 5 tiles (10 requests due to dual fetch)
                logging.debug(f"  Traffic: Fetched {i}/{total} tile pairs...")

            # Fetch absolute (current speed in kph)
            abs_data = self.fetch_tile(x, y, flow_type="absolute")
            if abs_data:
                abs_data["_tile_x"] = x
                abs_data["_tile_y"] = y
                abs_data["_tile_zoom"] = self.zoom
                abs_data["_flow_type"] = "absolute"
                absolute_results.append(abs_data)

            # Fetch relative (fraction of free-flow speed)
            rel_data = self.fetch_tile(x, y, flow_type="relative")
            if rel_data:
                rel_data["_tile_x"] = x
                rel_data["_tile_y"] = y
                rel_data["_tile_zoom"] = self.zoom
                rel_data["_flow_type"] = "relative"
                relative_results.append(rel_data)

        return absolute_results, relative_results


def extract_road_segments(tile_data: dict, flow_type: str = "absolute") -> list[dict]:
    """
    Extract road segments with speed data from decoded tile.

    Args:
        tile_data: Decoded MVT tile data
        flow_type: 'absolute' or 'relative' - affects how traffic_level is interpreted

    Returns:
        List of road segments with geometry, speed/ratio, and direction
    """
    segments = []

    # TomTom uses "Traffic flow" layer
    flow_layer = tile_data.get("Traffic flow")
    if not flow_layer:
        return segments

    tile_x = tile_data.get("_tile_x", 0)
    tile_y = tile_data.get("_tile_y", 0)
    tile_zoom = tile_data.get("_tile_zoom", DEFAULT_ZOOM)

    features = flow_layer.get("features", [])

    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        # traffic_level interpretation depends on flow_type:
        # - absolute: speed in kph
        # - relative: fraction of free-flow (0.0 to 1.0+)
        traffic_level = properties.get("traffic_level", 0)

        if geometry.get("type") == "LineString" and traffic_level > 0:
            # Convert tile-local coordinates to lat/lon
            coords = geometry.get("coordinates", [])
            geo_coords = _tile_coords_to_latlon(coords, tile_x, tile_y, tile_zoom)

            # Create a unique segment key based on geometry for matching
            segment_key = _make_segment_key(geo_coords)

            # Calculate bearing/direction from segment geometry
            bearing, direction = _calculate_segment_direction(geo_coords)

            segment = {
                "segment_key": segment_key,
                "coordinates": geo_coords,  # List of (lat, lon) tuples
                "bearing": bearing,
                "direction": direction,  # N, NE, E, SE, S, SW, W, NW
            }

            if flow_type == "absolute":
                segment["speed_kph"] = traffic_level
            else:  # relative
                segment["speed_ratio"] = traffic_level  # e.g., 0.6 = 60% of free-flow

            segments.append(segment)

    return segments


def _calculate_segment_direction(
    coords: list[tuple[float, float]],
) -> tuple[float, str]:
    """
    Calculate the overall bearing and cardinal direction of a road segment.

    Uses the first and last coordinates to determine overall direction.

    Returns:
        Tuple of (bearing in degrees, cardinal direction string)
    """
    from application.domain.spatial_utils import derive_bearing, get_cardinal_direction

    if len(coords) < 2:
        return (0.0, "N")  # Default to North if insufficient points

    # Use first and last point to get overall direction
    start = coords[0]
    end = coords[-1]

    bearing = derive_bearing(start[0], start[1], end[0], end[1])

    # If bearing is invalid (-1 for very short segments), default to North
    if bearing < 0:
        bearing = 0.0

    direction = get_cardinal_direction(bearing)
    return (bearing, direction)


def _make_segment_key(coords: list[tuple[float, float]]) -> str:
    """Create a hashable key from coordinates for segment matching."""
    if not coords:
        return ""
    # Use first and last points rounded to 5 decimals
    first = coords[0]
    last = coords[-1]
    return f"{round(first[0], 5)},{round(first[1], 5)}_{round(last[0], 5)},{round(last[1], 5)}"


def merge_absolute_relative_segments(
    absolute_segments: list[dict], relative_segments: list[dict]
) -> list[dict]:
    """
    Merge absolute and relative segment data to compute true free-flow speed.

    For each segment:
    - current_speed = absolute speed_kph
    - speed_ratio = relative speed_ratio (fraction of free-flow)
    - free_flow_speed = current_speed / speed_ratio
    - direction = cardinal direction (N, NE, E, SE, S, SW, W, NW)

    Args:
        absolute_segments: Segments with speed_kph
        relative_segments: Segments with speed_ratio

    Returns:
        Merged segments with current_speed, free_flow_speed, speed_ratio, and direction
    """
    # Index relative segments by key for O(1) lookup
    relative_by_key = {seg["segment_key"]: seg for seg in relative_segments}

    merged = []
    for abs_seg in absolute_segments:
        key = abs_seg["segment_key"]
        rel_seg = relative_by_key.get(key)

        current_speed = abs_seg.get("speed_kph", 0)
        direction = abs_seg.get("direction", "N")
        bearing = abs_seg.get("bearing", 0.0)

        if rel_seg and rel_seg.get("speed_ratio", 0) > 0:
            speed_ratio = rel_seg["speed_ratio"]
            # Calculate true free-flow speed
            free_flow_speed = current_speed / speed_ratio
        else:
            # Fallback: estimate free-flow as 20% higher
            speed_ratio = 0.8
            free_flow_speed = current_speed * 1.2

        merged.append(
            {
                "segment_key": key,
                "coordinates": abs_seg["coordinates"],
                "current_speed": current_speed,
                "free_flow_speed": free_flow_speed,
                "speed_ratio": speed_ratio,
                "direction": direction,
                "bearing": bearing,
            }
        )

    return merged


def _tile_coords_to_latlon(
    coords: list, tile_x: int, tile_y: int, zoom: int, extent: int = 4096
) -> list[tuple[float, float]]:
    """
    Convert tile-local coordinates to lat/lon.

    MVT uses coordinates in range 0-4095 (extent) within each tile.
    """
    result = []
    n = 2**zoom

    for coord in coords:
        if len(coord) >= 2:
            px, py = coord[0], coord[1]

            # Convert to fractional tile coordinates
            fx = tile_x + (px / extent)
            fy = tile_y + (py / extent)

            # Convert to lat/lon
            lon = fx / n * 360.0 - 180.0
            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * fy / n)))
            lat = math.degrees(lat_rad)

            result.append((lat, lon))

    return result
