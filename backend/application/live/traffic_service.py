"""
Traffic Service - Projects traffic data onto H3 hexagons with directional data.

Uses h3_utils for hexagon operations and cities.py for updating hexagon data.
Designed for dependency injection following the project's existing patterns.
Aggregates traffic by 8 cardinal directions (N, NE, E, SE, S, SW, W, NW).

Smart fetching: Only fetches tiles for hexagons with buses, respecting TTL.
"""

import logging
import time
from typing import TYPE_CHECKING
from application.domain import h3_utils
from application.domain.spatial_utils import get_unique_tiles_from_hexagons
from .traffic_fetcher import (
    TomTomTrafficFetcher,
    extract_road_segments,
    merge_absolute_relative_segments,
)

if TYPE_CHECKING:
    from application.domain.cities import City

# Cardinal directions
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Default TTL for tile cache (15 minutes)
TILE_TTL_SECONDS = 900


class TrafficService:
    """
    Projects traffic data from road segments onto H3 hexagons by direction.

    Uses TomTomTrafficFetcher for data retrieval and h3_utils for
    hexagon coordinate conversion. Fetches both absolute and relative
    data to compute accurate current_speed and free_flow_speed.

    Smart fetching: Only fetches tiles covering hexagons with buses,
    and only when tile data has expired (TTL-based).
    """

    def __init__(self, city: "City", fetcher: TomTomTrafficFetcher):
        """
        Initialize the traffic service.

        Args:
            city: City instance to update traffic data on
            fetcher: TomTom traffic fetcher instance
        """
        self._city = city
        self._fetcher = fetcher
        self._tile_last_update: dict[tuple[int, int], float] = {}  # Track tile TTLs
        self._paused_until = 0.0

    def pause(self, seconds: int):
        """
        Pause traffic updates for a specified duration.
        """
        self._paused_until = time.time() + seconds
        logging.info(f"Traffic Service paused for {seconds} seconds.")

    def _is_paused(self) -> bool:
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

        # 1. Calculate unique tiles covering the requested hexagons
        tiles = set(get_unique_tiles_from_hexagons(hex_ids, self._fetcher.zoom))

        # 2. Filter to only expired tiles (using set for O(1) dedup)
        expired_tiles = {t for t in tiles if self._is_tile_expired(t)}

        if not expired_tiles:
            logging.debug(f"  Traffic: All {len(tiles)} tiles are fresh, skipping fetch")
            return []

        logging.debug(
            f"  Traffic: Fetching {len(expired_tiles)}/{len(tiles)} expired tile pairs "
            f"at zoom {self._fetcher.zoom}..."
        )

        # 3. Fetch both absolute and relative tiles
        abs_tiles, rel_tiles = self._fetcher.fetch_tiles_dual(list(expired_tiles))

        if not abs_tiles:
            logging.warning("  Traffic: No absolute tile data received")
            return []

        logging.debug(
            f"  Traffic: Received {len(abs_tiles)} absolute tiles, "
            f"{len(rel_tiles)} relative tiles"
        )

        # 4. Extract road segments from both tile sets
        all_abs_segments = []
        for tile_data in abs_tiles:
            segments = extract_road_segments(tile_data, flow_type="absolute")
            all_abs_segments.extend(segments)
            # Mark this tile as updated
            tile_coord = (tile_data.get("_tile_x"), tile_data.get("_tile_y"))
            self._mark_tile_updated(tile_coord)

        all_rel_segments = []
        for tile_data in rel_tiles:
            segments = extract_road_segments(tile_data, flow_type="relative")
            all_rel_segments.extend(segments)

        # 5. Merge to get accurate speed data with direction
        merged_segments = merge_absolute_relative_segments(
            all_abs_segments, all_rel_segments
        )

        logging.debug(f"  Traffic: Merged into {len(merged_segments)} complete segments")

        if not merged_segments:
            return []

        # 6. Project segments to hexagons and aggregate speeds BY DIRECTION
        hex_dir_speeds = self._aggregate_speeds_per_hex_direction(merged_segments)

        # 7. Update ALL hexagons covered by the tiles and reset their TTLs
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
        Global loop: updates traffic for all hexagons with buses.

        Returns:
            Count of ALL updated hexagons (including tile coverage propagation)
        """
        hex_ids = self._city.get_hexagons_with_buses()

        if not hex_ids:
            logging.debug("  Traffic: No hexagons with buses, skipping update")
            return 0

        logging.debug(f"  Traffic: Updating traffic for {len(hex_ids)} hexagons with buses")
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


def create_traffic_service(
    city: "City", api_key: str, zoom: int = 12
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
    fetcher = TomTomTrafficFetcher(api_key=api_key, zoom=zoom)
    return TrafficService(city, fetcher)
