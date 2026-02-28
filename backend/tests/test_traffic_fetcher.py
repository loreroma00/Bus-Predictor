"""
Tests for Traffic Fetcher - TomTom API client and MVT decoding.
"""

from unittest.mock import Mock, patch


class TestTileCalculation:
    """Test tile coordinate calculation functions."""

    def test_lat_lon_to_tile_rome_center(self):
        """Verify tile calculation for Rome's approximate center."""
        from application.domain.spatial_utils import lat_lon_to_tile

        # Rome center: approximately 41.9, 12.5
        x, y = lat_lon_to_tile(41.9, 12.5, zoom=12)

        # At zoom 12, Rome tiles should be reasonable values in the Mediterranean
        # Just verify they're in a sensible range for Europe
        assert 2000 <= x <= 2300, f"x={x} out of expected range"
        assert 1400 <= y <= 1600, f"y={y} out of expected range"

    def test_tile_to_lat_lon_roundtrip(self):
        """Verify tile to lat/lon conversion is approximately reversible."""
        from application.domain.spatial_utils import (
            lat_lon_to_tile,
            tile_to_lat_lon,
        )

        original_lat, original_lon = 41.9, 12.5
        zoom = 12

        x, y = lat_lon_to_tile(original_lat, original_lon, zoom)
        recovered_lat, recovered_lon = tile_to_lat_lon(x, y, zoom)

        # Should be within one tile's width (~0.1 degrees at zoom 12)
        assert abs(recovered_lat - original_lat) < 0.2
        assert abs(recovered_lon - original_lon) < 0.2

    def test_get_tiles_for_bbox_generates_grid(self):
        """Verify bbox produces a rectangular grid of tiles."""
        from application.live.traffic_fetcher import TomTomTrafficFetcher

        fetcher = TomTomTrafficFetcher(api_key="dummy", zoom=12)

        # Small bbox that should cover 2x2 tiles
        tiles = fetcher.get_tiles_for_bbox(
            min_lat=41.85, min_lon=12.45, max_lat=41.95, max_lon=12.55
        )

        assert len(tiles) >= 1
        # All tiles should be unique
        assert len(tiles) == len(set(tiles))


class TestRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limit_enforced(self):
        """Verify rate limiting adds delay between requests."""
        from application.live.traffic_fetcher import TomTomTrafficFetcher
        import time

        fetcher = TomTomTrafficFetcher(api_key="dummy", zoom=12)

        # First call should not delay
        start1 = time.time()
        fetcher._rate_limit()
        elapsed1 = time.time() - start1
        assert elapsed1 < 0.05  # Should be nearly instant

        # Immediate second call should delay
        start2 = time.time()
        fetcher._rate_limit()
        elapsed2 = time.time() - start2
        assert elapsed2 >= 0.09  # Should wait ~100ms


class TestMVTDecoding:
    """Test MVT decoding and segment extraction."""

    def test_extract_road_segments_empty_tile(self):
        """Verify empty tile returns empty list."""
        from application.live.traffic_fetcher import extract_road_segments

        empty_tile = {}
        segments = extract_road_segments(empty_tile)
        assert segments == []

    def test_extract_road_segments_no_flow_layer(self):
        """Verify tile without traffic flow layer returns empty list."""
        from application.live.traffic_fetcher import extract_road_segments

        tile_data = {"Other layer": {"features": []}}
        segments = extract_road_segments(tile_data)
        assert segments == []

    def test_extract_absolute_segments(self):
        """Verify absolute segment extraction gets speed_kph."""
        from application.live.traffic_fetcher import extract_road_segments

        tile_data = {
            "Traffic flow": {
                "features": [
                    {
                        "properties": {"traffic_level": 45},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[100, 100], [200, 200]],
                        },
                    },
                ]
            },
            "_tile_x": 2187,
            "_tile_y": 1483,
            "_tile_zoom": 12,
        }

        segments = extract_road_segments(tile_data, flow_type="absolute")

        assert len(segments) == 1
        assert segments[0]["speed_kph"] == 45
        assert "segment_key" in segments[0]

    def test_extract_relative_segments(self):
        """Verify relative segment extraction gets speed_ratio."""
        from application.live.traffic_fetcher import extract_road_segments

        tile_data = {
            "Traffic flow": {
                "features": [
                    {
                        "properties": {"traffic_level": 0.75},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[100, 100], [200, 200]],
                        },
                    },
                ]
            },
            "_tile_x": 2187,
            "_tile_y": 1483,
            "_tile_zoom": 12,
        }

        segments = extract_road_segments(tile_data, flow_type="relative")

        assert len(segments) == 1
        assert segments[0]["speed_ratio"] == 0.75


class TestSegmentMerging:
    """Test merging absolute and relative segments."""

    def test_merge_matching_segments(self):
        """Verify segments with same key are merged correctly."""
        from application.live.traffic_fetcher import merge_absolute_relative_segments

        abs_segments = [
            {
                "segment_key": "key1",
                "coordinates": [(41.9, 12.5)],
                "speed_kph": 60,
            }
        ]
        rel_segments = [
            {
                "segment_key": "key1",
                "coordinates": [(41.9, 12.5)],
                "speed_ratio": 0.75,
            }
        ]

        merged = merge_absolute_relative_segments(abs_segments, rel_segments)

        assert len(merged) == 1
        assert merged[0]["current_speed"] == 60
        assert merged[0]["speed_ratio"] == 0.75
        # free_flow = current / ratio = 60 / 0.75 = 80
        assert merged[0]["free_flow_speed"] == 80.0

    def test_merge_unmatched_uses_fallback(self):
        """Verify unmatched absolute segments get fallback values."""
        from application.live.traffic_fetcher import merge_absolute_relative_segments

        abs_segments = [
            {
                "segment_key": "key_no_match",
                "coordinates": [(41.9, 12.5)],
                "speed_kph": 50,
            }
        ]
        rel_segments = []  # No matching relative data

        merged = merge_absolute_relative_segments(abs_segments, rel_segments)

        assert len(merged) == 1
        assert merged[0]["current_speed"] == 50
        # Fallback: free_flow = current * 1.2 = 60
        assert merged[0]["free_flow_speed"] == 60.0


class TestFetchTile:
    """Test tile fetching with mocked HTTP."""

    @patch("application.live.traffic_fetcher.requests.get")
    def test_fetch_tile_with_flow_type(self, mock_get):
        """Verify flow_type is included in URL."""
        from application.live.traffic_fetcher import TomTomTrafficFetcher

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b""

        mock_get.return_value = mock_response

        fetcher = TomTomTrafficFetcher(api_key="test_key", zoom=12)

        with patch("application.live.traffic_fetcher.mvt.decode") as mock_decode:
            mock_decode.return_value = {}

            fetcher.fetch_tile(2187, 1483, flow_type="relative")

            # Check that relative is in the URL
            call_args = mock_get.call_args
            assert "relative" in call_args[0][0]

    @patch("application.live.traffic_fetcher.requests.get")
    def test_fetch_tile_404_returns_none(self, mock_get):
        """Verify 404 response returns None (no data for tile)."""
        from application.live.traffic_fetcher import TomTomTrafficFetcher

        mock_response = Mock()
        mock_response.status_code = 404

        mock_get.return_value = mock_response

        fetcher = TomTomTrafficFetcher(api_key="test_key", zoom=12)
        result = fetcher.fetch_tile(0, 0)

        assert result is None

    @patch("application.live.traffic_fetcher.requests.get")
    def test_fetch_tile_403_returns_none(self, mock_get):
        """Verify 403 response returns None (invalid API key)."""
        from application.live.traffic_fetcher import TomTomTrafficFetcher

        mock_response = Mock()
        mock_response.status_code = 403

        mock_get.return_value = mock_response

        fetcher = TomTomTrafficFetcher(api_key="invalid_key", zoom=12)
        result = fetcher.fetch_tile(0, 0)

        assert result is None


class TestDualFetch:
    """Test dual tile fetching."""

    @patch.object(
        __import__(
            "application.live.traffic_fetcher", fromlist=["TomTomTrafficFetcher"]
        ).TomTomTrafficFetcher,
        "fetch_tile",
    )
    def test_fetch_tiles_dual_returns_two_lists(self, mock_fetch):
        """Verify fetch_tiles_dual returns both absolute and relative lists."""
        from application.live.traffic_fetcher import TomTomTrafficFetcher

        # Mock returns different data for each call
        mock_fetch.side_effect = [
            {"flow": "abs"},  # absolute for tile 1
            {"flow": "rel"},  # relative for tile 1
        ]

        fetcher = TomTomTrafficFetcher(api_key="test", zoom=12)
        abs_tiles, rel_tiles = fetcher.fetch_tiles_dual([(0, 0)])

        assert len(abs_tiles) == 1
        assert len(rel_tiles) == 1
        assert abs_tiles[0]["_flow_type"] == "absolute"
        assert rel_tiles[0]["_flow_type"] == "relative"
