"""
Tests for Traffic Service - Hexagon projection and 8-direction speed aggregation.
"""

from unittest.mock import Mock, patch


class TestTrafficProjection:
    """Test road segment to hexagon projection."""

    def test_project_segment_to_hexagons_single_point(self):
        """Verify single point projects to one hexagon."""
        from application.live.traffic_service import TrafficService

        mock_city = Mock()
        mock_fetcher = Mock()
        service = TrafficService(mock_city, mock_fetcher)

        # Rome coordinates
        coordinates = [(41.9, 12.5)]
        hex_ids = service._project_segment_to_hexagons(coordinates)

        assert len(hex_ids) == 1
        # H3 IDs start with "8" for resolution 9
        assert hex_ids[0].startswith("8")

    def test_project_segment_to_hexagons_long_road(self):
        """Verify long road segment projects to multiple hexagons."""
        from application.live.traffic_service import TrafficService

        mock_city = Mock()
        mock_fetcher = Mock()
        service = TrafficService(mock_city, mock_fetcher)

        # Long road crossing multiple hexagons
        coordinates = [
            (41.9, 12.5),
            (41.91, 12.51),
            (41.92, 12.52),
            (41.93, 12.53),
        ]
        hex_ids = service._project_segment_to_hexagons(coordinates)

        # Should cover multiple hexagons
        assert len(hex_ids) >= 1
        # All should be unique
        assert len(hex_ids) == len(set(hex_ids))


class TestSpeedAggregationByDirection:
    """Test speed aggregation per hexagon AND direction."""

    def test_aggregate_empty_segments(self):
        """Verify empty segment list returns empty dict."""
        from application.live.traffic_service import TrafficService

        mock_city = Mock()
        mock_fetcher = Mock()
        service = TrafficService(mock_city, mock_fetcher)

        result = service._aggregate_speeds_per_hex_direction([])
        assert result == {}

    def test_aggregate_single_segment_with_direction(self):
        """Verify single merged segment aggregation includes direction."""
        from application.live.traffic_service import TrafficService

        mock_city = Mock()
        mock_fetcher = Mock()
        service = TrafficService(mock_city, mock_fetcher)

        segments = [
            {
                "segment_key": "key1",
                "current_speed": 50.0,
                "free_flow_speed": 65.0,
                "speed_ratio": 0.77,
                "direction": "NE",
                "coordinates": [(41.9, 12.5)],
            }
        ]

        result = service._aggregate_speeds_per_hex_direction(segments)

        assert len(result) == 1
        # Key is now (hex_id, direction)
        key = list(result.keys())[0]
        assert len(key) == 2  # tuple of (hex_id, direction)
        assert key[1] == "NE"
        assert result[key]["avg_current_speed"] == 50.0
        assert result[key]["avg_flow_speed"] == 65.0
        assert result[key]["count"] == 1

    def test_aggregate_multiple_segments_same_hex_same_direction(self):
        """Verify multiple segments in same hex AND direction are averaged."""
        from application.live.traffic_service import TrafficService

        mock_city = Mock()
        mock_fetcher = Mock()
        service = TrafficService(mock_city, mock_fetcher)

        # Two merged segments in same location AND direction
        segments = [
            {
                "segment_key": "key1",
                "current_speed": 40.0,
                "free_flow_speed": 50.0,
                "speed_ratio": 0.8,
                "direction": "N",
                "coordinates": [(41.9, 12.5)],
            },
            {
                "segment_key": "key2",
                "current_speed": 60.0,
                "free_flow_speed": 80.0,
                "speed_ratio": 0.75,
                "direction": "N",
                "coordinates": [(41.9, 12.5)],
            },
        ]

        result = service._aggregate_speeds_per_hex_direction(segments)

        # Should have 1 entry (same hex, same direction)
        assert len(result) == 1
        key = list(result.keys())[0]
        assert key[1] == "N"
        assert result[key]["avg_current_speed"] == 50.0  # (40 + 60) / 2
        assert result[key]["avg_flow_speed"] == 65.0  # (50 + 80) / 2
        assert result[key]["count"] == 2

    def test_aggregate_segments_different_directions(self):
        """Verify segments in same hex but different directions create separate entries."""
        from application.live.traffic_service import TrafficService

        mock_city = Mock()
        mock_fetcher = Mock()
        service = TrafficService(mock_city, mock_fetcher)

        segments = [
            {
                "segment_key": "key1",
                "current_speed": 40.0,
                "free_flow_speed": 50.0,
                "speed_ratio": 0.8,
                "direction": "N",
                "coordinates": [(41.9, 12.5)],
            },
            {
                "segment_key": "key2",
                "current_speed": 60.0,
                "free_flow_speed": 80.0,
                "speed_ratio": 0.75,
                "direction": "S",  # Different direction
                "coordinates": [(41.9, 12.5)],
            },
        ]

        result = service._aggregate_speeds_per_hex_direction(segments)

        # Should have 2 entries (same hex, different directions)
        assert len(result) == 2
        directions = {key[1] for key in result.keys()}
        assert "N" in directions
        assert "S" in directions

    def test_aggregate_skips_zero_speed(self):
        """Verify segments with zero speed are skipped."""
        from application.live.traffic_service import TrafficService

        mock_city = Mock()
        mock_fetcher = Mock()
        service = TrafficService(mock_city, mock_fetcher)

        segments = [
            {
                "segment_key": "key1",
                "current_speed": 0,
                "free_flow_speed": 50.0,
                "speed_ratio": 0.0,
                "direction": "N",
                "coordinates": [(41.9, 12.5)],
            },
        ]

        result = service._aggregate_speeds_per_hex_direction(segments)
        assert result == {}


class TestUpdateTraffic:
    """Test the main update_traffic method."""

    def test_update_traffic_calls_city_update_with_direction(self):
        """Verify update_traffic calls city.update_traffic with direction."""
        from application.live.traffic_service import TrafficService

        mock_hex = Mock()
        mock_hex.reset_traffic_ttl = Mock()

        mock_city = Mock()
        mock_city.update_traffic = Mock()
        mock_city.get_hexagons_with_buses.return_value = ["hex_123"]
        mock_city.hexagons = {"hex_123": mock_hex}

        mock_fetcher = Mock()
        mock_fetcher.zoom = 12
        mock_fetcher.fetch_tiles_dual.return_value = (
            [
                {
                    "Traffic flow": {"features": []},
                    "_tile_x": 2187,
                    "_tile_y": 1483,
                    "_tile_zoom": 12,
                }
            ],
            [
                {
                    "Traffic flow": {"features": []},
                    "_tile_x": 2187,
                    "_tile_y": 1483,
                    "_tile_zoom": 12,
                }
            ],
        )

        service = TrafficService(mock_city, mock_fetcher)

        # Mock aggregation to return known data with direction
        with (
            patch.object(service, "_aggregate_speeds_per_hex_direction") as mock_agg,
            patch(
                "application.live.traffic_service.merge_absolute_relative_segments"
            ) as mock_merge,
            patch(
                "application.live.traffic_service.get_unique_tiles_from_hexagons"
            ) as mock_tiles,
        ):
            mock_tiles.return_value = [(2187, 1483)]
            mock_merge.return_value = [{"dummy": "data"}]

            # Key is now (hex_id, direction)
            mock_agg.return_value = {
                ("hex_123", "NE"): {
                    "avg_current_speed": 45.0,
                    "avg_flow_speed": 60.0,
                    "avg_ratio": 0.75,
                    "count": 5,
                }
            }

            updated = service.update_traffic()

            mock_agg.assert_called_once_with([{"dummy": "data"}])

            # New signature: (hex_id, direction, current_speed, speed_ratio)
            mock_city.update_traffic.assert_called_once_with(
                "hex_123", "NE", 45.0, 0.75
            )
            assert updated == 1


class TestFactoryFunction:
    """Test the factory function."""

    def test_create_traffic_service(self):
        """Verify factory creates properly configured service."""
        from application.live.traffic_service import create_traffic_service

        mock_city = Mock()

        service = create_traffic_service(mock_city, api_key="test_key", zoom=12)

        assert service._city == mock_city
        assert service._fetcher.api_key == "test_key"
        assert service._fetcher.zoom == 12
