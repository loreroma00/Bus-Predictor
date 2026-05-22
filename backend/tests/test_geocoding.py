"""
Tests for Geocoding and City Management.
"""

from unittest.mock import Mock


class TestGeocodingStrategy:
    """Test GeocodingStrategy protocol compliance."""

    def test_async_geocoding_service_satisfies_protocol(self):
        """AsyncGeocodingService should satisfy GeocodingStrategy protocol."""
        from application.domain.interfaces import GeocodingStrategy
        from application.domain.map_info import AsyncGeocodingService

        mock_city = Mock()
        service = AsyncGeocodingService(mock_city)

        assert isinstance(service, GeocodingStrategy)

    def test_async_geocoding_has_required_methods(self):
        """AsyncGeocodingService should have enqueue, get_street, process_one."""
        from application.domain.map_info import AsyncGeocodingService

        mock_city = Mock()
        service = AsyncGeocodingService(mock_city)

        assert hasattr(service, "enqueue")
        assert hasattr(service, "get_street")
        assert hasattr(service, "process_one")
        assert callable(service.enqueue)
        assert callable(service.get_street)
        assert callable(service.process_one)


class TestAsyncGeocodingService:
    """Test AsyncGeocodingService functionality."""

    def test_enqueue_adds_to_queue(self):
        """Enqueue should add coordinates to the queue."""
        from application.domain.map_info import AsyncGeocodingService
        from application.domain.cities import City

        city = City("TestCity")
        service = AsyncGeocodingService(city)

        service.enqueue(41.9028, 12.4964, "hex_123")

        assert service.get_queue_size() == 1

    def test_enqueue_deduplicates(self):
        """Enqueue should not add duplicate coordinates."""
        from application.domain.map_info import AsyncGeocodingService
        from application.domain.cities import City

        city = City("TestCity")
        service = AsyncGeocodingService(city)

        # Same coordinates should only be added once
        service.enqueue(41.9028, 12.4964, "hex_123")
        service.enqueue(41.9028, 12.4964, "hex_123")

        assert service.get_queue_size() == 1

    def test_process_one_empty_queue(self):
        """process_one should return False when queue is empty."""
        from application.domain.map_info import AsyncGeocodingService
        from application.domain.cities import City

        city = City("TestCity")
        service = AsyncGeocodingService(city)

        result = service.process_one()

        assert result is False

    def test_coordinate_rounding(self):
        """Coordinates should be rounded to 5 decimal places."""
        from application.domain.map_info import AsyncGeocodingService
        from application.domain.cities import City

        city = City("TestCity")
        service = AsyncGeocodingService(city)

        # These differ only in the 6th+ decimal place, so should be same after rounding to 5
        # 41.902811 rounds to 41.90281
        # 41.902812 also rounds to 41.90281
        service.enqueue(41.902811, 12.496411, "hex_1")
        service.enqueue(41.902812, 12.496412, "hex_2")

        # After rounding to 5 decimals, these should be the same point
        assert service.get_queue_size() == 1

    def test_get_and_reset_resolved_count(self):
        """Should track and reset resolved count."""
        from application.domain.map_info import AsyncGeocodingService
        from application.domain.cities import City

        city = City("TestCity")
        service = AsyncGeocodingService(city)

        # Initially zero
        count = service.get_and_reset_resolved_count()
        assert count == 0

        # After reset, still zero
        count = service.get_and_reset_resolved_count()
        assert count == 0


class TestObservatoryCityManagement:
    """Test Observatory's city management features."""

    def test_add_city(self):
        """Should be able to add a city."""
        from application.domain.virtual_entities import Observatory

        obs = Observatory()
        obs.add_city("Roma")

        city = obs.get_city("Roma")
        assert city is not None

    def test_get_nonexistent_city(self):
        """Should return None for nonexistent city."""
        from application.domain.virtual_entities import Observatory

        obs = Observatory()

        city = obs.get_city("NonexistentCity")
        assert city is None


class TestObservatoryLiveTripManagement:
    """Test Observatory's live-trip management features."""

    def test_add_live_trip_to_city(self):
        """Should be able to add a live trip to a city."""
        from application.domain.virtual_entities import Observatory

        class DummyLiveTrip:
            def __init__(self, live_trip_id: str):
                self.id = live_trip_id
                self.hexagon_id = None

            def get_bearing(self):
                return 0.0

            def set_is_in_preferential(self, value: bool):
                self.is_in_preferential = value

            def set_hexagon_id(self, hexagon_id: str):
                self.hexagon_id = hexagon_id

        obs = Observatory()
        obs.add_city("Roma")

        live_trip = DummyLiveTrip("V001")

        obs.add_live_trip_to_city("Roma", live_trip, hex_id="892d3fbe257ffff")

        retrieved = obs.get_city("Roma").get_live_trip("V001")
        assert retrieved == live_trip

    def test_get_nonexistent_live_trip(self):
        """Should return None for nonexistent live trip."""
        from application.domain.virtual_entities import Observatory

        obs = Observatory()
        obs.add_city("Roma")

        live_trip = obs.get_city("Roma").get_live_trip("nonexistent_live_trip")
        assert live_trip is None
