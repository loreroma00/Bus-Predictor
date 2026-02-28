"""
Tests for Live Data classes - Autobus, GPSData, Schedule, Update.
"""

from unittest.mock import Mock
import time


class TestAutobus:
    """Test Autobus class functionality."""

    def test_autobus_creation(self):
        """Autobus should be creatable with id and trip."""
        from application.domain.live_data import Autobus

        mock_trip = Mock()
        mock_trip.trip_id = "trip_123"

        bus = Autobus(id="V001", trip=mock_trip)

        assert bus.get_id() == "V001"
        assert bus.get_trip() == mock_trip

    def test_autobus_gps_data(self):
        """Should be able to set and get GPS data."""
        from application.domain.live_data import Autobus, GPSData

        mock_trip = Mock()
        bus = Autobus(id="V001", trip=mock_trip)

        gps = GPSData(
            id="V001",
            trip=mock_trip,
            timestamp=time.time(),
            latitude=41.9,
            longitude=12.5,
            speed=30,
            heading=180,
        )

        bus.set_gpsData(gps)
        assert bus.get_gpsData() == gps

    def test_autobus_observer(self):
        """Should be able to set and get observer."""
        from application.domain.live_data import Autobus

        mock_trip = Mock()
        bus = Autobus(id="V001", trip=mock_trip)

        mock_observer = Mock()
        bus.set_observer(mock_observer)

        assert bus.get_observer() == mock_observer

    def test_autobus_latest_update(self):
        """Should be able to set and get latest update."""
        from application.domain.live_data import Autobus

        mock_trip = Mock()
        bus = Autobus(id="V001", trip=mock_trip)

        mock_update = Mock()
        bus.set_latest_update(mock_update)

        assert bus.get_latest_update() == mock_update

    def test_autobus_location(self):
        """Should be able to set and get location info."""
        from application.domain.live_data import Autobus

        mock_trip = Mock()
        bus = Autobus(id="V001", trip=mock_trip)

        bus.set_hexagon_id("abc123")
        bus.set_location_name("Via Roma")

        assert bus.get_hexagon_id() == "abc123"
        assert bus.get_location_name() == "Via Roma"

    def test_autobus_crowding_level(self):
        """Should return readable crowding status."""
        from application.domain.live_data import Autobus

        mock_trip = Mock()
        bus = Autobus(id="V001", trip=mock_trip, occupancy_status=1)

        level = bus.get_crowding_level()
        assert isinstance(level, str)


class TestGPSData:
    """Test GPSData class functionality."""

    def test_gpsdata_creation(self):
        """GPSData should store all GPS fields."""
        from application.domain.live_data import GPSData

        mock_trip = Mock()
        gps = GPSData(
            id="V001",
            trip=mock_trip,
            timestamp=1234567890,
            latitude=41.9028,
            longitude=12.4964,
            speed=25.5,
            heading=180,
            next_stop_id="stop_1",
            current_stop_sequence=5,
            current_status=2,
        )

        assert gps.get_latitude() == 41.9028
        assert gps.get_longitude() == 12.4964
        assert gps.speed == 25.5
        assert gps.heading == 180
        assert gps.next_stop_id == "stop_1"
        assert gps.current_stop_sequence == 5
        assert gps.current_status == 2

    def test_gpsdata_defaults(self):
        """Optional fields should default to None."""
        from application.domain.live_data import GPSData

        mock_trip = Mock()
        gps = GPSData(
            id="V001",
            trip=mock_trip,
            timestamp=1234567890,
            latitude=41.9,
            longitude=12.5,
            speed=30,
            heading=90,
        )

        assert gps.next_stop_id is None
        assert gps.current_stop_sequence is None
        assert gps.current_status is None


class TestUpdate:
    """Test Update class functionality."""

    def test_update_creation(self):
        """Update should be creatable with autobus and next_stops."""
        from application.domain.live_data import Update, Autobus, GPSData

        mock_trip = Mock()
        mock_trip.stop_times = []

        bus = Autobus(id="V001", trip=mock_trip)

        # Create and set GPSData
        gps = GPSData(
            id="V001",
            trip=mock_trip,
            timestamp=1234567890,
            latitude=41.9,
            longitude=12.5,
            speed=30,
            heading=180,
            current_stop_sequence=1,
        )
        bus.set_gpsData(gps)

        next_stops = [{"stop_id": "S1", "stop_sequence": 1, "arrival_time": 1234567890}]

        update = Update(bus, next_stops)

        assert update.get_autobus() == bus
        assert update.next_stops == next_stops

    def test_update_next_stops_stored(self):
        """Update should store the next_stops list."""
        from application.domain.live_data import Update, Autobus, GPSData

        mock_trip = Mock()
        mock_trip.stop_times = []

        bus = Autobus(id="V001", trip=mock_trip)
        gps = GPSData(
            id="V001",
            trip=mock_trip,
            timestamp=1234567890,
            latitude=41.9,
            longitude=12.5,
            speed=30,
            heading=180,
        )
        bus.set_gpsData(gps)

        next_stops = [
            {"stop_id": "S1", "stop_sequence": 1, "arrival_time": 1234567890},
            {"stop_id": "S2", "stop_sequence": 2, "arrival_time": 1234567950},
        ]

        update = Update(bus, next_stops)

        # Verify the next_stops are stored correctly
        assert len(update.next_stops) == 2
        assert update.next_stops[0]["stop_id"] == "S1"
        assert update.next_stops[1]["stop_id"] == "S2"

    def test_update_str(self):
        """__str__ should return formatted string."""
        from application.domain.live_data import Update, Autobus, GPSData

        mock_trip = Mock()
        mock_trip.trip_id = "trip_123"
        mock_trip.stop_times = []

        bus = Autobus(id="V001", trip=mock_trip)
        gps = GPSData(
            id="V001",
            trip=mock_trip,
            timestamp=1234567890,
            latitude=41.9,
            longitude=12.5,
            speed=30,
            heading=180,
        )
        bus.set_gpsData(gps)

        update = Update(bus, [])

        result = str(update)
        assert isinstance(result, str)


class TestSchedule:
    """Test Schedule class functionality."""

    def test_schedule_creation(self):
        """Schedule should be creatable."""
        from application.domain.live_data import Schedule

        schedule = Schedule()
        assert schedule.index == {}

    def test_schedule_get_nonexistent(self):
        """Should return empty list for nonexistent route."""
        from application.domain.live_data import Schedule

        schedule = Schedule()
        result = schedule.get("route_1", 0, "20260115")

        assert result == []
