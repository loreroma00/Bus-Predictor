"""Tests for live data classes: Vehicle, LiveTrip, GPSData, Schedule, Update."""

from unittest.mock import Mock
import time


def make_trip():
    trip = Mock()
    trip.id = "trip_123"
    trip.route = Mock()
    trip.route.id = "route_1"
    trip.direction_id = 0
    trip.direction_name = "Termini"
    trip.stop_times = []
    return trip


def make_live_trip():
    from application.domain.static_data import Vehicle
    from application.domain.live_data import LiveTrip

    return LiveTrip(trip=make_trip(), vehicle=Vehicle(id="V001", label="V001"))


class TestVehicle:
    """Test static Vehicle identity."""

    def test_vehicle_creation(self):
        """Vehicle should store stable identity only."""
        from application.domain.static_data import Vehicle

        vehicle = Vehicle(id="V001", label="1001")

        assert vehicle.id == "V001"
        assert vehicle.label == "1001"
        assert vehicle.get_history() == []

    def test_vehicle_lazy_history(self):
        """Vehicle history should be loaded lazily through an injected loader."""
        from application.domain.static_data import Vehicle

        calls = []

        def loader(vehicle_id):
            calls.append(vehicle_id)
            return [{"trip_id": "trip_1"}]

        vehicle = Vehicle(id="V001", label="1001", history_loader=loader)

        assert vehicle.get_history() == [{"trip_id": "trip_1"}]
        assert vehicle.get_history() == [{"trip_id": "trip_1"}]
        assert calls == ["1001"]


class TestLiveTrip:
    """Test LiveTrip aggregate functionality."""

    def test_live_trip_creation(self):
        """LiveTrip should bind a trip to a static vehicle."""
        live_trip = make_live_trip()

        assert live_trip.get_id() == "V001"
        assert live_trip.get_trip().id == "trip_123"
        assert live_trip.measurements == []

    def test_live_trip_gps_data(self):
        """Should be able to set and get GPS data."""
        from application.domain.live_data import GPSData

        live_trip = make_live_trip()
        gps = GPSData(
            id="V001",
            trip=live_trip.trip,
            timestamp=time.time(),
            latitude=41.9,
            longitude=12.5,
            speed=30,
            heading=180,
        )

        live_trip.set_gps_data(gps)
        assert live_trip.get_gps_data() == gps

    def test_live_trip_latest_update(self):
        """Should be able to set and get latest update."""
        live_trip = make_live_trip()
        mock_update = Mock()

        live_trip.set_latest_update(mock_update)

        assert live_trip.get_latest_update() == mock_update

    def test_live_trip_location(self):
        """Should be able to set and get location info."""
        live_trip = make_live_trip()

        live_trip.set_hexagon_id("abc123")
        live_trip.set_location_name("Via Roma")

        assert live_trip.get_hexagon_id() == "abc123"
        assert live_trip.get_location_name() == "Via Roma"

    def test_live_trip_crowding_level(self):
        """Should return readable crowding status."""
        live_trip = make_live_trip()
        live_trip.set_occupancy_status(1)

        assert isinstance(live_trip.get_crowding_level(), str)


class TestGPSData:
    """Test GPSData class functionality."""

    def test_gpsdata_creation(self):
        """GPSData should store all GPS fields."""
        from application.domain.live_data import GPSData

        mock_trip = make_trip()
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

        gps = GPSData(
            id="V001",
            trip=make_trip(),
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
        """Update should be creatable with LiveTrip and next stops."""
        from application.domain.live_data import GPSData, Update

        live_trip = make_live_trip()
        live_trip.set_gps_data(
            GPSData(
                id="V001",
                trip=live_trip.trip,
                timestamp=1234567890,
                latitude=41.9,
                longitude=12.5,
                speed=30,
                heading=180,
                current_stop_sequence=1,
            )
        )

        next_stops = [{"stop_id": "S1", "stop_sequence": 1, "arrival_time": 1234567890}]
        update = Update(live_trip, next_stops)

        assert update.get_live_trip() == live_trip
        assert update.next_stops == next_stops

    def test_update_next_stops_stored(self):
        """Update should store the next_stops list."""
        from application.domain.live_data import GPSData, Update

        live_trip = make_live_trip()
        live_trip.set_gps_data(
            GPSData(
                id="V001",
                trip=live_trip.trip,
                timestamp=1234567890,
                latitude=41.9,
                longitude=12.5,
                speed=30,
                heading=180,
            )
        )

        next_stops = [
            {"stop_id": "S1", "stop_sequence": 1, "arrival_time": 1234567890},
            {"stop_id": "S2", "stop_sequence": 2, "arrival_time": 1234567950},
        ]

        update = Update(live_trip, next_stops)

        assert len(update.next_stops) == 2
        assert update.next_stops[0]["stop_id"] == "S1"
        assert update.next_stops[1]["stop_id"] == "S2"

    def test_update_str(self):
        """__str__ should return formatted string."""
        from application.domain.live_data import GPSData, Update

        live_trip = make_live_trip()
        live_trip.set_gps_data(
            GPSData(
                id="V001",
                trip=live_trip.trip,
                timestamp=1234567890,
                latitude=41.9,
                longitude=12.5,
                speed=30,
                heading=180,
            )
        )

        assert isinstance(str(Update(live_trip, [])), str)


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
        assert schedule.get("route_1", 0, "20260115") == []
