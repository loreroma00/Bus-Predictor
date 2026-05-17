"""Tests for live-domain objects and shared time helpers."""

from unittest.mock import Mock
import time


def _mock_trip():
    """Create the smallest Trip-like object required by LiveTrip/GPSData."""
    route = Mock()
    route.id = "R1"
    trip = Mock()
    trip.id = "trip_123"
    trip.route = route
    trip.direction_name = "Test Direction"
    return trip


def _mock_vehicle():
    """Create the smallest Vehicle-like object required by LiveTrip."""
    vehicle = Mock()
    vehicle.id = "vehicle_123"
    vehicle.label = "bus_123"
    vehicle.vehicle_type = None
    return vehicle


def _mock_gps():
    """Create a GPSData instance for measurement tests."""
    from application.domain.live_data import GPSData

    return GPSData(
        id="gps_1",
        trip=_mock_trip(),
        timestamp=time.time(),
        latitude=41.9,
        longitude=12.5,
        speed=30,
        heading=180,
    )


class TestLiveTrip:
    """Test LiveTrip aggregate behavior."""

    def test_live_trip_creation(self):
        """LiveTrip should bind one static vehicle to one running trip."""
        from application.domain.live_data import LiveTrip

        trip = _mock_trip()
        vehicle = _mock_vehicle()

        live_trip = LiveTrip(trip, vehicle)

        assert live_trip.trip == trip
        assert live_trip.vehicle == vehicle
        assert live_trip.trip_id == "trip_123"
        assert live_trip.vehicle_id == "vehicle_123"
        assert live_trip.measurements == []
        assert live_trip.is_finished is False

    def test_set_gps_data_updates_current_state(self):
        """LiveTrip should keep the latest GPS ping."""
        from application.domain.live_data import LiveTrip

        live_trip = LiveTrip(_mock_trip(), _mock_vehicle())
        gps = _mock_gps()

        live_trip.set_gps_data(gps)

        assert live_trip.get_gps_data() == gps
        assert live_trip.last_seen_timestamp > 0


class TestMeasurement:
    """Test Measurement records."""

    def test_measurement_creation(self):
        """Measurement should store all provided data."""
        from application.domain.live_data import Measurement

        gps = _mock_gps()
        measurement = Measurement(
            id=1,
            vehicle_id="bus_123",
            next_stop="stop_1",
            next_stop_distance=150.5,
            gpsdata=gps,
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )

        assert measurement.id == 1
        assert measurement.vehicle_id == "bus_123"
        assert measurement.next_stop == "stop_1"
        assert measurement.next_stop_distance == 150.5
        assert measurement.gpsdata == gps
        assert measurement.trip_id == "trip_123"
        assert measurement.measurement_time > 0

    def test_to_dict_contains_required_fields(self):
        """to_dict should contain all required fields for parquet."""
        from application.domain.live_data import Measurement

        measurement = Measurement(
            id=1,
            vehicle_id="bus_123",
            next_stop="stop_1",
            next_stop_distance=150.5,
            gpsdata=_mock_gps(),
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )
        data = measurement.to_dict("trip_123")

        required_fields = [
            "trip_id",
            "stop_id",
            "next_stop",
            "next_stop_distance",
            "lat",
            "lon",
            "speed",
            "bearing",
            "measurement_time",
            "formatted_time",
            "occupancy",
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_to_dict_has_formatted_time(self):
        """to_dict should include human-readable formatted_time."""
        from application.domain.live_data import Measurement

        measurement = Measurement(
            id=1,
            vehicle_id="bus_123",
            next_stop="stop_1",
            next_stop_distance=150.5,
            gpsdata=_mock_gps(),
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )
        data = measurement.to_dict("trip_123")

        assert data["formatted_time"] is not None
        assert isinstance(data["formatted_time"], str)


class TestTimeUtils:
    """Test time utility functions."""

    def test_to_unix_time(self):
        """to_unix_time should return integer timestamp."""
        from application.domain.time_utils import to_unix_time

        now = time.time()
        result = to_unix_time(now)

        assert isinstance(result, int)
        assert result > 0

    def test_to_readable_time(self):
        """to_readable_time should return formatted string."""
        from application.domain.time_utils import to_readable_time

        now = time.time()
        result = to_readable_time(now)

        assert isinstance(result, str)
        assert ":" in result
