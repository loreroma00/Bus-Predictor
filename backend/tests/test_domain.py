"""
Tests for Domain Layer - Observers, Diaries, Measurements.
"""

from unittest.mock import Mock
import time


class TestObserver:
    """Test Observer class functionality."""

    def test_observer_creation(self):
        """Observer should be creatable with vehicle and diary."""
        from application.domain.observers import Observer, Diary

        mock_vehicle = Mock()
        mock_vehicle.id = "V001"

        diary = Diary(None, "trip_123")
        observer = Observer(None, mock_vehicle, diary)

        assert observer.assignedVehicle == mock_vehicle
        assert observer.current_diary == diary
        assert observer.diary_history == []

    def test_archive_current_diary(self):
        """Archiving should move diary to history."""
        from application.domain.observers import Observer, Diary

        mock_vehicle = Mock()
        mock_vehicle.id = "V001"

        # Mock observatory to prevent crash in archive_current_diary
        mock_observatory = Mock()
        mock_observatory.search_trip.return_value = Mock()

        diary = Diary(None, "trip_123")
        observer = Observer(mock_observatory, mock_vehicle, diary)

        observer.archive_current_diary()

        assert observer.current_diary is None
        assert len(observer.diary_history) == 1
        assert observer.diary_history[0].is_finished is True

    def test_get_bus_returns_vehicle(self):
        """get_bus should return assigned vehicle."""
        from application.domain.observers import Observer, Diary

        mock_vehicle = Mock()
        diary = Diary(None, "trip_123")
        observer = Observer(None, mock_vehicle, diary)

        assert observer.get_bus() == mock_vehicle


class TestDiary:
    """Test Diary class functionality."""

    def test_diary_creation(self):
        """Diary should be creatable with observer and trip_id."""
        from application.domain.observers import Diary

        mock_observer = Mock()
        diary = Diary(mock_observer, "trip_123")

        assert diary.trip_id == "trip_123"
        assert diary.observer == mock_observer
        assert diary.measurements == []
        assert diary.is_finished is False

    def test_add_measurement(self):
        """Should be able to add measurements."""
        from application.domain.observers import Diary, Measurement

        diary = Diary(None, "trip_123")

        mock_gps = Mock()
        mock_gps.latitude = 41.9
        mock_gps.longitude = 12.5
        mock_gps.speed = 30
        mock_gps.heading = 180
        mock_gps.occupancy_status = 1

        measurement = Measurement(
            id=1,
            autobus_id="bus_123",
            next_stop="stop_1",
            next_stop_distance=100.0,
            gpsdata=mock_gps,
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )
        diary.add_measurement(measurement)

        assert len(diary.measurements) == 1
        assert diary.measurements[0] == measurement

    def test_get_measurements_amount(self):
        """Should return correct count of measurements."""
        from application.domain.observers import Diary, Measurement

        diary = Diary(None, "trip_123")

        mock_gps = Mock()
        mock_gps.latitude = 41.9
        mock_gps.longitude = 12.5
        mock_gps.speed = 30
        mock_gps.heading = 180
        mock_gps.occupancy_status = 1

        m1 = Measurement(
            id=1,
            autobus_id="bus_123",
            next_stop="stop_1",
            next_stop_distance=100.0,
            gpsdata=mock_gps,
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )
        m2 = Measurement(
            id=2,
            autobus_id="bus_123",
            next_stop="stop_2",
            next_stop_distance=200.0,
            gpsdata=mock_gps,
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )

        diary.add_measurement(m1)
        diary.add_measurement(m2)

        assert diary.get_measurements_amount() == 2

    def test_to_dict_list(self):
        """Should convert measurements to list of dicts."""
        from application.domain.observers import Diary, Measurement

        diary = Diary(None, "trip_123")

        mock_gps = Mock()
        mock_gps.latitude = 41.9
        mock_gps.longitude = 12.5
        mock_gps.speed = 30
        mock_gps.heading = 180
        mock_gps.occupancy_status = 1

        m = Measurement(
            id=1,
            autobus_id="bus_123",
            next_stop="stop_1",
            next_stop_distance=100.0,
            gpsdata=mock_gps,
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )
        diary.add_measurement(m)

        dict_list = diary.to_dict_list()

        assert len(dict_list) == 1
        assert dict_list[0]["trip_id"] == "trip_123"
        assert dict_list[0]["lat"] == 41.9


class TestMeasurement:
    """Test Measurement class functionality."""

    def test_measurement_creation(self):
        """Measurement should store all provided data."""
        from application.domain.observers import Measurement

        mock_gps = Mock()
        mock_gps.latitude = 41.9
        mock_gps.longitude = 12.5
        mock_gps.speed = 30
        mock_gps.heading = 180
        mock_gps.occupancy_status = 1

        m = Measurement(
            id=1,
            autobus_id="bus_123",
            next_stop="stop_1",
            next_stop_distance=150.5,
            gpsdata=mock_gps,
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )

        assert m.id == 1
        assert m.next_stop == "stop_1"
        assert m.next_stop_distance == 150.5
        assert m.gpsdata == mock_gps
        assert m.trip_id == "trip_123"
        assert m.measurement_time > 0  # Unix timestamp

    def test_to_dict_contains_required_fields(self):
        """to_dict should contain all required fields for parquet."""
        from application.domain.observers import Measurement

        mock_gps = Mock()
        mock_gps.latitude = 41.9
        mock_gps.longitude = 12.5
        mock_gps.speed = 30
        mock_gps.heading = 180
        mock_gps.occupancy_status = 1

        m = Measurement(
            id=1,
            autobus_id="bus_123",
            next_stop="stop_1",
            next_stop_distance=150.5,
            gpsdata=mock_gps,
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )
        d = m.to_dict("trip_123")

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
            assert field in d, f"Missing field: {field}"

    def test_to_dict_has_formatted_time(self):
        """to_dict should include human-readable formatted_time."""
        from application.domain.observers import Measurement

        mock_gps = Mock()
        mock_gps.latitude = 41.9
        mock_gps.longitude = 12.5
        mock_gps.speed = 30
        mock_gps.heading = 180
        mock_gps.occupancy_status = 1

        m = Measurement(
            id=1,
            autobus_id="bus_123",
            next_stop="stop_1",
            next_stop_distance=150.5,
            gpsdata=mock_gps,
            trip_id="trip_123",
            weather=None,
            occupancy_status=1,
            speed_ratio=1.0,
            current_speed=30.0,
            derived_speed=30.0,
            derived_bearing=180,
            is_in_preferential=False,
        )
        d = m.to_dict("trip_123")

        assert d["formatted_time"] is not None
        assert isinstance(d["formatted_time"], str)


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
        # Should contain colons (HH:MM:SS format)
        assert ":" in result
