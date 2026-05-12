import os
import pandas as pd
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from application.post_processing import data_cleaning
from application.domain.live_data import GPSData, Measurement
from application.domain.weather import Weather


def create_diary(trip_id: str):
    """Create a LiveTrip-like object for pipeline tests."""
    return SimpleNamespace(trip_id=trip_id, measurements=[])


# Helper to create a dummy Measurement
def create_measurement(
    id,
    trip_id,
    lat,
    lon,
    time_val,
    speed=10,
    next_stop=None,
    derived_speed=None,
    speed_ratio=1.0,
    current_speed=10.0,
):
    """Create a measurement."""
    # Mock/Stub GPSData
    gps = MagicMock(spec=GPSData)
    gps.latitude = lat
    gps.longitude = lon
    gps.speed = speed
    gps.timestamp = time_val
    gps.heading = 0
    gps.current_stop_sequence = 1

    # Mock/Stub Weather
    weather = MagicMock(spec=Weather)
    weather.weather_code = 0

    m = Measurement(
        id=id,
        vehicle_id="bus_123",
        next_stop=next_stop,
        next_stop_distance=100,
        gpsdata=gps,
        trip_id=str(trip_id),
        weather=weather,
        occupancy_status=1,
        speed_ratio=speed_ratio,
        current_speed=current_speed,
        derived_speed=derived_speed if derived_speed is not None else speed,
        derived_bearing=0,
        is_in_preferential=False,
        measurement_time=time_val,
    )
    return m


class TestDataCleaning:
    """Testdatacleaning."""
    def test_check_for_duplicates_diary(self):
        """Test _check_for_duplicates with a LiveTrip-like object."""
        diary = create_diary("trip_1")

        m1 = create_measurement("1", "trip_1", 10.0, 10.0, 1000)
        m2 = create_measurement("2", "trip_1", 10.0, 10.0, 1000)
        m3 = create_measurement("3", "trip_1", 10.1, 10.1, 1002)

        diary.measurements = [m1, m2, m3]

        cleaned_diary = data_cleaning._check_for_duplicates(diary)

        assert len(cleaned_diary.measurements) == 2
        assert cleaned_diary.measurements[0].id == "1"
        assert cleaned_diary.measurements[1].id == "3"

    def test_same_position_different_time_is_not_duplicate(self):
        """Standing still is valid and must not be collapsed as a duplicate."""
        diary = create_diary("trip_1")
        m1 = create_measurement("1", "trip_1", 10.0, 10.0, 1000)
        m2 = create_measurement("2", "trip_1", 10.0, 10.0, 1010)

        diary.measurements = [m1, m2]
        cleaned_diary = data_cleaning._check_for_duplicates(diary)

        assert [m.id for m in cleaned_diary.measurements] == ["1", "2"]

    def test_integrity_drops_bad_speed_ratio_and_uptime(self):
        """Integrity keeps only finite positive speed_ratio values seen while online."""
        good = create_measurement("1", "trip_1", 10.0, 10.0, 1000)
        zero_ratio = create_measurement(
            "2", "trip_1", 10.1, 10.1, 1010, speed_ratio=0.0
        )
        offline = create_measurement("3", "trip_1", 10.2, 10.2, 2000)

        cleaned = data_cleaning._check_data_integrity(
            {"trip_1": [good, zero_ratio, offline]},
            uptime_timestamps=[995, 1005],
            served_ratio=1.0,
        )

        assert [m.id for m in cleaned["trip_1"]] == ["1"]

    def test_process_parquet_file(self):
        """Test _process_parquet_file by creating a real temporary parquet file."""

        # Create a dataframe mimicking the schema
        data = {
            "stop_id": ["1", "2"],
            "trip_id": ["trip_A", "trip_B"],
            "next_stop": ["stop_X", "stop_Y"],
            "next_stop_distance": [100.0, 200.0],
            "gps_timestamp": [1000.0, 2000.0],
            "lat": [10.0, 11.0],
            "lon": [10.0, 11.0],
            "speed": [30.0, 40.0],
            "bearing": [90.0, 180.0],
            "measurement_time": [1000.0, 2000.0],
            "occupancy": [1, 2],
            "speed_ratio": [0.8, 0.9],
            "traffic_speed": [25.0, 35.0],
            "derived_speed": [28.0, 38.0],
            "derived_bearing": [90.0, 180.0],
            "is_in_preferential": [False, True],
            # Weather
            "temperature": [20.0, 21.0],
            "apparent_temperature": [20.0, 21.0],
            "humidity": [50.0, 55.0],
            "precip_intensity": [0.0, 0.1],
            "wind_speed": [5.0, 6.0],
            "weather_code": [0, 1],
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            df.to_parquet(tmp_path)

            diaries = data_cleaning._process_parquet_file(tmp_path)

            assert len(diaries) == 2
            # Sort by trip_id to ensure order
            diaries.sort(key=lambda d: d.trip_id)

            d_a = diaries[0]
            assert d_a.trip_id == "trip_A"
            assert len(d_a.measurements) == 1
            m_a = d_a.measurements[0]
            assert m_a.id == "1"
            assert m_a.gpsdata.latitude == 10.0

            d_b = diaries[1]
            assert d_b.trip_id == "trip_B"
            assert len(d_b.measurements) == 1
            m_b = d_b.measurements[0]
            assert m_b.id == "2"
            assert m_b.is_in_preferential is True

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_overload_behavior(self):
        """Test that functions accept dicts as well (backward compatibility)."""
        m1 = create_measurement("1", "trip_1", 10.0, 10.0, 1000)
        m2 = create_measurement("2", "trip_1", 10.1, 10.1, 1010)
        data_dict = {"trip_1": [m1, m2]}

        res = data_cleaning._check_data_integrity(
            data_dict, uptime_timestamps=[], served_ratio=1.0
        )
        assert isinstance(res, dict)
        assert "trip_1" in res

    def test_check_for_duplicates_dict(self):
        """Test _check_for_duplicates with dict input."""
        m1 = create_measurement("1", "trip_1", 10.0, 10.0, 1000)
        m2 = create_measurement("2", "trip_1", 10.0, 10.0, 1000)
        m3 = create_measurement("3", "trip_1", 10.1, 10.1, 1002)

        data_dict = {"trip_1": [m1, m2, m3]}

        cleaned_dict = data_cleaning._check_for_duplicates(data_dict)

        assert isinstance(cleaned_dict, dict)
        assert len(cleaned_dict["trip_1"]) == 2
        ids = [m.id for m in cleaned_dict["trip_1"]]
        assert "1" in ids
        assert "3" in ids

    def test_vehicle_pipeline(self):
        """Test VehiclePipeline vectorization."""
        from application.post_processing.data_cleaning import VehiclePipeline
        from application.post_processing.vectorization import (
            VehicleVector,
            VehicleLabel,
        )

        diary = create_diary("trip_1")

        # Create a measurement with bus_type
        m1 = create_measurement("1", "trip_1", 10.0, 10.0, 1000)
        m1.bus_type = 1
        m2 = create_measurement("2", "trip_1", 10.1, 10.1, 1010)
        m2.bus_type = 1

        diary.measurements = [m1, m2]

        # Mock topology with proper route structure
        mock_route = SimpleNamespace(id="route_A")
        mock_trip = SimpleNamespace(
            id="trip_1", route=mock_route, direction_id=0, direction_name="Outbound"
        )
        from application.domain.virtual_entities import TopologyLedger
        topology = TopologyLedger(trips={"trip_1": mock_trip})

        with patch(
            "application.post_processing.data_cleaning._load_uptime_data",
            return_value=[],
        ):
            pipeline = VehiclePipeline(
                diary=diary,
                topology=topology,
                vehicle_type_name="DieselBus",
            )

            vectors = pipeline.clean()

        assert len(vectors) == 1
        vec, label, ts = vectors[0]

        assert isinstance(vec, VehicleVector)
        assert isinstance(label, VehicleLabel)
        assert vec.trip_id == "trip_1"
        assert vec.route_id == "route_A"
        assert label.vehicle_type == "DieselBus"
