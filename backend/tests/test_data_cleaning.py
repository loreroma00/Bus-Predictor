import os
import pytest
import pandas as pd
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from application.post_processing import data_cleaning
from application.domain.observers import Diary, Measurement
from application.domain.live_data import GPSData
from application.domain.weather import Weather


# Helper to create a dummy Measurement
def create_measurement(
    id, trip_id, lat, lon, time_val, speed=10, next_stop=None, derived_speed=None
):
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
        autobus_id="bus_123",
        next_stop=next_stop,
        next_stop_distance=100,
        gpsdata=gps,
        trip_id=str(trip_id),
        weather=weather,
        occupancy_status=1,
        speed_ratio=1.0,
        current_speed=10.0,
        derived_speed=derived_speed if derived_speed is not None else speed,
        derived_bearing=0,
        is_in_preferential=False,
    )
    # Set time manually as __init__ might use current time
    m.measurement_time = time_val
    return m


class TestDataCleaning:
    def test_check_for_duplicates_diary(self):
        """Test _check_for_duplicates with a Diary object."""
        diary = Diary(observer=None, trip_id="trip_1")

        # Add 3 measurements, 2 are identical (lat/lon)
        m1 = create_measurement("1", "trip_1", 10.0, 10.0, 1000)
        m2 = create_measurement("2", "trip_1", 10.0, 10.0, 1001)  # Duplicate pos
        m3 = create_measurement("3", "trip_1", 10.1, 10.1, 1002)

        diary.measurements = [m1, m2, m3]

        cleaned_diary = data_cleaning._check_for_duplicates(diary)

        assert len(cleaned_diary.measurements) == 2
        # m1 and m2 share location, usually first one is kept
        assert cleaned_diary.measurements[0].id == "1"
        assert cleaned_diary.measurements[1].id == "3"

    def test_remove_outliers_diary(self):
        """Test _remove_outliers with a Diary object."""
        diary = Diary(observer=None, trip_id="trip_1")

        # m1: Normal
        # m2: Speed outlier (derived_speed=150)
        # m3: Time gap outlier (gap > 600s)

        m1 = create_measurement("1", "trip_1", 10.0, 10.0, 1000, derived_speed=50)
        m2 = create_measurement(
            "2", "trip_1", 10.0, 10.1, 1060, derived_speed=150
        )  # Speed outlier
        m3 = create_measurement(
            "3", "trip_1", 10.2, 10.2, 2000, derived_speed=50
        )  # Time gap outlier (1060 -> 2000 is 940s)

        diary.measurements = [m1, m2, m3]

        cleaned_diary = data_cleaning._remove_outliers(
            diary, max_bus_speed_kmh=120, max_time_gap_seconds=600
        )

        # m2 should be removed (speed)
        # m3 should NOT be removed? Wait, logic:
        # if i > 0 and i < len - 1: check gaps.
        # m1 (index 0): start
        # m2 (index 1): speed check -> fail.
        # m3 (index 2): end.

        # If m2 is removed inside the loop, does it affect m3?
        # The loop iterates over original sorted list.
        # m2 fails speed check.
        # m3 is last, so gap check logic: `if i > 0 and i < len(sorted_m) - 1`
        # m3 is at index 2, len is 3. 2 < 2 is False. So m3 gap check skipped (it's end of trip).

        # So expected: m1 kept, m2 dropped, m3 kept.
        ids = [m.id for m in cleaned_diary.measurements]
        assert "2" not in ids
        assert "1" in ids
        assert "3" in ids

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

        # _remove_outliers with dict
        res = data_cleaning._remove_outliers(data_dict)
        assert isinstance(res, dict)
        assert "trip_1" in res

        # _check_for_runs with dict
        # Default min is 7, we have 1 -> should be dropped
        res_runs = data_cleaning._check_for_runs(data_dict, min_measurements_per_run=7)
        assert isinstance(res_runs, dict)
        assert len(res_runs) == 0  # Dropped

    def test_check_for_duplicates_dict(self):
        """Test _check_for_duplicates with dict input."""
        m1 = create_measurement("1", "trip_1", 10.0, 10.0, 1000)
        m2 = create_measurement("2", "trip_1", 10.0, 10.0, 1001)  # Duplicate pos
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

        diary = Diary(observer=None, trip_id="trip_1")

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
        from application.domain.ledgers import TopologyLedger
        topology = TopologyLedger(trips={"trip_1": mock_trip})

        with patch(
            "application.post_processing.data_cleaning._load_uptime_data",
            return_value=[],
        ):
            pipeline = VehiclePipeline(
                diary=diary,
                topology=topology,
                config={"data_cleaning": {"min_measurements_per_run": 1}},
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
