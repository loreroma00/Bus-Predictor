"""
Pytest Configuration and Fixtures.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_gps_data():
    """Create a mock GPS data object."""
    from unittest.mock import Mock

    gps = Mock()
    gps.latitude = 41.9028
    gps.longitude = 12.4964
    gps.speed = 25.5
    gps.heading = 180
    gps.current_status = 1
    gps.next_stop_id = "stop_123"
    gps.occupancy_status = 2
    return gps


@pytest.fixture
def mock_live_trip():
    """Create a mock LiveTrip object."""
    from unittest.mock import Mock

    live_trip = Mock()
    live_trip.id = 12345
    live_trip.hexagon_id = "8a1234567890fff"
    live_trip.location_name = "Via Roma"
    live_trip.occupancy_status = 2
    live_trip.derived_speed = 25.0
    live_trip.derived_bearing = 180.0
    live_trip.is_in_preferential = False
    return live_trip


@pytest.fixture
def mock_observatory():
    """Create a mock Observatory."""
    from unittest.mock import Mock

    obs = Mock()
    obs.search_live_trip.return_value = None
    obs.search_completed_live_trip.return_value = None
    obs.get_all_current_measurements.return_value = ([], 0, 0)
    obs.get_completed_measurements.return_value = []
    return obs


@pytest.fixture
def fresh_event_bus():
    """Create a fresh ConsoleEventBus for testing."""
    from application.domain.internal_events import ConsoleEventBus

    return ConsoleEventBus()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir
