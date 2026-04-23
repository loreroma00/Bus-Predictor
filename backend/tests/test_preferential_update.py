from unittest.mock import Mock
from application.domain.cities import City, Hexagon
from application.domain.live_data import Autobus


class TestPreferentialUpdate:
    """Test `is_in_preferential` update logic in City."""

    def test_preferential_update_on_add_and_move(self):
        # 1. Setup City and Hexagon with preferential lane
        """Test: preferential update on add and move."""
        city = City("TestCity")
        hex_id = "892d3fbe257ffff"
        hexagon = Hexagon(hex_id)

        # Set preferential lane at 90 degrees
        hexagon.preferentials = [90.0]
        city.add_hexagon(hexagon)

        # 2. Add Bus 1 (Aligned)
        bus1 = Autobus(id="BUS1", trip=Mock())
        # Manually set bearing since strictly we need GPSData to derive it usually,
        # but get_bearing() returns derived_bearing.
        bus1.derived_bearing = 90.0

        city.add_bus_to_city(bus1, hex_id=hex_id)

        assert bus1.get_is_in_preferential() is True, (
            "Bus 1 should be in preferential lane upon addition"
        )

        # 3. Add Bus 2 (Misaligned)
        bus2 = Autobus(id="BUS2", trip=Mock())
        bus2.derived_bearing = 180.0

        city.add_bus_to_city(bus2, hex_id=hex_id)

        assert bus2.get_is_in_preferential() is False, (
            "Bus 2 should NOT be in preferential lane upon addition"
        )

        # 4. Update Bus 2 Bearing to match (90.0)
        bus2.derived_bearing = 90.0

        # 5. Move Bus 2 (Stay in same hex)
        # This mirrors the update loop: calculate new position (same hex) -> move_bus
        city.move_bus("BUS2", new_hex_id=hex_id)

        assert bus2.get_is_in_preferential() is True, (
            "Bus 2 should detected preferential lane after move (same hex)"
        )
