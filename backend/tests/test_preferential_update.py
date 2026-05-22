from application.domain.cities import City, Hexagon


class DummyLiveTrip:
    """Small stand-in for the City live-trip contract."""

    def __init__(self, live_trip_id: str, bearing: float):
        self.id = live_trip_id
        self.derived_bearing = bearing
        self._is_in_preferential = False

    def get_bearing(self):
        return self.derived_bearing

    def set_is_in_preferential(self, value: bool):
        self._is_in_preferential = value

    def get_is_in_preferential(self):
        return self._is_in_preferential


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

        # 2. Add LiveTrip 1 (aligned)
        live_trip1 = DummyLiveTrip("TRIP1", 90.0)

        city.add_live_trip_to_city(live_trip1, hex_id=hex_id)

        assert live_trip1.get_is_in_preferential() is True, (
            "LiveTrip 1 should be in preferential lane upon addition"
        )

        # 3. Add LiveTrip 2 (misaligned)
        live_trip2 = DummyLiveTrip("TRIP2", 180.0)

        city.add_live_trip_to_city(live_trip2, hex_id=hex_id)

        assert live_trip2.get_is_in_preferential() is False, (
            "LiveTrip 2 should NOT be in preferential lane upon addition"
        )

        # 4. Update LiveTrip 2 bearing to match (90.0)
        live_trip2.derived_bearing = 90.0

        # 5. Move LiveTrip 2 (stay in same hex)
        city.move_live_trip("TRIP2", new_hex_id=hex_id)

        assert live_trip2.get_is_in_preferential() is True, (
            "LiveTrip 2 should detect preferential lane after move (same hex)"
        )
