from .live_data import Autobus as Bus
from .weather import Weather
from . import h3_utils
from typing import overload, Callable
import requests
import time

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


class Traffic:
    def __init__(
        self,
        north_speed: float = 0,
        north_east_speed: float = 0,
        east_speed: float = 0,
        south_east_speed: float = 0,
        south_speed: float = 0,
        south_west_speed: float = 0,
        west_speed: float = 0,
        north_west_speed: float = 0,
        north_speed_ratio: float = 0,
        north_east_speed_ratio: float = 0,
        east_speed_ratio: float = 0,
        south_east_speed_ratio: float = 0,
        south_speed_ratio: float = 0,
        south_west_speed_ratio: float = 0,
        west_speed_ratio: float = 0,
        north_west_speed_ratio: float = 0,
    ):
        self.traffic_dict = {
            "N": {
                "current_speed": north_speed,
                "speed_ratio": north_speed_ratio,
                "flow_speed": north_speed / north_speed_ratio
                if north_speed_ratio > 0
                else 0,
            },
            "NE": {
                "current_speed": north_east_speed,
                "speed_ratio": north_east_speed_ratio,
                "flow_speed": north_east_speed / north_east_speed_ratio
                if north_east_speed_ratio > 0
                else 0,
            },
            "E": {
                "current_speed": east_speed,
                "speed_ratio": east_speed_ratio,
                "flow_speed": east_speed / east_speed_ratio
                if east_speed_ratio > 0
                else 0,
            },
            "SE": {
                "current_speed": south_east_speed,
                "speed_ratio": south_east_speed_ratio,
                "flow_speed": south_east_speed / south_east_speed_ratio
                if south_east_speed_ratio > 0
                else 0,
            },
            "S": {
                "current_speed": south_speed,
                "speed_ratio": south_speed_ratio,
                "flow_speed": south_speed / south_speed_ratio
                if south_speed_ratio > 0
                else 0,
            },
            "SW": {
                "current_speed": south_west_speed,
                "speed_ratio": south_west_speed_ratio,
                "flow_speed": south_west_speed / south_west_speed_ratio
                if south_west_speed_ratio > 0
                else 0,
            },
            "W": {
                "current_speed": west_speed,
                "speed_ratio": west_speed_ratio,
                "flow_speed": west_speed / west_speed_ratio
                if west_speed_ratio > 0
                else 0,
            },
            "NW": {
                "current_speed": north_west_speed,
                "speed_ratio": north_west_speed_ratio,
                "flow_speed": north_west_speed / north_west_speed_ratio
                if north_west_speed_ratio > 0
                else 0,
            },
        }

    def set_current_speed(self, direction: str, current_speed: float):
        self.traffic_dict[direction]["current_speed"] = current_speed

    def _set_flow_speed(self, direction: str):
        if (
            self.traffic_dict[direction]["speed_ratio"] > 0
            and self.traffic_dict[direction]["current_speed"] > 0
        ):
            self.traffic_dict[direction]["flow_speed"] = (
                self.traffic_dict[direction]["current_speed"]
                / self.traffic_dict[direction]["speed_ratio"]
            )
        elif self.traffic_dict[direction]["current_speed"] > 0:
            self.traffic_dict[direction]["flow_speed"] = -1  # Unknown

    def set_speed_ratio(self, direction: str, speed_ratio: float):
        """
        Set traffic congestion from the relative speed ratio (0-1).

        Args:
            speed_ratio: Fraction of free-flow speed (0.6 = 60% of normal speed)
                         A lower value means MORE congestion.

        This also computes flow_speed = current_speed / speed_ratio.
        """
        # Store speed_ratio directly
        self.traffic_dict[direction]["speed_ratio"] = speed_ratio
        self._set_flow_speed(direction)

    def get_flow_speed(self, direction: str) -> float:
        return self.traffic_dict[direction]["flow_speed"]

    def get_current_speed(self, direction: str) -> float:
        return self.traffic_dict[direction]["current_speed"]

    def get_speed_ratio(self, direction: str) -> float:
        return self.traffic_dict[direction]["speed_ratio"]


class Hexagon:
    def __init__(self, hex_id: str):
        self.hex_id = hex_id
        self.streets: dict[tuple[float, float], str] = {}  # (lat, lng) -> street name
        self.neighbours = []
        self.mobile_agents = {
            "buses": {},  # str -> bus
        }
        self.weather = Weather(0, 0, 0, 0, 0, 0, 0)
        self.weather_forecast_by_hour: list[dict[int, Weather]] = []
        self.preferentials: list[float] = []
        self.traffic = Traffic()
        self.last_traffic_update: float = 0  # Unix timestamp of last traffic update

    def reset_traffic_ttl(self):
        """Reset traffic TTL to current time after fresh data is received."""
        self.last_traffic_update = time.time()

    def is_traffic_expired(self, ttl_seconds: int = 900) -> bool:
        """Check if traffic data is expired (default 15 min TTL)."""
        return (time.time() - self.last_traffic_update) > ttl_seconds

    def add_bus(self, bus: Bus):
        self.mobile_agents["buses"][str(bus.id)] = bus

    def remove_bus(self, bus_id: str) -> Bus | None:
        return self.mobile_agents["buses"].pop(str(bus_id), None)

    def get_bus(self, bus_id: str) -> Bus | None:
        return self.mobile_agents["buses"].get(str(bus_id), None)

    def add_street(self, lat: float, lng: float, street_name: str):
        """Add street name at coords. Coords are rounded to 5 decimals for consistent cache keys."""
        key = (round(lat, 5), round(lng, 5))
        self.streets[key] = street_name

    def get_street_by_name(self, street_name: str) -> tuple[float, float] | None:
        for (lat, lng), name in self.streets.items():
            if name == street_name:
                return (lat, lng)
        return None

    def get_street_by_coords(self, lat: float, lng: float) -> str | None:
        """Get street name at coords. Coords are rounded to 5 decimals for lookup."""
        key = (round(lat, 5), round(lng, 5))
        return self.streets.get(key, None)

    def get_weather(self) -> Weather:
        return self.weather

    def set_weather(self, weather: Weather):
        self.weather = weather
        hour_bucket = int(float(weather.time) // 3600)
        self.set_weather_for_hour_bucket(hour_bucket, weather)

    def set_weather_for_hour_bucket(self, hour_bucket: int, weather: Weather):
        for item in self.weather_forecast_by_hour:
            if hour_bucket in item:
                item[hour_bucket] = weather
                return
        self.weather_forecast_by_hour.append({hour_bucket: weather})

    def set_weather_forecast(self, weather_forecast: list[dict[int, Weather]]):
        self.weather_forecast_by_hour = weather_forecast

    def get_weather_for_hour_bucket(self, hour_bucket: int) -> Weather | None:
        for item in self.weather_forecast_by_hour:
            weather = item.get(hour_bucket)
            if weather is not None:
                return weather
        return None

    def get_temperature(self) -> float:
        return self.weather.temperature

    def set_temperature(self, temperature: float):
        self.weather.temperature = temperature

    def get_humidity(self) -> float:
        return self.weather.humidity

    def set_humidity(self, humidity: float):
        self.weather.humidity = humidity

    def get_wind_speed(self) -> float:
        return self.weather.wind_speed

    def set_wind_speed(self, wind_speed: float):
        self.weather.wind_speed = wind_speed

    def get_precip_intensity(self) -> float:
        return self.weather.precip_intensity

    def set_precip_intensity(self, precip_intensity: float):
        self.weather.precip_intensity = precip_intensity

    def get_preferentials(self):
        return self.preferentials

    def get_current_speed(self, direction: str = None) -> float:
        """Get current traffic speed. If direction is None, returns average across all directions."""
        if direction:
            return self.traffic.get_current_speed(direction)
        # Average across all directions with data
        speeds = [
            self.traffic.get_current_speed(d)
            for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            if self.traffic.get_current_speed(d) > 0
        ]
        return sum(speeds) / len(speeds) if speeds else 0

    def get_flow_speed(self, direction: str = None) -> float:
        """Get free-flow speed. If direction is None, returns average across all directions."""
        if direction:
            return self.traffic.get_flow_speed(direction)
        speeds = [
            self.traffic.get_flow_speed(d)
            for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            if self.traffic.get_flow_speed(d) > 0
        ]
        return sum(speeds) / len(speeds) if speeds else 0

    def get_speed_ratio(self, direction: str = None) -> float:
        """Get speed ratio (fraction of free-flow). If direction is None, returns average."""
        if direction:
            return self.traffic.get_speed_ratio(direction)
        # Average ratio across all directions with data
        ratios = [
            self.traffic.get_speed_ratio(d)
            for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            if self.traffic.get_speed_ratio(d) > 0
        ]
        return sum(ratios) / len(ratios) if ratios else 0

    def check_alignment(
        self, bus_heading: float | None, tolerance: float = 30.0
    ) -> int:
        """
        Ritorna 1 se l'heading del bus è compatibile con una corsia preferenziale
        presente in questo esagono.
        """
        # Se non ci sono corsie o il bus non ha un angolo valido (es. è fermo), niente protezione.
        if not self.preferentials or bus_heading is None or bus_heading == -1:
            return False

        for lane_angle in self.preferentials:
            # Calcolo della differenza minima su un cerchio (0-360)
            diff = abs(bus_heading - lane_angle)
            diff = min(diff, 360 - diff)

            if diff <= tolerance:
                return True  # MATCH! Sei sulla preferenziale

        return False  # Sei nell'esagono giusto, ma direzione sbagliata (es. incrocio)

    def get_traffic(self) -> Traffic:
        return self.traffic

    def set_traffic(self, traffic: Traffic):
        self.traffic = traffic


class City:
    def __init__(self, name, static_bus_lanes: dict = None):
        self.name = name
        self.hexagons: dict[str, Hexagon] = {}
        self.bus_index: dict[str, str] = {}  # BusID -> HexID (O(1) lookup)
        self.bus_deposit: dict[
            str, Bus
        ] = {}  # BusID -> Bus (O(1) lookup); Where buses go "to die".
        self.static_bus_lanes = static_bus_lanes if static_bus_lanes else {}
        # Injected callback for expired traffic handling (LOW COUPLING)
        self._on_bus_entered_expired_hex: Callable[[str, str], None] = None

    def set_on_bus_entered_expired_hex(self, callback: Callable[[str, str], None]):
        """Inject callback (bus_id, hex_id) for expired traffic handling."""
        self._on_bus_entered_expired_hex = callback

    def get_hexagons_with_buses(self) -> list[str]:
        """Return unique hex IDs that currently contain at least one bus."""
        return list(set(self.bus_index.values()))

    @overload
    def add_bus_to_city(self, bus: Bus, lat: float, lng: float): ...

    @overload
    def add_bus_to_city(self, bus: Bus, hex_id: str): ...

    def add_bus_to_city(
        self, bus: Bus, hex_id: str = None, lat: float = None, lng: float = None
    ):
        """Registers a bus in the city at a specific hexagon."""
        if hex_id is None:
            new_hexagon = Hexagon(h3_utils.get_h3_index(lat, lng))
            self.add_hexagon(new_hexagon)
            hex_id = new_hexagon.hex_id

        if hex_id not in self.hexagons:
            new_hexagon = Hexagon(hex_id)
            self.add_hexagon(new_hexagon)

        self.hexagons[hex_id].add_bus(bus)
        bus.set_is_in_preferential(
            self.hexagons[hex_id].check_alignment(bus.get_bearing())
        )
        self.bus_index[str(bus.id)] = hex_id

        # Resurrect if in deposit
        if str(bus.id) in self.bus_deposit:
            del self.bus_deposit[str(bus.id)]

    @overload
    def move_bus(self, bus_id: str, new_hex_id: str) -> bool: ...

    @overload
    def move_bus(self, bus_id: str, latitude: float, longitude: float) -> bool: ...

    def move_bus(
        self,
        bus_id: str,
        new_hex_id: str = None,
        latitude: float = None,
        longitude: float = None,
    ) -> bool:
        """
        Atomically moves a bus from its current hexagon to a new one.
        Returns True if successful, False if bus not found or new hex invalid.
        """
        if new_hex_id is None:
            new_hex_id = h3_utils.get_h3_index(latitude, longitude)

        current_hex_id = self.bus_index.get(str(bus_id))
        if not current_hex_id:
            return False  # Bus not in city

        if new_hex_id not in self.hexagons:
            new_hexagon = Hexagon(new_hex_id)
            self.add_hexagon(new_hexagon)

        if current_hex_id == new_hex_id:
            # Even if staying in the same hex, update preferential status as bearing might change
            self.hexagons[current_hex_id].get_bus(bus_id).set_is_in_preferential(
                self.hexagons[current_hex_id].check_alignment(
                    self.hexagons[current_hex_id].get_bus(bus_id).get_bearing()
                )
            )
            return True  # Already there

        # 1. Remove from old hex
        bus: Bus = self.hexagons[current_hex_id].remove_bus(str(bus_id))

        if bus:
            # 2. Add to new hex
            self.hexagons[new_hex_id].add_bus(bus)
            bus.set_is_in_preferential(
                self.hexagons[new_hex_id].check_alignment(bus.get_bearing())
            )

            # 3. Update index
            self.bus_index[str(bus_id)] = new_hex_id

            # Safety: Ensure not in deposit
            if str(bus_id) in self.bus_deposit:
                del self.bus_deposit[str(bus_id)]

            # 4. Check if new hexagon has expired traffic data
            if self.hexagons[new_hex_id].is_traffic_expired():
                if self._on_bus_entered_expired_hex:
                    self._on_bus_entered_expired_hex(str(bus_id), new_hex_id)

            return True

        return False

    def add_hexagon(self, hexagon: Hexagon):
        """Add a hexagon if it doesn't exist. Returns existing if already present."""
        if hexagon.hex_id in self.hexagons:
            return  # Don't overwrite existing hexagon (preserves traffic data)
        if hexagon.hex_id in self.static_bus_lanes:
            hexagon.preferentials = self.static_bus_lanes[hexagon.hex_id]
        self.hexagons[hexagon.hex_id] = hexagon

    def add_hexagon_with_coords(self, lat: float, lng: float):
        hex_id = h3_utils.get_h3_index(lat, lng)
        hexagon = Hexagon(hex_id)
        self.add_hexagon(hexagon)

    def get_hex_id(self, lat: float, lng: float) -> str:
        return h3_utils.get_h3_index(lat, lng)

    def get_hexagon(self, hex_id: str):
        return self.hexagons.get(hex_id)

    def get_hexagons(self):
        return self.hexagons

    def get_bus(self, bus_id: str) -> Bus | None:
        hex_id = self.bus_index.get(bus_id)
        if hex_id:
            return self.hexagons[hex_id].mobile_agents["buses"].get(bus_id)
        return None

    def remove_bus(self, bus: Bus) -> tuple[Bus | None, str] | None:
        hex_id = self.bus_index.pop(bus.id, None)
        removed_bus: Bus = None
        if hex_id:
            removed_bus = self.hexagons[hex_id].remove_bus(bus.id)
        return (removed_bus, hex_id)

    def bus_to_deposit(self, bus_id: str):
        # Pop returns keys, so get key first, then pop. Or just pop and process.
        hex_id = self.bus_index.pop(bus_id, None)  # Remove from index immediately

        if hex_id:
            # Remove from Hex returns the object
            bus = self.hexagons[hex_id].remove_bus(bus_id)
            if bus:
                self.bus_deposit[bus_id] = bus

    def is_bus_in_deposit(self, bus_id: str) -> bool:
        """Returns True if the bus is currently in the deposit."""
        return bus_id in self.bus_deposit

    def update_weather(self):
        """Updates current + hourly forecast weather for all hexagons in a single API call."""
        import logging

        # Snapshot to avoid "dict changed size during iteration"
        hex_snapshot = list(self.hexagons.values())
        if not hex_snapshot:
            return

        # Collect coordinates for batch request
        lats = []
        lons = []
        for hexagon in hex_snapshot:
            coords = h3_utils.get_coords_from_h3(hexagon.hex_id)
            lats.append(str(coords[0]))
            lons.append(str(coords[1]))

        params = {
            "latitude": ",".join(lats),
            "longitude": ",".join(lons),
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,apparent_temperature",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,apparent_temperature,precipitation_probability",
            "wind_speed_unit": "ms",
            "timeformat": "unixtime",
            "timezone": "auto",
        }

        response = requests.get(WEATHER_URL, params=params, timeout=30)
        response.raise_for_status()
        results = response.json()

        # Single location returns a dict, multiple returns a list
        if isinstance(results, dict):
            results = [results]

        for hexagon, data in zip(hex_snapshot, results):
            try:
                self._apply_weather_to_hexagon(hexagon, data)
            except Exception as e:
                logging.warning(f"Failed to parse weather for hex {hexagon.hex_id}: {e}")

    @staticmethod
    def _apply_weather_to_hexagon(hexagon, data: dict):
        """Parse API response and update a single hexagon's current + forecast weather."""
        current_data = data["current"]

        interval = current_data.get("interval", 900)
        precip_intensity = current_data["precipitation"] * (3600 / interval)

        weather = Weather(
            valid_time=current_data["time"],
            temperature=current_data["temperature_2m"],
            apparent_temperature=current_data["apparent_temperature"],
            humidity=current_data["relative_humidity_2m"],
            precip_intensity=precip_intensity,
            wind_speed=current_data["wind_speed_10m"],
            weather_code=current_data["weather_code"],
        )
        hexagon.set_weather(weather)

        # Hourly forecast
        hourly = data.get("hourly") or {}
        hourly_times = hourly.get("time") or []
        hourly_temps = hourly.get("temperature_2m") or []
        hourly_apparent = hourly.get("apparent_temperature") or []
        hourly_humidity = hourly.get("relative_humidity_2m") or []
        hourly_precip = hourly.get("precipitation") or []
        hourly_wind = hourly.get("wind_speed_10m") or []
        hourly_code = hourly.get("weather_code") or []
        hourly_prob = hourly.get("precipitation_probability") or []

        forecast_items = []
        for i, ts in enumerate(hourly_times):
            if ts is None:
                continue
            bucket = int(ts // 3600)
            probability = Weather.FORECAST_PROBABILITY_UNKNOWN
            if i < len(hourly_prob) and hourly_prob[i] is not None:
                probability = float(hourly_prob[i]) / 100.0

            fw = Weather(
                valid_time=ts,
                temperature=hourly_temps[i] if i < len(hourly_temps) else 0,
                apparent_temperature=hourly_apparent[i] if i < len(hourly_apparent) else 0,
                humidity=hourly_humidity[i] if i < len(hourly_humidity) else 0,
                precip_intensity=hourly_precip[i] if i < len(hourly_precip) else 0,
                wind_speed=hourly_wind[i] if i < len(hourly_wind) else 0,
                weather_code=hourly_code[i] if i < len(hourly_code) else 0,
                forecast_probability=probability,
                is_forecast=True,
            )
            forecast_items.append({bucket: fw})

        hexagon.set_weather_forecast(forecast_items)

    def get_weather(self, hex_id: str) -> Weather:
        return self.hexagons[hex_id].weather

    def get_weather_for_hour_bucket(
        self, hex_id: str, hour_bucket: int
    ) -> Weather | None:
        hexagon = self.hexagons.get(hex_id)
        if not hexagon:
            return None
        return hexagon.get_weather_for_hour_bucket(hour_bucket)

    def get_bounding_box(self) -> tuple[float, float, float, float] | None:
        """
        Calculate the bounding box from all hexagons in this city.

        Returns:
            Tuple of (min_lat, min_lon, max_lat, max_lon) or None if no hexagons.
        """
        if not self.hexagons:
            return None

        min_lat = float("inf")
        min_lon = float("inf")
        max_lat = float("-inf")
        max_lon = float("-inf")

        for hex_id in self.hexagons:
            lat, lon = h3_utils.get_coords_from_h3(hex_id)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)

        # Add padding (~1km buffer)
        padding = 0.01
        return (
            min_lat - padding,
            min_lon - padding,
            max_lat + padding,
            max_lon + padding,
        )

    def update_traffic(
        self, hex_id: str, direction: str, current_speed: float, speed_ratio: float
    ) -> bool:
        """Update traffic data for a hexagon in a specific direction.

        Args:
            hex_id: The hexagon ID
            direction: Cardinal direction (N, NE, E, SE, S, SW, W, NW)
            current_speed: Current traffic speed in kph
            speed_ratio: Relative speed (0-1), fraction of free-flow speed

        Returns:
            True if update was applied, False otherwise
        """
        if hex_id not in self.hexagons:
            # Create hexagon if it doesn't exist (traffic data available before bus visits)
            self.add_hexagon(Hexagon(hex_id))
        hexagon = self.hexagons[hex_id]
        traffic = hexagon.get_traffic()
        traffic.set_current_speed(direction, current_speed)
        traffic.set_speed_ratio(direction, speed_ratio)
        return True
