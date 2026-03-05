"""
Weather Service - Fetches and caches weather for City hexagons.

Uses Open-Meteo API with 15-minute cache TTL.
Supports querying by lat/lon coordinates OR hexagon_id.
"""

import logging
import time
import requests
from typing import Optional
from datetime import datetime

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL = 900  # 15 minutes


class WeatherService:
    """
    Fetches weather for City hexagons with 15-minute cache.

    Uses City singleton to store/update weather per hexagon.
    Can query by coordinates (lat/lon) or hexagon_id.
    """

    def __init__(self, city):
        self._city = city
        self._last_update: float = 0

    def get_weather(self, lat: float = None, lon: float = None, hex_id: str = None):
        """
        Get weather for coordinates OR hexagon_id.

        Args:
            lat: Latitude (optional, used with lon)
            lon: Longitude (optional, used with lat)
            hex_id: H3 hexagon ID (optional, alternative to lat/lon)

        Returns:
            Weather object for the specified location.

        Raises:
            ValueError: If neither valid coords nor hex_id provided.
        """
        from application.domain import h3_utils
        from application.domain.weather import Weather

        if hex_id:
            target_hex = hex_id
            coords = h3_utils.get_coords_from_h3(hex_id)
            target_lat, target_lon = coords[0], coords[1]
        elif lat is not None and lon is not None:
            target_hex = h3_utils.get_h3_index(lat, lon)
            target_lat, target_lon = lat, lon
        else:
            target_hex = self._get_default_hex()
            coords = h3_utils.get_coords_from_h3(target_hex)
            target_lat, target_lon = coords[0], coords[1]

        hexagon = self._city.get_hexagon(target_hex)
        now = time.time()

        if hexagon and now - self._last_update <= CACHE_TTL:
            weather = hexagon.get_weather()
            if weather and weather.temperature is not None:
                return weather

        return self._fetch_weather(target_hex, target_lat, target_lon)

    def get_weather_for_datetime(
        self,
        target_dt: datetime,
        lat: float = None,
        lon: float = None,
        hex_id: str = None,
    ):
        """Get weather for a target datetime using hourly forecast buckets."""
        from application.domain import h3_utils
        from application.domain.weather import Weather

        if hex_id:
            target_hex = hex_id
            coords = h3_utils.get_coords_from_h3(hex_id)
            target_lat, target_lon = coords[0], coords[1]
        elif lat is not None and lon is not None:
            target_hex = h3_utils.get_h3_index(lat, lon)
            target_lat, target_lon = lat, lon
        else:
            target_hex = self._get_default_hex()
            coords = h3_utils.get_coords_from_h3(target_hex)
            target_lat, target_lon = coords[0], coords[1]

        hexagon = self._city.get_hexagon(target_hex)
        now = time.time()
        hour_bucket = int(target_dt.timestamp() // 3600)

        if hexagon and now - self._last_update <= CACHE_TTL:
            weather_for_hour = hexagon.get_weather_for_hour_bucket(hour_bucket)
            if weather_for_hour is not None:
                return weather_for_hour

            current_weather = hexagon.get_weather()
            if current_weather and current_weather.temperature is not None:
                return current_weather

        self._fetch_weather(target_hex, target_lat, target_lon)
        hexagon = self._city.get_hexagon(target_hex)
        if hexagon:
            weather_for_hour = hexagon.get_weather_for_hour_bucket(hour_bucket)
            if weather_for_hour is not None:
                return weather_for_hour

            current_weather = hexagon.get_weather()
            if current_weather and current_weather.temperature is not None:
                return current_weather

        return Weather(
            valid_time=time.time(),
            temperature=0,
            apparent_temperature=0,
            humidity=0,
            precip_intensity=0,
            wind_speed=0,
            weather_code=0,
            forecast_probability=Weather.FORECAST_PROBABILITY_UNKNOWN,
            is_forecast=False,
        )

    def _get_default_hex(self) -> str:
        """Get default hexagon (Rome center)."""
        from application.domain import h3_utils

        return h3_utils.get_h3_index(41.9028, 12.4964)

    def _fetch_weather(self, hex_id: str, lat: float, lon: float):
        """
        Fetch fresh weather from Open-Meteo and update hexagon.

        Args:
            hex_id: Target hexagon ID
            lat: Latitude
            lon: Longitude

        Returns:
            Weather object with fetched data.
        """
        from application.domain.weather import Weather

        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,apparent_temperature",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,apparent_temperature,precipitation_probability",
            "wind_speed_unit": "ms",
            "timeformat": "unixtime",
            "timezone": "auto",
        }

        try:
            response = requests.get(WEATHER_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            current = data["current"]

            interval = current.get("interval", 900)
            precip_amount = current["precipitation"]
            precip_intensity = precip_amount * (3600 / interval)

            weather = Weather(
                valid_time=current["time"],
                temperature=current["temperature_2m"],
                apparent_temperature=current["apparent_temperature"],
                humidity=current["relative_humidity_2m"],
                precip_intensity=precip_intensity,
                wind_speed=current["wind_speed_10m"],
                weather_code=current["weather_code"],
                forecast_probability=1.0,
                is_forecast=False,
            )

            self._city.add_hexagon_with_coords(lat, lon)
            hexagon = self._city.get_hexagon(hex_id)
            if hexagon:
                hexagon.set_weather(weather)

                hourly = data.get("hourly", {}) or {}
                hourly_times = hourly.get("time", []) or []
                hourly_temps = hourly.get("temperature_2m", []) or []
                hourly_apparent = hourly.get("apparent_temperature", []) or []
                hourly_humidity = hourly.get("relative_humidity_2m", []) or []
                hourly_precip = hourly.get("precipitation", []) or []
                hourly_wind = hourly.get("wind_speed_10m", []) or []
                hourly_code = hourly.get("weather_code", []) or []
                hourly_precip_prob = hourly.get("precipitation_probability", []) or []

                forecast_items = []
                length = len(hourly_times)
                for i in range(length):
                    ts = hourly_times[i]
                    if ts is None:
                        continue
                    bucket = int(ts // 3600)
                    probability = Weather.FORECAST_PROBABILITY_UNKNOWN
                    if (
                        i < len(hourly_precip_prob)
                        and hourly_precip_prob[i] is not None
                    ):
                        probability = float(hourly_precip_prob[i]) / 100.0

                    forecast_weather = Weather(
                        valid_time=ts,
                        temperature=hourly_temps[i] if i < len(hourly_temps) else 0,
                        apparent_temperature=hourly_apparent[i]
                        if i < len(hourly_apparent)
                        else 0,
                        humidity=hourly_humidity[i] if i < len(hourly_humidity) else 0,
                        precip_intensity=hourly_precip[i]
                        if i < len(hourly_precip)
                        else 0,
                        wind_speed=hourly_wind[i] if i < len(hourly_wind) else 0,
                        weather_code=hourly_code[i] if i < len(hourly_code) else 0,
                        forecast_probability=probability,
                        is_forecast=True,
                    )
                    forecast_items.append({bucket: forecast_weather})

                hexagon.set_weather_forecast(forecast_items)

            self._last_update = time.time()
            logging.info(
                f"Weather updated for hex {hex_id}: {weather.temperature}C, code {weather.weather_code}"
            )

            return weather

        except Exception as e:
            logging.error(f"Failed to fetch weather: {e}")

            self._city.add_hexagon_with_coords(lat, lon)
            hexagon = self._city.get_hexagon(hex_id)
            if hexagon:
                existing = hexagon.get_weather()
                if existing:
                    return existing

            return Weather(
                valid_time=time.time(),
                temperature=0,
                apparent_temperature=0,
                humidity=0,
                precip_intensity=0,
                wind_speed=0,
                weather_code=0,
                forecast_probability=Weather.FORECAST_PROBABILITY_UNKNOWN,
                is_forecast=False,
            )

    def invalidate_cache(self):
        """Force cache invalidation on next request."""
        self._last_update = 0
