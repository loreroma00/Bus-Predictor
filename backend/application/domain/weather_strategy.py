"""
Weather Update Strategies - Control how hexagon weather is fetched from Open-Meteo.

Strategies are discovered by name via get_weather_strategy() factory.
"""

import logging
import requests
from abc import ABC, abstractmethod
from .weather import Weather
from . import h3_utils


CURRENT_FIELDS = "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,apparent_temperature"
HOURLY_FIELDS = "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,apparent_temperature,precipitation_probability"


def apply_weather_to_hexagon(hexagon, data: dict):
    """Parse Open-Meteo API response and update a hexagon's current + forecast weather."""
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


def _fetch_and_apply(hexagons: list, weather_url: str):
    """Batch-fetch weather for a list of hexagons and apply results."""
    if not hexagons:
        return

    lats = []
    lons = []
    for hexagon in hexagons:
        coords = h3_utils.get_coords_from_h3(hexagon.hex_id)
        lats.append(str(coords[0]))
        lons.append(str(coords[1]))

    params = {
        "latitude": ",".join(lats),
        "longitude": ",".join(lons),
        "current": CURRENT_FIELDS,
        "hourly": HOURLY_FIELDS,
        "wind_speed_unit": "ms",
        "timeformat": "unixtime",
        "timezone": "auto",
    }

    response = requests.get(weather_url, params=params, timeout=30)
    response.raise_for_status()
    results = response.json()

    # Single location returns a dict, multiple returns a list
    if isinstance(results, dict):
        results = [results]

    for hexagon, data in zip(hexagons, results):
        try:
            apply_weather_to_hexagon(hexagon, data)
        except Exception as e:
            logging.warning(f"Failed to parse weather for hex {hexagon.hex_id}: {e}")


class WeatherUpdateStrategy(ABC):
    """Base class for weather update strategies."""

    strategy_name: str

    @abstractmethod
    def update(self, city, weather_url: str) -> None:
        """Fetch weather and apply to city hexagons."""
        pass


class GreedyWeatherStrategy(WeatherUpdateStrategy):
    """Update ALL hexagons in a single batch API call every cycle."""

    strategy_name = "greedy"

    def update(self, city, weather_url: str) -> None:
        """Fetch weather for every hexagon in the city and apply it."""
        hex_snapshot = list(city.hexagons.values())
        _fetch_and_apply(hex_snapshot, weather_url)
        logging.info(f"Greedy weather update: {len(hex_snapshot)} hexagons")


class SubsetWeatherStrategy(WeatherUpdateStrategy):
    """Rotate through hexagon subsets, updating 1/N per cycle."""

    strategy_name = "subset"

    def __init__(self, n_subsets: int = 4):
        """Store the partition size and the rotating subset index."""
        self.n_subsets = max(1, n_subsets)
        self._current_index = 0

    def update(self, city, weather_url: str) -> None:
        """Fetch weather for the next 1/N subset of hexagons and advance the rotation."""
        hex_snapshot = list(city.hexagons.values())
        if not hex_snapshot:
            return

        subset = hex_snapshot[self._current_index::self.n_subsets]
        subset_index = self._current_index
        self._current_index = (self._current_index + 1) % self.n_subsets

        _fetch_and_apply(subset, weather_url)
        logging.info(
            f"Subset weather update: {len(subset)}/{len(hex_snapshot)} hexagons "
            f"(subset {subset_index + 1}/{self.n_subsets})"
        )


def get_weather_strategy(name: str, n_subsets: int = 4) -> WeatherUpdateStrategy:
    """Factory: get a weather update strategy by name."""
    for subclass in WeatherUpdateStrategy.__subclasses__():
        if hasattr(subclass, "strategy_name") and subclass.strategy_name == name:
            if subclass is SubsetWeatherStrategy:
                return subclass(n_subsets=n_subsets)
            return subclass()

    available = get_available_weather_strategies()
    logging.warning(f"Unknown weather strategy '{name}'. Available: {available}")
    logging.info("Defaulting to 'greedy'")
    return GreedyWeatherStrategy()


def get_available_weather_strategies() -> list[str]:
    """List all available weather strategy names."""
    return [
        s.strategy_name
        for s in WeatherUpdateStrategy.__subclasses__()
        if hasattr(s, "strategy_name")
    ]
