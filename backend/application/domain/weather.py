"""Weather domain object mapping Open-Meteo WMO codes to human descriptions."""

import time

WEATHER_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Dense Drizzle",
    56: "Light Freezing Drizzle",
    57: "Dense Freezing Drizzle",
    61: "Slight Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    66: "Light Freezing Rain",
    67: "Heavy Freezing Rain",
    71: "Slight Snow fall",
    73: "Moderate Snow fall",
    75: "Heavy Snow fall",
    77: "Snow grains",
    80: "Slight Rain showers",
    81: "Moderate Rain showers",
    82: "Violent Rain showers",
    85: "Slight Snow showers",
    86: "Heavy Snow showers",
    95: "Thunderstorm: Slight or moderate",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class Weather:
    """Point-in-time weather observation or forecast for a hexagon."""

    FORECAST_PROBABILITY_UNKNOWN = -1.0

    def __init__(
        self,
        valid_time: float,
        temperature: float,
        apparent_temperature: float,
        humidity: float,
        precip_intensity: float,
        wind_speed: float,
        weather_code: int,
        forecast_probability: float = FORECAST_PROBABILITY_UNKNOWN,
        is_forecast: bool = False,
    ):
        """Store weather fields, defaulting ``valid_time`` to now and normalising weather_code to int."""
        self.time = valid_time if valid_time else time.time()
        self.temperature = temperature  # in °C
        self.apparent_temperature = apparent_temperature  # in °C
        self.humidity = humidity  # in %
        self.precip_intensity = precip_intensity  # in mm/h
        self.wind_speed = wind_speed  # in m/s
        self.weather_code = int(weather_code) if weather_code is not None else 0
        self.forecast_probability = (
            float(forecast_probability)
            if forecast_probability is not None
            else self.FORECAST_PROBABILITY_UNKNOWN
        )
        self.is_forecast = bool(is_forecast)

    @property
    def temperature(self) -> float:
        """Air temperature in °C."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the air temperature in °C."""
        self._temperature = value

    @property
    def humidity(self) -> float:
        """Relative humidity in %."""
        return self._humidity

    @humidity.setter
    def humidity(self, value: float) -> None:
        """Set the relative humidity in %."""
        self._humidity = value

    @property
    def precip_intensity(self) -> float:
        """Precipitation intensity in mm/h."""
        return self._precip_intensity

    @precip_intensity.setter
    def precip_intensity(self, value: float) -> None:
        """Set the precipitation intensity in mm/h."""
        self._precip_intensity = value

    @property
    def wind_speed(self) -> float:
        """Wind speed in m/s."""
        return self._wind_speed

    @wind_speed.setter
    def wind_speed(self, value: float) -> None:
        """Set the wind speed in m/s."""
        self._wind_speed = value

    @property
    def weather_code(self) -> int:
        """Open-Meteo WMO weather code."""
        return self._weather_code

    @weather_code.setter
    def weather_code(self, value: int) -> None:
        """Set the WMO weather code."""
        self._weather_code = value

    @property
    def description(self) -> str:
        """Returns the text description of the weather code."""
        return WEATHER_MAP.get(self._weather_code, "Unknown")

    @property
    def has_forecast_probability(self) -> bool:
        """True when a real forecast probability was provided (not the unknown sentinel)."""
        return self.forecast_probability != self.FORECAST_PROBABILITY_UNKNOWN
