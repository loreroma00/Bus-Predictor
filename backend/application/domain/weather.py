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
    def __init__(
        self,
        valid_time: float,
        temperature: float,
        apparent_temperature: float,
        humidity: float,
        precip_intensity: float,
        wind_speed: float,
        weather_code: int,
    ):
        self.time = valid_time if valid_time else time.time()
        self.temperature = temperature  # in °C
        self.apparent_temperature = apparent_temperature  # in °C
        self.humidity = humidity  # in %
        self.precip_intensity = precip_intensity  # in mm/h
        self.wind_speed = wind_speed  # in m/s
        self.weather_code = int(weather_code) if weather_code is not None else 0

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value

    @property
    def humidity(self) -> float:
        return self._humidity

    @humidity.setter
    def humidity(self, value: float) -> None:
        self._humidity = value

    @property
    def precip_intensity(self) -> float:
        return self._precip_intensity

    @precip_intensity.setter
    def precip_intensity(self, value: float) -> None:
        self._precip_intensity = value

    @property
    def wind_speed(self) -> float:
        return self._wind_speed

    @wind_speed.setter
    def wind_speed(self, value: float) -> None:
        self._wind_speed = value

    @property
    def weather_code(self) -> int:
        return self._weather_code

    @weather_code.setter
    def weather_code(self, value: int) -> None:
        self._weather_code = value

    @property
    def description(self) -> str:
        """Returns the text description of the weather code."""
        return WEATHER_MAP.get(self._weather_code, "Unknown")
