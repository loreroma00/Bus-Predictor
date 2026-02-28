"""
Weather fetching from Open-Meteo API.
"""

import requests

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

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


def get_weather_code(latitude: float = 41.9, longitude: float = 12.5) -> int:
    """
    Fetch current weather code from Open-Meteo API.
    Defaults to Rome coordinates.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "weather_code",
        "timezone": "auto",
    }
    response = requests.get(WEATHER_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data["current"]["weather_code"]


def get_weather_description(code: int) -> str:
    """Get human-readable description for a weather code."""
    return WEATHER_MAP.get(code, "Unknown")
