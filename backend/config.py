"""
Configuration Module - Centralized access to all configuration values.

Loads from config.ini, with environment variable overrides.
Environment variables take precedence over config file values.
"""

import os
from configparser import ConfigParser
from pathlib import Path

# Find config file (project root)
_CONFIG_PATH = Path(__file__).parent / "config.ini"
_config = ConfigParser()

if _CONFIG_PATH.exists():
    _config.read(_CONFIG_PATH)
else:
    print(f"⚠️ Config file not found at {_CONFIG_PATH}")


def load_config(config_path: Path = _CONFIG_PATH) -> dict:
    """
    Load configuration from INI file and environment variables.

    Returns:
        dict: Dictionary containing configuration sections and values.
    """
    config_parser = ConfigParser()
    if config_path.exists():
        config_parser.read(config_path)

    # Base configuration dictionary
    config = {section: dict(config_parser.items(section)) for section in config_parser.sections()}

    # Environment variable mappings (Section, Key, EnvVar)
    env_overrides = [
        ("api", "tomtom_api_key", "TOMTOM_API_KEY"),
        ("database", "timescale_connection_string", "TIMESCALE_CONNECTION_STRING"),
    ]

    # Apply environment overrides
    for section, key, env_var in env_overrides:
        env_val = os.environ.get(env_var)
        # Override only if env_val is not None and not empty string
        if env_val:
            if section not in config:
                config[section] = {}
            config[section][key] = env_val

    # Ensure critical sections exist to avoid KeyErrors downstream
    required_sections = ["api", "database", "urls", "paths", "timings", "services", "data_cleaning", "prediction", "traffic"]
    for section in required_sections:
        if section not in config:
            config[section] = {}

    return config


def _get(section: str, key: str, default: str = None, env_var: str = None) -> str:
    """
    Get config value with environment variable override.

    Priority: Environment Variable > Config File > Default
    """
    # Check environment variable first
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value

    # Check config file
    try:
        return _config.get(section, key)
    except Exception:
        return default


def _get_int(section: str, key: str, default: int = 0, env_var: str = None) -> int:
    """Get config value as integer."""
    value = _get(section, key, str(default), env_var)
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_float(
    section: str, key: str, default: float = 0.0, env_var: str = None
) -> float:
    """Get config value as float."""
    value = _get(section, key, str(default), env_var)
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ============================================================
# API Configuration
# ============================================================
class API:
    """API keys and endpoints."""

    TOMTOM_API_KEY = _get("api", "tomtom_api_key", "", "TOMTOM_API_KEY")


# ============================================================
# URL Configuration
# ============================================================
class URLs:
    """External service URLs."""

    # GTFS Static
    GTFS_STATIC = _get(
        "urls",
        "gtfs_static",
        "https://romamobilita.it/wp-content/uploads/drupal/rome_static_gtfs.zip",
    )
    GTFS_MD5 = _get(
        "urls",
        "gtfs_md5",
        "https://romamobilita.it/wp-content/uploads/drupal/rome_static_gtfs.zip.md5",
    )

    # GTFS Realtime
    RTGTFS_VEHICLES = _get(
        "urls",
        "rtgtfs_vehicles",
        "https://romamobilita.it/wp-content/uploads/drupal/rome_rtgtfs_vehicle_positions_feed.pb",
    )
    RTGTFS_TRIP_UPDATES = _get(
        "urls",
        "rtgtfs_trip_updates",
        "https://romamobilita.it/wp-content/uploads/drupal/rome_rtgtfs_trip_updates_feed.pb",
    )

    # Weather
    WEATHER_API = _get("urls", "weather_api", "https://api.open-meteo.com/v1/forecast")

    # Traffic
    TOMTOM_TRAFFIC = _get(
        "urls", "tomtom_traffic", "https://api.tomtom.com/traffic/map/4/tile/flow"
    )

    # Geocoding
    NOMINATIM_URL = _get("urls", "nominatim_url", "http://localhost:8080")


# ============================================================
# Path Configuration
# ============================================================
class Paths:
    """File and directory paths."""

    LEDGER_CACHE = _get("paths", "ledger_cache", "ledger_cache.pkl")
    CITY_CACHE = _get("paths", "city_cache", "city_cache.pkl")
    DIARIES_PATH = _get("paths", "diaries_path", "diaries/")
    DIARIES_FILE = _get("paths", "diaries_file", "diaries.parquet")
    STATIC_BUS_LANES = _get("paths", "static_bus_lanes", "static_bus_lanes_roma.pkl")


# ============================================================
# Timing Configuration
# ============================================================
class Timings:
    """Update intervals and timeouts (in seconds)."""

    UPDATE_INTERVAL = _get_int("timings", "update_interval", 900)
    TRAFFIC_UPDATE_INTERVAL = _get_int("timings", "traffic_update_interval", 900)
    SAVING_INTERVAL = _get_int("timings", "saving_interval", 1800)
    WEATHER_UPDATE_INTERVAL = _get_int("timings", "weather_update_interval", 3600)


# ============================================================
# Service Configuration
# ============================================================
class Services:
    """Service-level settings."""

    CACHE_STRATEGY = _get("services", "cache_strategy", "file")
    DEBUG_GUI_PORT = _get_int("services", "debug_gui_port", 8050)


# ============================================================
# Data Cleaning Configuration
# ============================================================
class DataCleaning:
    """Parameters for data cleaning pipeline."""

    MIN_MEASUREMENTS_PER_RUN = _get_int("data_cleaning", "min_measurements_per_run", 7)
    MAX_BUS_SPEED_KMH = _get_int("data_cleaning", "max_bus_speed_kmh", 120)
    MAX_TIME_GAP_SECONDS = _get_int("data_cleaning", "max_time_gap_seconds", 600)


# ============================================================
# Prediction Configuration
# ============================================================
class Prediction:
    """Prediction pipeline settings."""

    ENABLED = _get("prediction", "enabled", "true").lower() == "true"
    VECTOR_DB_CONNECTION = _get(
        "prediction",
        "vector_db_connection",
        "postgresql://user:password@localhost:1515/transit",
    )
    VECTOR_TABLE = _get("prediction", "vector_table", "prediction_vectors")
    LABEL_TABLE = _get("prediction", "label_table", "prediction_label")


# ============================================================
# Traffic Configuration
# ============================================================
class Traffic:
    """Traffic pipeline settings."""

    ENABLED = _get("traffic", "enabled", "true").lower() == "true"
    VECTOR_DB_CONNECTION = _get(
        "traffic",
        "vector_db_connection",
        "postgresql://user:password@localhost:1515/transit",
    )
    VECTOR_TABLE = _get("traffic", "vector_table", "traffic_vectors")
    LABEL_TABLE = _get("traffic", "label_table", "traffic_label")


# ============================================================
# Vehicle Configuration
# ============================================================
class Vehicle:
    """Vehicle pipeline settings."""

    ENABLED = _get("vehicle", "enabled", "true").lower() == "true"
    VECTOR_DB_CONNECTION = _get(
        "vehicle",
        "vector_db_connection",
        "postgresql://user:password@localhost:1515/transit",
    )
    VECTOR_TABLE = _get("vehicle", "vector_table", "vehicle_vectors")
    LABEL_TABLE = _get("vehicle", "label_table", "vehicle_label")


# ============================================================
# Database Configuration (Legacy Alias)
# ============================================================
class Database:
    """Legacy Database settings aliased to Prediction."""

    TIMESCALE_CONNECTION_STRING = Prediction.VECTOR_DB_CONNECTION
    TIMESCALE_TABLE = Prediction.VECTOR_TABLE


# ============================================================
# Convenience Exports
# ============================================================
# For backwards compatibility and easy access
TOMTOM_API_KEY = API.TOMTOM_API_KEY
TIMESCALE_CONNECTION_STRING = Database.TIMESCALE_CONNECTION_STRING
