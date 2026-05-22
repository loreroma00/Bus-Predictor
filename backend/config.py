"""Centralized backend configuration.

The backend reads configuration once into ``Config`` and passes that object
through ``ApplicationContext``.  Runtime code should depend on the public
attributes on ``Config`` instead of parsing ``config.ini`` directly.
"""

from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

CONFIG_PATH = Path(__file__).parent / "config.ini"
_CONFIG_PATH = CONFIG_PATH

_ENV_OVERRIDES = {
    ("api", "tomtom_api_key"): "TOMTOM_API_KEY",
    ("ledger", "db_connection"): "TIMESCALE_CONNECTION_STRING",
}


def _read_section(
    data: Mapping[str, Mapping[str, Any]], section: str
) -> Mapping[str, Any]:
    return data.get(section, {})


def _get(
    data: Mapping[str, Mapping[str, Any]],
    section: str,
    key: str,
    default: str = "",
) -> str:
    env_var = _ENV_OVERRIDES.get((section, key))
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value

    value = _read_section(data, section).get(key, default)
    return str(value) if value is not None else default


def _get_int(
    data: Mapping[str, Mapping[str, Any]],
    section: str,
    key: str,
    default: int = 0,
) -> int:
    try:
        return int(_get(data, section, key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_bool(
    data: Mapping[str, Mapping[str, Any]],
    section: str,
    key: str,
    default: bool = False,
) -> bool:
    value = _get(data, section, key, str(default)).strip().lower()
    return value in {"1", "yes", "true", "on"}


@dataclass(frozen=True)
class API:
    """API keys and HTTP server settings."""

    tomtom_api_key: str = ""
    frontend_url: str = "http://localhost:3000"
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass(frozen=True)
class URLs:
    """External service URLs used by the backend."""

    gtfs_static: str = (
        "https://romamobilita.it/wp-content/uploads/drupal/rome_static_gtfs.zip"
    )
    gtfs_md5: str = (
        "https://romamobilita.it/wp-content/uploads/drupal/rome_static_gtfs.zip.md5"
    )
    rtgtfs_vehicles: str = (
        "https://romamobilita.it/wp-content/uploads/drupal/"
        "rome_rtgtfs_vehicle_positions_feed.pb"
    )
    rtgtfs_trip_updates: str = (
        "https://romamobilita.it/wp-content/uploads/drupal/"
        "rome_rtgtfs_trip_updates_feed.pb"
    )
    weather_api: str = "https://api.open-meteo.com/v1/forecast"
    tomtom_traffic: str = "https://api.tomtom.com/traffic/map/4/tile/flow"
    nominatim_url: str = "http://localhost:8080"
    gtfs_referer: str = "https://romamobilita.it/sistemi-e-tecnologie/open-data/"
    map_tile_url: str = (
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
    )
    map_tile_attribution: str = '&copy; <a href="https://carto.com/">CARTO</a>'


@dataclass(frozen=True)
class Paths:
    """File and directory paths."""

    ledger_cache: str = "ledger_cache.pkl"
    city_cache: str = "city_cache.pkl"
    measurements_path: str = "measurements/"
    measurements_file: str = "measurements.parquet"
    static_bus_lanes: str = "static_bus_lanes_roma.pkl"
    gtfs_static_zip: str = "rome_static_gtfs.zip"


@dataclass(frozen=True)
class Timings:
    """Update intervals and timeouts in seconds."""

    update_interval: int = 900
    traffic_update_interval: int = 900
    saving_interval: int = 1800
    weather_update_interval: int = 3600
    live_trip_stale_ttl: int = 600


@dataclass(frozen=True)
class Services:
    """Service-level settings."""

    cache_strategy: str = "file"
    debug_gui_port: int = 8050
    debug_gui_host: str = "0.0.0.0"
    debug_gui_display_host: str = "localhost"
    debug_gui_display_scheme: str = "http"
    weather_strategy: str = "greedy"
    weather_subsets: int = 4
    nominatim_user_agent: str = "thesis_project_real_bus_tracker"


@dataclass(frozen=True)
class DataCleaning:
    """Parameters for data cleaning pipelines."""

    min_measurements_per_run: int = 7
    max_bus_speed_kmh: int = 120
    max_time_gap_seconds: int = 600
    served_ratio_lookback_minutes: int = 60
    lenient_pipeline: bool = False


@dataclass(frozen=True)
class Ledger:
    """Database-backed ledger table settings."""

    db_connection: str = "postgresql://user:password@localhost:1515/transit"
    historical_table: str = "historical_measurements"
    predicted_table: str = "predicted_arrivals"
    vehicle_table: str = "vehicle_trips"


@dataclass(frozen=True)
class Traffic:
    """Traffic service settings."""

    tomtom_zoom: int = 15
    qps_limit: int = 10
    tile_ttl_seconds: int = 900


@dataclass(frozen=True)
class Config:
    """Typed application configuration."""

    api: API = API()
    urls: URLs = URLs()
    paths: Paths = Paths()
    timings: Timings = Timings()
    services: Services = Services()
    data_cleaning: DataCleaning = DataCleaning()
    ledger: Ledger = Ledger()
    traffic: Traffic = Traffic()
    source_path: Path = CONFIG_PATH

    @classmethod
    def load(cls, config_path: Path | str = CONFIG_PATH) -> "Config":
        """Load configuration from an INI file plus supported env overrides."""
        path = Path(config_path)
        parser = ConfigParser()
        if path.exists():
            parser.read(path)
        data = {section: dict(parser.items(section)) for section in parser.sections()}
        return cls.from_mapping(data, source_path=path)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Mapping[str, Any]] | None,
        source_path: Path | str = CONFIG_PATH,
    ) -> "Config":
        """Build a config object from a section/key mapping."""
        data = data or {}
        return cls(
            api=API(
                tomtom_api_key=_get(data, "api", "tomtom_api_key", ""),
                frontend_url=_get(data, "api", "frontend_url", API.frontend_url),
                host=_get(data, "api", "host", API.host),
                port=_get_int(data, "api", "port", API.port),
            ),
            urls=URLs(
                gtfs_static=_get(data, "urls", "gtfs_static", URLs.gtfs_static),
                gtfs_md5=_get(data, "urls", "gtfs_md5", URLs.gtfs_md5),
                rtgtfs_vehicles=_get(
                    data,
                    "urls",
                    "rtgtfs_vehicles",
                    URLs.rtgtfs_vehicles,
                ),
                rtgtfs_trip_updates=_get(
                    data,
                    "urls",
                    "rtgtfs_trip_updates",
                    URLs.rtgtfs_trip_updates,
                ),
                weather_api=_get(data, "urls", "weather_api", URLs.weather_api),
                tomtom_traffic=_get(
                    data,
                    "urls",
                    "tomtom_traffic",
                    URLs.tomtom_traffic,
                ),
                nominatim_url=_get(
                    data,
                    "urls",
                    "nominatim_url",
                    URLs.nominatim_url,
                ),
                gtfs_referer=_get(
                    data,
                    "urls",
                    "gtfs_referer",
                    URLs.gtfs_referer,
                ),
                map_tile_url=_get(
                    data,
                    "urls",
                    "map_tile_url",
                    URLs.map_tile_url,
                ),
                map_tile_attribution=_get(
                    data,
                    "urls",
                    "map_tile_attribution",
                    URLs.map_tile_attribution,
                ),
            ),
            paths=Paths(
                ledger_cache=_get(data, "paths", "ledger_cache", Paths.ledger_cache),
                city_cache=_get(data, "paths", "city_cache", Paths.city_cache),
                measurements_path=_get(
                    data,
                    "paths",
                    "measurements_path",
                    Paths.measurements_path,
                ),
                measurements_file=_get(
                    data,
                    "paths",
                    "measurements_file",
                    Paths.measurements_file,
                ),
                static_bus_lanes=_get(
                    data,
                    "paths",
                    "static_bus_lanes",
                    Paths.static_bus_lanes,
                ),
                gtfs_static_zip=_get(
                    data,
                    "paths",
                    "gtfs_static_zip",
                    Paths.gtfs_static_zip,
                ),
            ),
            timings=Timings(
                update_interval=_get_int(
                    data,
                    "timings",
                    "update_interval",
                    Timings.update_interval,
                ),
                traffic_update_interval=_get_int(
                    data,
                    "timings",
                    "traffic_update_interval",
                    Timings.traffic_update_interval,
                ),
                saving_interval=_get_int(
                    data,
                    "timings",
                    "saving_interval",
                    Timings.saving_interval,
                ),
                weather_update_interval=_get_int(
                    data,
                    "timings",
                    "weather_update_interval",
                    Timings.weather_update_interval,
                ),
                live_trip_stale_ttl=_get_int(
                    data,
                    "timings",
                    "live_trip_stale_ttl",
                    Timings.live_trip_stale_ttl,
                ),
            ),
            services=Services(
                cache_strategy=_get(
                    data,
                    "services",
                    "cache_strategy",
                    Services.cache_strategy,
                ),
                debug_gui_port=_get_int(
                    data,
                    "services",
                    "debug_gui_port",
                    Services.debug_gui_port,
                ),
                debug_gui_host=_get(
                    data,
                    "services",
                    "debug_gui_host",
                    Services.debug_gui_host,
                ),
                debug_gui_display_host=_get(
                    data,
                    "services",
                    "debug_gui_display_host",
                    Services.debug_gui_display_host,
                ),
                debug_gui_display_scheme=_get(
                    data,
                    "services",
                    "debug_gui_display_scheme",
                    Services.debug_gui_display_scheme,
                ),
                weather_strategy=_get(
                    data,
                    "services",
                    "weather_strategy",
                    Services.weather_strategy,
                ),
                weather_subsets=_get_int(
                    data,
                    "services",
                    "weather_subsets",
                    Services.weather_subsets,
                ),
                nominatim_user_agent=_get(
                    data,
                    "services",
                    "nominatim_user_agent",
                    Services.nominatim_user_agent,
                ),
            ),
            data_cleaning=DataCleaning(
                min_measurements_per_run=_get_int(
                    data,
                    "data_cleaning",
                    "min_measurements_per_run",
                    DataCleaning.min_measurements_per_run,
                ),
                max_bus_speed_kmh=_get_int(
                    data,
                    "data_cleaning",
                    "max_bus_speed_kmh",
                    DataCleaning.max_bus_speed_kmh,
                ),
                max_time_gap_seconds=_get_int(
                    data,
                    "data_cleaning",
                    "max_time_gap_seconds",
                    DataCleaning.max_time_gap_seconds,
                ),
                served_ratio_lookback_minutes=_get_int(
                    data,
                    "data_cleaning",
                    "served_ratio_lookback_minutes",
                    DataCleaning.served_ratio_lookback_minutes,
                ),
                lenient_pipeline=_get_bool(
                    data,
                    "data_cleaning",
                    "lenient_pipeline",
                    DataCleaning.lenient_pipeline,
                ),
            ),
            ledger=Ledger(
                db_connection=_get(
                    data,
                    "ledger",
                    "db_connection",
                    Ledger.db_connection,
                ),
                historical_table=_get(
                    data,
                    "ledger",
                    "historical_table",
                    Ledger.historical_table,
                ),
                predicted_table=_get(
                    data,
                    "ledger",
                    "predicted_table",
                    Ledger.predicted_table,
                ),
                vehicle_table=_get(
                    data,
                    "ledger",
                    "vehicle_table",
                    Ledger.vehicle_table,
                ),
            ),
            traffic=Traffic(
                tomtom_zoom=_get_int(
                    data,
                    "traffic",
                    "tomtom_zoom",
                    Traffic.tomtom_zoom,
                ),
                qps_limit=_get_int(
                    data,
                    "traffic",
                    "qps_limit",
                    Traffic.qps_limit,
                ),
                tile_ttl_seconds=_get_int(
                    data,
                    "traffic",
                    "tile_ttl_seconds",
                    Traffic.tile_ttl_seconds,
                ),
            ),
            source_path=Path(source_path),
        )

    @classmethod
    def coerce(cls, value: "Config | Mapping[str, Mapping[str, Any]] | None") -> "Config":
        """Return a ``Config`` for existing ``Config``/dict/None inputs."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.load()
        return cls.from_mapping(value)

    def with_lenient_pipeline(self, enabled: bool = True) -> "Config":
        """Return a copy with the lenient data-cleaning flag set."""
        return replace(
            self,
            data_cleaning=replace(self.data_cleaning, lenient_pipeline=enabled),
        )


def load_config(config_path: Path | str = CONFIG_PATH) -> Config:
    """Compatibility wrapper returning the typed backend ``Config``."""
    return Config.load(config_path)
