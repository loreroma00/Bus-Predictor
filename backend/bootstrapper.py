"""Backend bootstrap helpers.

The goal of this module is to keep entry points thin while avoiding a swarm of
tiny lifecycle files.  It builds the runtime context, wires the console/GUI
state interface, starts/stops collection services, and performs final cleanup.
"""

import logging
import threading
from typing import Any

from application import domain
from application.domain.map_info import create_geocoding_service
from application.live import data
from application.runtime import ApplicationContext
from persistence import (
    get_available_strategies,
    get_cache_strategy,
    get_db_connection,
    load_pickle,
    save_diaries_incremental,
)


def configure_logging(debug_mode: bool = False, log_file: str = "backend.log"):
    """Configure console and file logging for interactive backend modes."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if debug_mode else logging.ERROR)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)


def build_runtime_context(
    config: dict[str, Any] | None = None,
    lenient_pipeline: bool = False,
) -> ApplicationContext:
    """Initialize Observatory, City, cache strategy, DB singleton and live services."""
    from config import load_config

    if config is None:
        config = load_config()

    file_setting = (
        config.get("data_cleaning", {})
        .get("lenient_pipeline", "false")
        .lower()
        == "true"
    )
    if lenient_pipeline or file_setting:
        config["lenient_pipeline"] = True
        logging.info("Lenient Pipeline ENABLED.")

    logging.info("Available cache strategies: %s", get_available_strategies())
    strategy_name = config.get("services", {}).get("cache_strategy", "file")
    logging.info("Using cache strategy: '%s'", strategy_name)
    cache_strategy = get_cache_strategy(strategy_name)

    logging.info("Creating Observatory...")
    observatory = domain.Observatory(cache_strategy=cache_strategy, config=config)

    get_db_connection(config)

    static_bus_lanes = load_pickle("static_bus_lanes_roma.pkl", default={})
    if static_bus_lanes:
        logging.info("Loaded static bus lanes map (%s hexes)", len(static_bus_lanes))

    observatory.add_city("Rome", static_bus_lanes=static_bus_lanes)
    city = observatory.get_city("Rome")

    from application.domain.weather_strategy import (
        get_available_weather_strategies,
        get_weather_strategy,
    )

    services_cfg = config.get("services", {})
    weather_strategy_name = services_cfg.get("weather_strategy", "greedy")
    weather_subsets = int(services_cfg.get("weather_subsets", "4"))
    city.weather_strategy = get_weather_strategy(
        weather_strategy_name,
        n_subsets=weather_subsets,
    )
    logging.info(
        "Weather strategy: '%s' (available: %s)",
        weather_strategy_name,
        get_available_weather_strategies(),
    )

    if hasattr(cache_strategy, "load_city_cache"):
        cache_strategy.load_city_cache(city)

    geocoding_service = create_geocoding_service(city)
    observatory._geocoding = geocoding_service

    traffic_service = None
    from application.live.traffic_service import create_traffic_service

    tomtom_api_key = config.get("api", {}).get("tomtom_api_key")
    if tomtom_api_key:
        traffic_service = create_traffic_service(city, api_key=tomtom_api_key, zoom=15)
        city.set_traffic_service(traffic_service)
        logging.info("Traffic service initialized.")
    else:
        logging.warning("TOMTOM_API_KEY not set, traffic updates disabled.")

    logging.info("Loading ledger...")
    observatory._ensure_static_data_loaded()

    from application.live.feed_fetcher import LiveFeedFetcher

    urls = config.get("urls", {})
    feed_fetcher = LiveFeedFetcher(
        vehicles_url=urls.get("rtgtfs_vehicles"),
        trips_url=urls.get("rtgtfs_trip_updates"),
    )
    stop_event = threading.Event()
    shutdown_event = threading.Event()

    context = ApplicationContext(
        config=config,
        observatory=observatory,
        city=city,
        cache_strategy=cache_strategy,
        geocoding_service=geocoding_service,
        traffic_service=traffic_service,
        feed_fetcher=feed_fetcher,
        stop_event=stop_event,
        shutdown_event=shutdown_event,
    )
    data.initialize(
        observatory,
        cache_strategy=cache_strategy,
        config=config,
        feed_fetcher=feed_fetcher,
        stop_event=stop_event,
        shutdown_event=shutdown_event,
    )
    return context


def wire_state_interface(
    context: ApplicationContext,
    predictor: Any = None,
    bus_type_predictor: Any = None,
    loaded_model_name: str | None = None,
):
    """Create the GUI state interface and register console commands."""
    from interaction import console, services
    from interaction.state_interface import StateInterface

    context.predictor = predictor
    context.bus_type_predictor = bus_type_predictor

    console.register_commands(
        context.observatory,
        predictor=predictor,
        bus_type_predictor=bus_type_predictor,
    )

    state_interface = StateInterface(context.observatory, predictor=predictor)
    if predictor is not None and loaded_model_name:
        state_interface.set_predictor_info(loaded_model_name)
    state_interface.set_command_registry(console._command_registry)
    services.set_state_interface(state_interface)

    context.state_interface = state_interface
    return state_interface


def start_collection_services(context: ApplicationContext):
    """Start collection, persistence, weather, traffic and debug GUI services."""
    from interaction import services

    services.start_services(context=context)


def stop_collection_services(join: bool = False):
    """Stop collection services, optionally waiting for core threads to finish."""
    from interaction import services

    services.stop_services()
    if join:
        services.join_core_threads(timeout=5)


def save_completed_diaries(observatory):
    """Final diary save on exit."""
    if observatory is None:
        return
    diaries = observatory.get_completed_diaries()
    if diaries:
        save_diaries_incremental(diaries)
    logging.info("Goodbye.")


def shutdown_runtime(
    context: ApplicationContext | None = None,
    save_diaries: bool = True,
    join_services: bool = False,
):
    """Stop services, save diaries and close database resources."""
    from persistence.database import shutdown_database

    stop_collection_services(join=join_services)

    if save_diaries:
        observatory = context.observatory if context is not None else data.OBSERVATORY
        save_completed_diaries(observatory)

    logging.info("Closing database connections...")
    shutdown_database()
