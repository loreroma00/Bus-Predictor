"""
Main - Application entry point and dependency wiring.

This is where all the pieces come together:
- Choose cache strategy (discovered by name)
- Initialize the Observatory with chosen strategy
- Register commands with their dependencies
- Subscribe event handlers
- Start the application
"""

import logging
import os
from application.live import data
from application import domain
from application.domain.map_info import create_geocoding_service
from . import services
from . import console
from persistence import (
    get_cache_strategy,
    get_available_strategies,
    FileCacheStrategy,
    save_diaries_incremental,
)


# ============================================================
# Configuration
# ============================================================
CACHE_STRATEGY = "file"  # Options: "file", "none" (add more in persistence.py)


# ============================================================
# Main Application
# ============================================================


def initialize_collection(config=None, lenient_pipeline: bool = False):
    """
    Initialize the data collection pipeline: Observatory, City, services.

    Returns (observatory, city, config) for use by both the standalone
    ``collect`` command and the unified ``serve`` mode.
    """
    from config import load_config

    if config is None:
        config = load_config()

    # Check config file (data_cleaning section)
    file_setting = config.get("data_cleaning", {}).get("lenient_pipeline", "false").lower() == "true"
    if lenient_pipeline or file_setting:
        config["lenient_pipeline"] = True
        logging.info("Lenient Pipeline ENABLED.")

    # Cache strategy
    logging.info(f"Available cache strategies: {get_available_strategies()}")
    strategy_name = config.get("services", {}).get("cache_strategy", "file")
    logging.info(f"Using cache strategy: '{strategy_name}'")
    cache_strategy = get_cache_strategy(strategy_name)

    logging.info("Creating Observatory...")
    observatory = domain.Observatory(cache_strategy=cache_strategy, config=config)

    # Initialize data module
    data.initialize(observatory, cache_strategy=cache_strategy, config=config)

    # Database singleton
    from persistence import get_db_connection
    get_db_connection(config)

    # Static bus lane map
    static_bus_lanes = {}
    try:
        loader = FileCacheStrategy(cache_path="static_bus_lanes_roma.pkl")
        static_bus_lanes = loader.load() or {}
        if static_bus_lanes:
            logging.info(f"Loaded static bus lanes map ({len(static_bus_lanes)} hexes)")
    except Exception as e:
        logging.warning(f"Could not load static bus lanes: {e}")

    observatory.add_city("Rome", static_bus_lanes=static_bus_lanes)
    city = observatory.get_city("Rome")

    # City cache (street names)
    if hasattr(cache_strategy, "load_city_cache"):
        cache_strategy.load_city_cache(city)

    # Geocoding service
    geocoding_service = create_geocoding_service(city)
    observatory._geocoding = geocoding_service

    # Traffic service
    from application.live.traffic_service import create_traffic_service

    tomtom_api_key = config.get("api", {}).get("tomtom_api_key")
    if tomtom_api_key:
        traffic_service = create_traffic_service(
            city, api_key=tomtom_api_key, zoom=15,
        )
        data.TRAFFIC_SERVICE = traffic_service
        data.wire_traffic_callback()
        logging.info("Traffic service initialized.")
    else:
        logging.warning("TOMTOM_API_KEY not set, traffic updates disabled.")

    # Load static GTFS data
    logging.info("Loading ledger...")
    observatory._ensure_static_data_loaded()

    return observatory, city, config


def main(debug_mode: bool = False, lenient_pipeline: bool = False):
    # Set up logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter
    root_logger.handlers = []  # Clear existing handlers

    # 1. Console Handler - Clean output for the user (INFO+)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # 2. File Handler - Detailed log for debugging
    file_handler = logging.FileHandler("backend.log", encoding="utf-8")
    # Default: only ERRORS in file. Debug mode: EVERYTHING in file.
    file_level = logging.DEBUG if debug_mode else logging.ERROR
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)

    try:
        observatory, city, config = initialize_collection(
            lenient_pipeline=lenient_pipeline,
        )

        # Register commands with dependencies
        console.register_commands(observatory)

        # Create state interface for Debug GUI
        from .state_interface import StateInterface

        state_interface = StateInterface(observatory)
        services.set_state_interface(state_interface)

        # Start Services (including Debug GUI)
        services.start_services(observatory, config)

        # Start Interactive Console (Main Thread)
        console.run_console_loop()

        # Cleanup
        services.stop_services()

        if services.COLLECTION_THREAD:
            services.COLLECTION_THREAD.join(timeout=5)
        if services.SAVING_THREAD:
            services.SAVING_THREAD.join(timeout=5)

        logging.info("Archiving Diaries...")
        last_save(observatory)

    except KeyboardInterrupt:
        logging.info("\nStopping Observer (User Interrupt)...")
        services.stop_services()
        logging.info("Archiving Diaries...")
        last_save(data.OBSERVATORY)
    except Exception as e:
        logging.exception(f"Unexpected Error: {e}")
        logging.error("Attempting failsafe save...")
        if data.OBSERVATORY:
            last_save(data.OBSERVATORY)


def last_save(observatory):
    """Final diary save on exit - uses incremental save to single file."""
    diaries = observatory.get_completed_diaries()
    if diaries:
        save_diaries_incremental(diaries)
    logging.info("Goodbye.")


if __name__ == "__main__":
    main()
