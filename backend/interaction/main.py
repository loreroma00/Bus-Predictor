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
        # Load Configuration
        from config import load_config
        config = load_config()

        # Check config file (data_cleaning section)
        file_setting = config.get("data_cleaning", {}).get("lenient_pipeline", "false").lower() == "true"
        
        # Override config with CLI args or file setting
        if lenient_pipeline or file_setting:
            config["lenient_pipeline"] = True
            logging.info("🔧 Lenient Pipeline ENABLED.")

        # 1. Create and wire dependencies
        logging.info(f"Available cache strategies: {get_available_strategies()}")
        
        # Get cache strategy from config
        strategy_name = config.get("services", {}).get("cache_strategy", "file")
        logging.info(f"Using cache strategy: '{strategy_name}'")
        cache_strategy = get_cache_strategy(strategy_name)

        logging.info("Creating Observatory...")
        observatory = domain.Observatory(cache_strategy=cache_strategy, config=config)

        # 2. Initialize data module with the observatory and cache strategy
        data.initialize(observatory, cache_strategy=cache_strategy, config=config)

        # 2a. Initialize Database singleton with config
        from persistence import get_db_connection
        get_db_connection(config)

        # 1b. Load static bus lane map
        static_bus_lanes = {}
        try:
            loader = FileCacheStrategy(cache_path="static_bus_lanes_roma.pkl")
            static_bus_lanes = loader.load() or {}
            if static_bus_lanes:
                logging.info(f"Loaded static bus lanes map ({len(static_bus_lanes)} hexes)")
        except Exception as e:
            logging.warning(f"Could not load static bus lanes: {e}")

        observatory.add_city("Rome", static_bus_lanes=static_bus_lanes)

        # 2b. Load city cache (street names) if available
        if hasattr(cache_strategy, "load_city_cache"):
            cache_strategy.load_city_cache(observatory.get_city("Rome"))

        # 2c. Create geocoding service (auto-selects sync/async based on server)
        geocoding_service = create_geocoding_service(observatory.get_city("Rome"))
        observatory._geocoding = geocoding_service

        # 2d. Create traffic service if API key is available
        from application.live.traffic_service import create_traffic_service

        tomtom_api_key = config.get("api", {}).get("tomtom_api_key")
        if tomtom_api_key:
            traffic_service = create_traffic_service(
                observatory.get_city("Rome"),
                api_key=tomtom_api_key,
                zoom=15,  # Street-level detail for better bus route coverage
            )
            data.TRAFFIC_SERVICE = traffic_service
            data.wire_traffic_callback()  # Wire callback for expired traffic handling
            logging.info("🚗 Traffic service initialized.")
        else:
            logging.warning("⚠️ TOMTOM_API_KEY not set, traffic updates disabled.")

        # 3. Load static data
        logging.info("Loading ledger...")
        observatory._ensure_static_data_loaded()

        # 4. Register commands with dependencies
        console.register_commands(observatory)

        # 5. Create state interface for Debug GUI
        from .state_interface import StateInterface

        state_interface = StateInterface(observatory)
        services.set_state_interface(state_interface)

        # 6. Start Services (including Debug GUI)
        services.start_services(observatory, config)

        # 6. Start Interactive Console (Main Thread)
        console.run_console_loop()

        # 7. Cleanup
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
