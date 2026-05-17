"""Backend bootstrap helpers.

The goal of this module is to keep entry points thin while avoiding a swarm of
tiny lifecycle files.  It builds the runtime context, wires the console/GUI
state interface, starts/stops collection services, and performs final cleanup.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any

from application import domain
from application.domain.map_info import create_geocoding_service
from application.runtime import ApplicationContext
from application.services.persistence_gateway import (
    configure_persistence_gateway,
    get_persistence_gateway,
)
from persistence import (
    get_available_strategies,
    get_cache_strategy,
    load_pickle,
)
from persistence.gateway import create_persistence_gateway


PROJECT_ROOT = Path(__file__).resolve().parent
PARQUET_DIR = PROJECT_ROOT / "parquets"


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
    persistence_gateway = configure_persistence_gateway(create_persistence_gateway())

    logging.info("Creating Observatory...")
    observatory = domain.Observatory(
        cache_strategy=cache_strategy,
        config=config,
        persistence_gateway=persistence_gateway,
    )

    persistence_gateway.ensure_database(config)

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
        persistence_gateway=persistence_gateway,
        geocoding_service=geocoding_service,
        traffic_service=traffic_service,
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

    state_interface = StateInterface(
        context.observatory,
        predictor=predictor,
        context=context,
    )
    if predictor is not None and loaded_model_name:
        state_interface.set_predictor_info(loaded_model_name)
    state_interface.set_command_registry(console._command_registry)
    services.set_state_interface(state_interface)

    context.state_interface = state_interface
    return state_interface


def build_serving_runtime(
    time_model_name: str | None = None,
    crowd_model_name: str | None = None,
    lenient_pipeline: bool = False,
):
    """Build the full runtime used by ``main.py serve``."""
    print("\n[1/5] Initializing collection pipeline (Observatory, City, services)...")
    context = build_runtime_context(lenient_pipeline=lenient_pipeline)

    print("\n[2/5] Generating canonical route map (if needed)...")
    _ensure_canonical_map()

    print("\n[3/5] Loading ML model...")
    from application.model.model_loader import ModelLoader
    from application.model.predictor import Predictor
    from api_runtime import APIRuntime

    model_loader = ModelLoader()
    loaded_model = _select_loaded_model(
        model_loader,
        time_model_name=time_model_name,
        crowd_model_name=crowd_model_name,
    )
    predictor = Predictor(
        loaded_model=loaded_model,
        observatory=context.observatory,
        persistence_gateway=context.persistence_gateway,
    )

    runtime = APIRuntime(
        context=context,
        predictor=predictor,
        available_models=_model_cards(model_loader.discover_models()),
        loaded_model_name=loaded_model.name,
    )

    print("\n[4/5] Starting data collection services...")
    wire_state_interface(
        context,
        predictor=predictor,
        bus_type_predictor=runtime.bus_type_predictor,
        loaded_model_name=loaded_model.name,
    )
    start_collection_services(context)
    return runtime


def _ensure_canonical_map():
    """Build the canonical static map required by the predictor when absent."""
    if (PARQUET_DIR / "stop_route_map.parquet").exists():
        return

    print("Generating stop_route_map.parquet...")
    from prepare_dataset import build_canonical_shape_map

    build_canonical_shape_map()


def _model_cards(candidates) -> list[dict[str, str]]:
    """Return API-facing model cards without exposing ModelLoader types."""
    return [
        {
            "filename": candidate.name,
            "path": str(candidate.time_path),
        }
        for candidate in candidates
    ]


def _select_loaded_model(
    loader,
    time_model_name: str | None,
    crowd_model_name: str | None,
):
    """Select and load one complete TIME+CROWD model pair."""
    available = loader.discover_models()
    for time_filename, missing in loader.find_incomplete_models():
        print(f"[WARN] Skipping {time_filename}: missing {', '.join(missing)}")

    if not available:
        raise RuntimeError(
            "No trained model pairs found in application/model/. Expected "
            "bus_model_TIME_*.pth + bus_model_CROWD_*.pth + hyperparameters_DUAL_*.json"
        )

    cli_time = time_model_name or os.environ.get("TIME_MODEL_NAME")
    cli_crowd = crowd_model_name or os.environ.get("CROWD_MODEL_NAME")
    if cli_time and cli_crowd:
        return loader.load_pair(cli_time, cli_crowd)
    if cli_time:
        exp_id = cli_time.replace("bus_model_TIME_", "").replace(".pth", "")
        return loader.load_by_exp_id(exp_id)

    return _interactive_model_selection(loader, available)


def _interactive_model_selection(loader, available_models):
    """Prompt for a model pair when the CLI did not specify one."""
    print("\n" + "=" * 50)
    print("ATAC Bus Delay Prediction - Model Selection")
    print("=" * 50)
    print("\nAvailable model pairs:")
    for idx, model in enumerate(available_models):
        print(f"  [{idx}] {model.name}")
        print(f"       TIME:  {model.time_filename}")
        print(f"       CROWD: {model.crowd_filename}")

    while True:
        try:
            choice_idx = int(input("\nSelect model number: ").strip())
            if 0 <= choice_idx < len(available_models):
                break
            print(f"Please enter a number between 0 and {len(available_models) - 1}")
        except (ValueError, EOFError):
            print("Invalid input. Please enter a number.")

    return loader.load_by_exp_id(available_models[choice_idx].name)


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


def save_completed_measurements(observatory, persistence_gateway=None):
    """Final completed-measurement save on exit."""
    if observatory is None:
        return
    measurements = observatory.get_completed_measurements()
    if measurements:
        gateway = persistence_gateway or get_persistence_gateway()
        gateway.save_completed_measurements(measurements)
    logging.info("Goodbye.")


def shutdown_runtime(
    context: ApplicationContext | None = None,
    save_measurements: bool = True,
    join_services: bool = False,
):
    """Stop services, save completed measurements and close database resources."""
    stop_collection_services(join=join_services)
    gateway = (
        context.persistence_gateway
        if context is not None and context.persistence_gateway is not None
        else get_persistence_gateway()
    )

    if save_measurements:
        observatory = context.observatory if context is not None else None
        save_completed_measurements(observatory, persistence_gateway=gateway)

    logging.info("Closing database connections...")
    gateway.shutdown_database()
