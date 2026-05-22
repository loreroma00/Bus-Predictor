# Persistence Package
# Re-exports from new modular structure

from .strategy import (
    TOPOLOGY_CACHE_FILE,
    SCHEDULE_CACHE_FILE,
    CITY_CACHE_FILE,
    BaseCacheStrategy,
    FileCacheStrategy,
    NoCacheStrategy,
    get_cache_strategy,
    get_available_strategies,
    shutdown_db_loop,
)

from .measurements import (
    MEASUREMENTS_PATH,
    MEASUREMENTS_FILE,
    save_measurements_incremental,
    saving_loop,
)

from .cache import log_uptime, save_pickle, load_pickle
from .gateway import PersistenceFacade, create_persistence_gateway

from .database import (
    write_historical,
    write_predicted,
    write_vehicle_trips,
    read_historical,
    read_predicted,
    read_vehicle_trips,
    close_ledger_writers,
)

from .database import (
    TimescaleDBConnection,
    get_db_connection,
    init_database,
    close_database,
    shutdown_database,
    test_database_connection,
)

__all__ = [
    # Strategy module
    "TOPOLOGY_CACHE_FILE",
    "SCHEDULE_CACHE_FILE",
    "CITY_CACHE_FILE",
    "BaseCacheStrategy",
    "FileCacheStrategy",
    "NoCacheStrategy",
    "get_cache_strategy",
    "get_available_strategies",
    "shutdown_db_loop",
    # Measurements module
    "MEASUREMENTS_PATH",
    "MEASUREMENTS_FILE",
    "save_measurements_incremental",
    "saving_loop",
    # Cache module
    "log_uptime",
    "save_pickle",
    "load_pickle",
    "PersistenceFacade",
    "create_persistence_gateway",
    # Ledger DB module
    "write_historical",
    "write_predicted",
    "write_vehicle_trips",
    "read_historical",
    "read_predicted",
    "read_vehicle_trips",
    "close_ledger_writers",
    # Database module
    "TimescaleDBConnection",
    "get_db_connection",
    "init_database",
    "close_database",
    "shutdown_database",
    "test_database_connection",
]
