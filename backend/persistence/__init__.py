# Persistence Package
# Re-exports from new modular structure

from .strategy import (
    CACHE_FILE,
    CITY_CACHE_FILE,
    BaseCacheStrategy,
    FileCacheStrategy,
    NoCacheStrategy,
    get_cache_strategy,
    get_available_strategies,
    saving_strategy,
    saving_parquet,
    saving_database,
    shutdown_db_loop,
)

from .diaries import (
    DIARIES_PATH,
    DIARIES_FILE,
    val,
    readParquet,
    writeParquet,
    updateParquet,
    get_latest_diary_index,
    save_diaries,
    save_diaries_incremental,
    saving_loop,
)

from .cache import log_uptime

from .database import (
    TimescaleDBConnection,
    get_db_connection,
    init_database,
    close_database,
    shutdown_database,
)

__all__ = [
    # Strategy module
    "CACHE_FILE",
    "CITY_CACHE_FILE",
    "BaseCacheStrategy",
    "FileCacheStrategy",
    "NoCacheStrategy",
    "get_cache_strategy",
    "get_available_strategies",
    "saving_strategy",
    "saving_parquet",
    "saving_database",
    "shutdown_db_loop",
    # Diaries module
    "DIARIES_PATH",
    "DIARIES_FILE",
    "val",
    "readParquet",
    "writeParquet",
    "updateParquet",
    "get_latest_diary_index",
    "save_diaries",
    "save_diaries_incremental",
    "saving_loop",
    # Cache module
    "log_uptime",
    # Database module
    "TimescaleDBConnection",
    "get_db_connection",
    "init_database",
    "close_database",
    "shutdown_database",
]
