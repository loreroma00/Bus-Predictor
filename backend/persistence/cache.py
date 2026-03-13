"""
Cache - Pickle-based caching for non-diary data (uptime logging, etc.).
"""

import logging
import os
import pickle
from datetime import datetime
import time


def save_pickle(data, path: str):
    """Save arbitrary data to a pickle file."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logging.error(f"Could not save pickle to {path}: {e}")


def load_pickle(path: str, default=None):
    """Load data from a pickle file. Returns *default* if missing or corrupt."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Could not load pickle from {path}: {e}")
        return default


def log_uptime(interval_seconds=60):
    """Appends a heartbeat to uptime.csv."""
    filename = "uptime.csv"
    timestamp = int(time.time())
    readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Check if header needs to be written
        header_needed = not os.path.exists(filename) or os.path.getsize(filename) == 0

        with open(filename, "a") as f:
            if header_needed:
                f.write("timestamp,readable_time,status\n")
            f.write(f"{timestamp},{readable},ALIVE\n")
    except Exception as e:
        logging.error(f"Error Logging Uptime: {e}")
