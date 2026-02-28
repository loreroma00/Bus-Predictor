"""
Cache - Pickle-based caching for non-diary data (uptime logging, etc.).
"""

import logging
import os
from datetime import datetime
import time


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
