from datetime import datetime, timedelta
from math import pi, sin, cos


def to_unix_time(time_input, date_ref=None):
    """
    Converts various time inputs to UNIX timestamp (int).
    - int/float: returned as-is (assumed already UNIX)
    - str "HH:MM:SS": converted to date_ref (or today) in local TZ. Supports >24h.
    - datetime: converted to UNIX timestamp
    """
    if time_input is None:
        return None
    if isinstance(time_input, (int, float)):
        return int(time_input)
    if isinstance(time_input, str):
        try:
            h, m, s = map(int, time_input.split(":"))

            local_tz = datetime.now().astimezone().tzinfo

            if date_ref:
                base = date_ref.replace(
                    hour=0, minute=0, second=0, microsecond=0, tzinfo=local_tz
                )
            else:
                base = datetime.now(local_tz).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

            dt = base + timedelta(hours=h, minutes=m, seconds=s)
            return int(dt.timestamp())
        except Exception:
            return None
    if isinstance(time_input, datetime):
        return int(time_input.timestamp())
    return None


def to_readable_time(unix_timestamp, fmt="%H:%M:%S"):
    """
    Converts UNIX timestamp to human-readable local time string.
    Automatically adjusts for daylight saving time.
    """
    if unix_timestamp is None:
        return "N/A"
    try:
        local_tz = datetime.now().astimezone().tzinfo
        dt = datetime.fromtimestamp(unix_timestamp, tz=local_tz)
        return dt.strftime(fmt)
    except Exception:
        return "Invalid"


def get_seconds_since_midnight(unix_timestamp):
    """
    Converts a UNIX timestamp to seconds since midnight (local time).
    """
    if unix_timestamp is None:
        return 0
    try:
        local_tz = datetime.now().astimezone().tzinfo
        dt = datetime.fromtimestamp(unix_timestamp, tz=local_tz)
        return dt.hour * 3600 + dt.minute * 60 + dt.second
    except Exception:
        return 0


def get_timestamp_components(unix_timestamp: float) -> tuple[str, str, str] | None:
    """
    Returns a tuple of (date, day, time) from a UNIX timestamp.
    Example: ('2023-10-27', 'Friday', '14:30:05')
    """
    if unix_timestamp is None:
        return None
    try:
        local_tz = datetime.now().astimezone().tzinfo
        dt = datetime.fromtimestamp(unix_timestamp, tz=local_tz)
        return (
            dt.strftime("%Y-%m-%d"),
            dt.strftime("%A"),
            dt.strftime("%H:%M:%S"),
        )
    except Exception:
        return None


def get_time_sin_cos(unix_timestamp: float) -> tuple[float, float] | None:
    """
    Returns the sine and cosine of the time of day from a UNIX timestamp.
    Useful for cyclical time features in machine learning models.
    """
    if unix_timestamp is None:
        return None
    try:
        seconds = get_seconds_since_midnight(unix_timestamp)
        seconds_in_day = 86400
        angle = 2 * pi * (seconds / seconds_in_day)
        return sin(angle), cos(angle)
    except Exception:
        return None


def get_time_sin_cos_from_str(time_str: str) -> tuple[float, float] | None:
    """
    Returns the sine and cosine of the time from a "HH:MM:SS" string.
    Useful for encoding scheduled start times from GTFS-RT feeds.
    """
    if time_str is None:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) != 3:
            return None
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        seconds = h * 3600 + m * 60 + s
        seconds_in_day = 86400
        angle = 2 * pi * (seconds / seconds_in_day)
        return sin(angle), cos(angle)
    except Exception:
        return None
