#!/usr/bin/env python3
"""
ATAC Bus Delay Prediction CLI Tool.

Interactively prompts for bus line, direction, date, and time,
then finds the closest scheduled trip and queries the prediction API.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import joblib

from static_data import StaticDataFetcher
from weather import get_weather_code, get_weather_description
from api_client import APIClient


def load_supported_routes(cli_dir: str) -> list[str]:
    """Load list of routes supported by the model from route_encoder.pkl."""
    path = os.path.join(cli_dir, "route_encoder.pkl")
    encoder_data = joblib.load(path)
    route_encoder = encoder_data["route"]
    return list(route_encoder.classes_)


def ensure_static_data(base_path: str) -> bool:
    """Ensure GTFS static data is available. Returns True on success."""
    fetcher = StaticDataFetcher(base_path)
    fetcher.fetch()

    required = ["routes.txt", "trips.txt", "stop_times.txt", "calendar_dates.txt"]
    for fn in required:
        if not os.path.exists(os.path.join(base_path, fn)):
            print(f"Error: {fn} not found after extraction.")
            return False
    return True


def load_gtfs_data(base_path: str) -> dict[str, pd.DataFrame]:
    """Load all needed GTFS files into DataFrames."""
    files = ["routes.txt", "trips.txt", "stop_times.txt", "calendar_dates.txt"]
    data = {}
    for fn in files:
        data[fn.replace(".txt", "")] = pd.read_csv(
            os.path.join(base_path, fn),
            sep=",",
            header=0,
            dtype=str,
            low_memory=False,
        )
    return data


def get_available_routes(trips: pd.DataFrame, supported: list[str]) -> list[str]:
    """Get sorted list of route_ids that are supported by the model."""
    all_routes = set(trips["route_id"].unique())
    return sorted(all_routes & set(supported))


def get_route_directions(trips: pd.DataFrame, route_id: str) -> list[tuple[int, str]]:
    """
    Get available directions for a route.
    Returns list of (direction_id, trip_headsign) tuples.
    """
    route_trips = trips[trips["route_id"] == route_id]
    directions = (
        route_trips.groupby(["direction_id", "trip_headsign"])
        .first()
        .reset_index()[["direction_id", "trip_headsign"]]
    )
    result = []
    for _, row in directions.iterrows():
        result.append((int(row["direction_id"]), row["trip_headsign"]))
    return result


def find_closest_trip(
    data: dict[str, pd.DataFrame],
    route_id: str,
    direction_id: int,
    user_date: str,
    user_time: str,
) -> Optional[tuple[str, str, str]]:
    """
    Find the scheduled trip closest to user's requested date/time.

    Args:
        data: GTFS dataframes dict
        route_id: Bus line
        direction_id: Direction (0 or 1)
        user_date: DD-MM-YYYY format
        user_time: HH:MM format

    Returns:
        Tuple of (trip_id, scheduled_start_time, scheduled_start_date)
        or None if no trip found.
    """
    trips = data["trips"]
    stop_times = data["stop_times"]
    calendar_dates = data["calendar_dates"]

    date_obj = datetime.strptime(user_date, "%d-%m-%Y")
    date_yyyymmdd = date_obj.strftime("%Y%m%d")
    user_minutes = int(user_time.split(":")[0]) * 60 + int(user_time.split(":")[1])

    service_ids = set(
        calendar_dates[calendar_dates["date"] == date_yyyymmdd]["service_id"].tolist()
    )

    if not service_ids:
        return None

    route_trips = trips[
        (trips["route_id"] == route_id)
        & (trips["direction_id"].astype(int) == direction_id)
        & (trips["service_id"].isin(service_ids))
    ]

    if route_trips.empty:
        return None

    trip_ids = route_trips["trip_id"].tolist()

    first_stops = stop_times[
        (stop_times["trip_id"].isin(trip_ids)) & (stop_times["stop_sequence"] == "1")
    ][["trip_id", "departure_time"]]

    if first_stops.empty:
        return None

    best_trip = None
    best_diff = float("inf")

    for _, row in first_stops.iterrows():
        dep_time = row["departure_time"]
        parts = dep_time.split(":")
        dep_minutes = int(parts[0]) * 60 + int(parts[1])

        diff = abs(dep_minutes - user_minutes)
        if diff < best_diff:
            best_diff = diff
            best_trip = (row["trip_id"], dep_time[:5], user_date)

    return best_trip


def find_operating_dates(
    data: dict[str, pd.DataFrame],
    route_id: str,
    direction_id: int,
    around_date: str,
    num_days: int = 7,
) -> list[str]:
    """
    Find dates when the route operates, around the given date.

    Returns list of dates in DD-MM-YYYY format.
    """
    trips = data["trips"]
    calendar_dates = data["calendar_dates"]

    route_service_ids = set(
        trips[
            (trips["route_id"] == route_id)
            & (trips["direction_id"].astype(int) == direction_id)
        ]["service_id"].unique()
    )

    if not route_service_ids:
        return []

    route_dates = calendar_dates[
        (calendar_dates["service_id"].isin(route_service_ids))
        & (calendar_dates["exception_type"] == "1")
    ]["date"].unique()

    date_obj = datetime.strptime(around_date, "%d-%m-%Y")
    result = []

    for offset in range(-num_days, num_days + 1):
        check_date = date_obj + timedelta(days=offset)
        check_yyyymmdd = check_date.strftime("%Y%m%d")
        if check_yyyymmdd in route_dates:
            result.append(check_date.strftime("%d-%m-%Y"))

    return sorted(result)


def prompt_line(available_routes: list[str]) -> str:
    """Prompt user to select a bus line."""
    print("\n" + "=" * 50)
    print("ATAC Bus Delay Prediction")
    print("=" * 50)
    print("\nAvailable lines: ", end="")
    sample = available_routes[:20]
    print(", ".join(sample))
    if len(available_routes) > 20:
        print(f"... and {len(available_routes) - 20} more")

    while True:
        line = input("\nEnter line number (e.g., 211, C2, 05): ").strip()
        if line in available_routes:
            return line
        print(f"Line '{line}' not found. Please try again.")


def prompt_direction(directions: list[tuple[int, str]]) -> int:
    """Prompt user to select a direction. Returns direction_id."""
    print("\nAvailable directions:")
    for dir_id, headsign in directions:
        print(f"  [{dir_id}] {headsign}")

    valid_ids = [d[0] for d in directions]
    while True:
        choice = input("\nSelect direction (0 or 1): ").strip()
        try:
            dir_id = int(choice)
            if dir_id in valid_ids:
                return dir_id
            print(f"Please enter one of: {valid_ids}")
        except ValueError:
            print("Invalid input. Enter 0 or 1.")


def prompt_date() -> str:
    """Prompt for date in DD-MM-YYYY format."""
    while True:
        date_str = input("\nEnter date (DD-MM-YYYY): ").strip()
        try:
            datetime.strptime(date_str, "%d-%m-%Y")
            return date_str
        except ValueError:
            print("Invalid format. Use DD-MM-YYYY (e.g., 15-03-2025)")


def prompt_time() -> str:
    """Prompt for time in HH:MM format."""
    while True:
        time_str = input("Enter time (HH:MM): ").strip()
        try:
            datetime.strptime(time_str, "%H:%M")
            return time_str
        except ValueError:
            print("Invalid format. Use HH:MM (e.g., 14:30)")


def prompt_bus_type() -> int:
    """Prompt for bus type. Defaults to 1 if empty."""
    while True:
        bus_type_str = input("Enter bus type (press Enter for default 1): ").strip()
        if not bus_type_str:
            return 1
        try:
            return int(bus_type_str)
        except ValueError:
            print("Invalid input. Enter a number or press Enter for default.")


def get_stop_scheduled_times(
    stop_times: pd.DataFrame, trip_id: str
) -> dict[int, tuple[str, float]]:
    """
    Get scheduled arrival times and distances for each stop in a trip.

    Returns dict mapping stop_sequence -> (scheduled_arrival_time, shape_dist_traveled).
    """
    trip_stops = stop_times[stop_times["trip_id"] == trip_id][
        ["stop_sequence", "arrival_time", "shape_dist_traveled"]
    ]
    result = {}
    for _, row in trip_stops.iterrows():
        stop_seq = int(row["stop_sequence"])
        arrival = row["arrival_time"]
        dist = float(row["shape_dist_traveled"]) if row["shape_dist_traveled"] else 0.0
        result[stop_seq] = (arrival, dist)
    return result


def time_str_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to seconds since midnight. Handles 24+ hours."""
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2]) if len(parts) > 2 else 0
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_time_str(seconds: int) -> str:
    """Convert seconds since midnight to HH:MM:SS."""
    seconds = seconds % 86400
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_delay(delay_seconds: float) -> str:
    """Format delay as +Xm Ys or -Xm Ys."""
    total_seconds = int(round(delay_seconds))
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{sign}{minutes}m {seconds}s"


def display_prediction(
    prediction: dict, trip_id: str, stop_info: dict[int, tuple[str, float]]
):
    """Display prediction results in a formatted table."""
    print("\n" + "=" * 80)
    print(f"Route: {prediction['route_id']} | Direction: {prediction['direction_id']}")
    print(f"Trip ID: {trip_id}")
    print(
        f"Date: {prediction['trip_date']} | Scheduled Start: {prediction['scheduled_start']}"
    )
    print("=" * 80)
    print(
        f"{'Stop':>5} | {'Distance':>10} | {'Scheduled':>10} | {'Delay':>12} | {'Expected':>10} | {'Crowd':>6}"
    )
    print("-" * 80)

    for stop in prediction["stops"]:
        stop_seq = stop["stop_sequence"]
        delay_sec = stop["cumulative_delay_sec"]

        info = stop_info.get(stop_seq)
        if info:
            scheduled_str, distance_m = info
            sched_seconds = time_str_to_seconds(scheduled_str)
            expected_seconds = sched_seconds + delay_sec
            expected_str = seconds_to_time_str(int(expected_seconds))
        else:
            scheduled_str = "N/A"
            distance_m = stop["distance_m"]
            expected_str = "N/A"

        delay_formatted = format_delay(delay_sec)

        print(
            f"{stop_seq:>5} | "
            f"{distance_m:>10.0f}m | "
            f"{scheduled_str:>10} | "
            f"{delay_formatted:>12} | "
            f"{expected_str:>10} | "
            f"{stop['crowd_level']:>6}"
        )
    print("=" * 80)


def display_validation(report: dict):
    """Display validation results."""
    print("\n" + "=" * 80)
    print(f"MODEL VALIDATION REPORT - Date: {report['date']}")
    print("=" * 80)
    print(f"Total Scheduled Trips:      {report['total_scheduled_trips']}")
    print(f"Trips with Ground Truth:    {report['total_trips_with_ground_truth']}")
    print(f"Trips Predicted:            {report['total_trips_predicted']}")
    print(f"Trips Validated:            {report['total_trips_validated']}")
    print(f"Total Measurements:         {report['total_measurements']}")
    print("-" * 80)
    print(f"Median RMSE:                {report['median_rmse']:.2f} s")
    print(f"Median MSE:                 {report['median_mse']:.2f} s²")
    print(f"RMSE Range:                 [{report['min_rmse']:.2f}, {report['max_rmse']:.2f}]")
    print("-" * 80)
    print(f"Log File:                   {report['log_file']}")
    print(f"Report File:                {report['report_file']}")
    print("=" * 80)

    if report["trips"]:
        print(f"\nTrip Samples (first 10 of {len(report['trips'])}):")
        print(f"{'Route':>6} | {'Dir':>3} | {'Start':>10} | {'RMSE':>8} | {'N':>4} | {'Error'}")
        print("-" * 80)
        for trip in report["trips"][:10]:
            error_str = trip["error"] if trip["error"] else ""
            print(
                f"{trip['route_id']:>6} | "
                f"{trip['direction_id']:>3} | "
                f"{trip['scheduled_start']:>10} | "
                f"{trip['rmse']:>8.1f} | "
                f"{trip['n_measurements']:>4} | "
                f"{error_str}"
            )
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="ATAC Bus Delay Prediction CLI")
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API base URL (default: https://atacapi.loreromaphotos.it)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory for GTFS static data (default: current directory)",
    )
    parser.add_argument(
        "--test-model",
        type=str,
        metavar="YYYY-MM-DD",
        help="Validate model against ground truth for a specific date",
    )
    args = parser.parse_args()

    api_client = APIClient(args.api_url)

    if args.test_model:
        # Convert YYYY-MM-DD to DD-MM-YYYY for the API
        try:
            date_obj = datetime.strptime(args.test_model, "%Y-%m-%d")
            api_date = date_obj.strftime("%d-%m-%Y")
        except ValueError:
            # Maybe it's already DD-MM-YYYY
            try:
                datetime.strptime(args.test_model, "%d-%m-%Y")
                api_date = args.test_model
            except ValueError:
                print(f"Error: Invalid date format '{args.test_model}'. Use YYYY-MM-DD.")
                sys.exit(1)

        print(f"Validating model for date: {api_date}...")
        try:
            report = api_client.validate(api_date)
            display_validation(report)
            sys.exit(0)
        except requests.HTTPError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = e.response.text or str(e)
            print(f"API Error ({e.response.status_code}): {detail}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    cli_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = args.data_dir or os.getcwd()

    print("Fetching static data...")
    if not ensure_static_data(base_path):
        sys.exit(1)

    print("Loading GTFS data...")
    data = load_gtfs_data(base_path)
    supported_routes = load_supported_routes(cli_dir)
    available_routes = get_available_routes(data["trips"], supported_routes)

    api_client = APIClient(args.api_url)
    print(f"API URL: {api_client.base_url}")

    if not api_client.health_check():
        print("Warning: API health check failed. Continuing anyway...")

    route_id = prompt_line(available_routes)
    directions = get_route_directions(data["trips"], route_id)

    if not directions:
        print(f"No directions found for line {route_id}")
        sys.exit(1)

    direction_id = prompt_direction(directions)
    user_date = prompt_date()
    user_time = prompt_time()

    print(f"\nFinding closest scheduled trip to {user_time} on {user_date}...")
    trip_info = find_closest_trip(data, route_id, direction_id, user_date, user_time)

    if not trip_info:
        operating_dates = find_operating_dates(
            data, route_id, direction_id, user_date, num_days=7
        )
        if operating_dates:
            print(f"No trips on {user_date}. This route operates on:")
            for d in operating_dates[:10]:
                print(f"  - {d}")
            if len(operating_dates) > 10:
                print(f"  ... and {len(operating_dates) - 10} more dates")
        else:
            print("No scheduled trips found for this route/direction.")
        sys.exit(1)

    trip_id, scheduled_time, scheduled_date = trip_info
    print(f"Found trip {trip_id} departing at {scheduled_time}")

    stop_info = get_stop_scheduled_times(data["stop_times"], trip_id)

    bus_type = prompt_bus_type()

    print("\nFetching weather...")
    try:
        weather_code = get_weather_code()
        print(
            f"Weather: {get_weather_description(weather_code)} (code: {weather_code})"
        )
    except Exception as e:
        print(f"Warning: Could not fetch weather ({e}). Using default code 0.")
        weather_code = 0

    print("\nRequesting prediction...")
    try:
        prediction = api_client.predict(
            route_id=route_id,
            direction_id=direction_id,
            start_date=scheduled_date,
            start_time=scheduled_time,
            weather_code=weather_code,
            bus_type=bus_type,
        )
        display_prediction(prediction, trip_id, stop_info)
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = e.response.text or str(e)
        print(f"API Error ({e.response.status_code}): {detail}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
