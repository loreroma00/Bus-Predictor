"""Parquet persistence for completed live-trip measurements."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def _measurement_file(
    measurements_path: str | Path,
    measurements_file: str,
    filename: str | None = None,
) -> Path:
    """Return the configured completed-measurements parquet path."""
    return Path(measurements_path) / (filename or measurements_file)


def save_measurements_incremental(
    new_measurements: list[dict],
    filename: str | None = None,
    measurements_path: str | Path = "measurements/",
    measurements_file: str = "measurements.parquet",
) -> int:
    """Append completed measurements, deduplicating by trip and measurement time."""
    if not new_measurements:
        return 0

    path = Path(measurements_path)
    path.mkdir(parents=True, exist_ok=True)
    full_path = _measurement_file(path, measurements_file, filename)
    new_df = pd.DataFrame(new_measurements)

    for col in ["sequence", "stop_sequence", "weather_code"]:
        if col in new_df.columns:
            new_df[col] = (
                pd.to_numeric(new_df[col], errors="coerce")
                .fillna(0)
                .astype("int64")
            )

    try:
        if full_path.exists() and full_path.stat().st_size > 0:
            existing_df = pd.read_parquet(full_path, engine="pyarrow")
            if (
                "weather_code" in existing_df.columns
                and existing_df["weather_code"].dtype == "object"
            ):
                logging.warning(
                    "Migrating weather_code from string to int in existing parquet..."
                )
                existing_df["weather_code"] = (
                    pd.to_numeric(existing_df["weather_code"], errors="coerce")
                    .fillna(0)
                    .astype("int64")
                )

            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["trip_id", "measurement_time"],
                keep="last",
            )
            new_count = len(combined) - len(existing_df)
        else:
            combined = new_df
            new_count = len(new_df)

        combined.to_parquet(full_path, engine="pyarrow")
        if new_count > 0:
            logging.info(
                "Saved %s new measurement rows to %s (total: %s)",
                new_count,
                full_path,
                len(combined),
            )
        return new_count
    except Exception as exc:
        logging.error("Error saving measurements incrementally: %s", exc)
        return 0


def saving_loop(
    observatory,
    stop_event,
    cache_strategy=None,
    timeout: int = 1800,
) -> None:
    """Persist completed measurements and city cache on a background interval."""
    measurements = observatory.get_completed_measurements()
    last_measurement_count = len(measurements) if measurements else 0

    city = observatory.get_city("Rome")
    last_street_count = 0
    if city and cache_strategy and hasattr(cache_strategy, "get_city_cache_size"):
        last_street_count = cache_strategy.get_city_cache_size(city)

    while not stop_event.is_set():
        if stop_event.wait(timeout):
            logging.info("Auto-save loop stopping...")
            break

        try:
            measurements = observatory.get_completed_measurements()
            current_measurement_count = len(measurements) if measurements else 0
            if current_measurement_count > last_measurement_count:
                save_measurements_incremental(measurements)
                last_measurement_count = current_measurement_count
            else:
                logging.info("No new completed measurements to save")

            if city and cache_strategy and hasattr(cache_strategy, "save_city_cache"):
                current_street_count = cache_strategy.get_city_cache_size(city)
                if current_street_count > last_street_count:
                    cache_strategy.save_city_cache(city)
                    last_street_count = current_street_count
                else:
                    logging.info("No new streets to cache")

        except Exception as exc:
            logging.error("Error in saving loop: %s", exc)

    logging.info("Performing final auto-save...")
    try:
        measurements = observatory.get_completed_measurements()
        if measurements:
            save_measurements_incremental(measurements)

        if city and cache_strategy and hasattr(cache_strategy, "save_city_cache"):
            cache_strategy.save_city_cache(city)
    except Exception as exc:
        logging.error("Error in final save: %s", exc)
