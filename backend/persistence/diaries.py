"""
Diaries Persistence - Parquet I/O operations for diary data.
"""

import logging
import os
import pandas as pd

DIARIES_PATH = "diaries/"
DIARIES_FILE = "diaries.parquet"


def val(path):
    """Check if path exists and has content."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def readParquet(path):
    """Read a parquet file."""
    return pd.read_parquet(path, engine="pyarrow")


def writeParquet(df, path):
    """Write a dataframe to parquet."""
    df.to_parquet(path, engine="pyarrow")


def updateParquet(df, path):
    """Append to existing parquet or create new."""
    if val(path):
        try:
            existing_df = pd.read_parquet(path, engine="pyarrow")
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(path, engine="pyarrow")
            return True
        except Exception as e:
            logging.error(f"Error appending to parquet: {e}. Saving to new file instead.")
            base, ext = os.path.splitext(path)
            new_path = f"{base}_new{ext}"
            df.to_parquet(new_path, engine="pyarrow")
            logging.info(f"Saved to {new_path}")
            return False
    else:
        df.to_parquet(path, engine="pyarrow")
        return False


def get_latest_diary_index():
    """Get the highest index from numbered diary files."""
    if not os.path.exists(DIARIES_PATH):
        return -1
    indices = [
        int(f.split(".")[0].split("_")[-1])
        for f in os.listdir(DIARIES_PATH)
        if f.startswith("diaries_")
        and f.endswith(".parquet")
        and f.split(".")[0].split("_")[-1].isdigit()
    ]
    return max(indices) if indices else -1


def save_diaries(diaries_list, saving_strategy, filename="diaries.parquet"):
    """Saves a list of diary dictionaries using the given strategy."""
    if not diaries_list:
        logging.info("No diaries to save.")
        return
    try:
        saving_strategy.execute(diaries_list, filename)
    except Exception as e:
        logging.error(f"Error saving diaries: {e}")


def save_diaries_incremental(new_diaries, filename=None):
    """
    Append NEW diaries to a single file, deduplicating by (trip_id, measurement_time).
    """
    if not new_diaries:
        return 0

    if filename is None:
        filename = DIARIES_FILE

    os.makedirs(DIARIES_PATH, exist_ok=True)
    full_path = os.path.join(DIARIES_PATH, filename)

    new_df = pd.DataFrame(new_diaries)

    # Ensure proper column types
    for col in ["sequence", "stop_sequence", "weather_code"]:
        if col in new_df.columns:
            new_df[col] = (
                pd.to_numeric(new_df[col], errors="coerce").fillna(0).astype("int64")
            )

    try:
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            existing_df = pd.read_parquet(full_path, engine="pyarrow")
            
            # Migrate existing schema if needed
            if "weather_code" in existing_df.columns and existing_df["weather_code"].dtype == "object":
                logging.warning("Migrating weather_code from String to Int in existing parquet...")
                existing_df["weather_code"] = (
                    pd.to_numeric(existing_df["weather_code"], errors="coerce")
                    .fillna(0)
                    .astype("int64")
                )
            
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["trip_id", "measurement_time"], keep="last"
            )
            new_count = len(combined) - len(existing_df)
        else:
            combined = new_df
            new_count = len(new_df)

        combined.to_parquet(full_path, engine="pyarrow")
        if new_count > 0:
            logging.info(
                f"📦 Saved {new_count} new diary entries to {full_path} (total: {len(combined)})"
            )
        return new_count
    except Exception as e:
        logging.error(f"Error saving diaries incrementally: {e}")
        return 0


def saving_loop(
    observatory, saving_strategy, STOP_EVENT, cache_strategy=None, timeout=1800
):
    """
    Background saving loop with smart dirty checking.

    - Saves diaries incrementally to single file (with deduplication)
    - Saves city cache only when street count increases
    """
    last_diary_count = 0
    last_street_count = 0

    # Get initial counts
    diaries = observatory.get_completed_diaries()
    last_diary_count = len(diaries) if diaries else 0

    city = observatory.get_city("Rome")
    if city and cache_strategy and hasattr(cache_strategy, "get_city_cache_size"):
        last_street_count = cache_strategy.get_city_cache_size(city)

    while not STOP_EVENT.is_set():
        # Wait for timeout (30m by default) OR until Stop Event is set
        if STOP_EVENT.wait(timeout):
            logging.info("Auto-Save loop stopping...")
            break

        try:
            # === DIARIES ===
            diaries_list = observatory.get_completed_diaries()
            current_diary_count = len(diaries_list) if diaries_list else 0

            if current_diary_count > last_diary_count:
                save_diaries_incremental(diaries_list)
                last_diary_count = current_diary_count
            else:
                logging.info("📦 No new diaries to save")

            # === CITY CACHE ===
            if city and cache_strategy and hasattr(cache_strategy, "save_city_cache"):
                current_street_count = cache_strategy.get_city_cache_size(city)
                if current_street_count > last_street_count:
                    cache_strategy.save_city_cache(city)
                    last_street_count = current_street_count
                else:
                    logging.info("🗺️ No new streets to cache")

        except Exception as e:
            logging.error(f"Error in saving loop: {e}")

    # Final Save on Exit
    logging.info("Performing final auto-save...")
    try:
        diaries_list = observatory.get_completed_diaries()
        if diaries_list:
            save_diaries_incremental(diaries_list)

        if city and cache_strategy and hasattr(cache_strategy, "save_city_cache"):
            cache_strategy.save_city_cache(city)
    except Exception as e:
        logging.error(f"Error in final save: {e}")
