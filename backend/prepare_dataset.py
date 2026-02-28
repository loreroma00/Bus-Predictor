#!/usr/bin/env python
"""
Prepare Dataset - Orchestrates the full preprocessing pipeline.

Pipeline stages:
  1. canonical_shape_mapper - Generate canonical route map from GTFS static data
  2. preprocessing - Extract data from database to daily parquet files
  3. vector_processing - Interpolate and create unscaled training dataset
  4. scaling - Apply feature scaling and create final training dataset

Usage:
  python prepare_dataset.py [--skip-db] [--force-canonical] [--start-date YYYY-MM-DD]

Options:
  --skip-db          Skip preprocessing stage (use existing dataset_*.parquet files)
  --force-canonical  Force regeneration of canonical route map
  --start-date       Only process data from this date onwards (YYYY-MM-DD format)
"""

import os
import sys
import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PARQUET_DIR = PROJECT_ROOT / "parquets"
CANONICAL_MAP_PATH = PARQUET_DIR / "canonical_route_map.parquet"
GTFS_MD5_PATH = PARQUET_DIR / "gtfs_md5.json"


def get_remote_gtfs_md5() -> str | None:
    """Fetch the remote GTFS MD5 hash."""
    import requests as rq

    URL_MD5 = (
        "https://romamobilita.it/wp-content/uploads/drupal/rome_static_gtfs.zip.md5"
    )
    HEADERS = {
        "Referer": "https://romamobilita.it/sistemi-e-tecnologie/open-data/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        response = rq.get(URL_MD5, headers=HEADERS, timeout=15)
        return response.text.strip()
    except Exception as e:
        print(f"Warning: Could not fetch remote MD5: {e}")
        return None


def get_local_gtfs_md5() -> str | None:
    """Get the MD5 of the local GTFS zip file."""
    gtfs_zip = PROJECT_ROOT / "rome_static_gtfs.zip"
    if not gtfs_zip.exists():
        return None

    md5 = hashlib.md5()
    with open(gtfs_zip, "rb") as f:
        while chunk := f.read(4096):
            md5.update(chunk)
    return md5.hexdigest()


def get_cached_md5() -> str | None:
    """Get the cached MD5 from the last canonical map generation."""
    if GTFS_MD5_PATH.exists():
        try:
            with open(GTFS_MD5_PATH, "r") as f:
                data = json.load(f)
                return data.get("md5")
        except Exception:
            pass
    return None


def save_cached_md5(md5: str):
    """Save the MD5 hash after canonical map generation."""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    with open(GTFS_MD5_PATH, "w") as f:
        json.dump({"md5": md5}, f)


def needs_canonical_regeneration(force: bool = False) -> tuple[bool, str]:
    """
    Check if canonical route map needs regeneration.

    Returns:
        (needs_regeneration, reason)
    """
    if force:
        return True, "forced by user"

    if not CANONICAL_MAP_PATH.exists():
        return True, "canonical map does not exist"

    remote_md5 = get_remote_gtfs_md5()
    if remote_md5 is None:
        local_md5 = get_local_gtfs_md5()
        if local_md5 is None:
            return True, "no GTFS data available"
        return False, f"cannot check remote, using local GTFS (md5: {local_md5[:8]}...)"

    cached_md5 = get_cached_md5()
    if cached_md5 is None:
        return True, f"no cached MD5, remote has new version ({remote_md5[:8]}...)"

    if cached_md5 != remote_md5:
        return (
            True,
            f"GTFS updated: cached {cached_md5[:8]}... vs remote {remote_md5[:8]}...",
        )

    return False, f"canonical map is up to date (md5: {cached_md5[:8]}...)"


def run_canonical_mapper():
    """Run Stage 1: Canonical shape mapper."""
    print("\n" + "=" * 60)
    print("STAGE 1: Canonical Shape Mapper")
    print("=" * 60)

    from application.preprocessing.canonical_shape_mapper import main as mapper_main

    mapper_main()

    local_md5 = get_local_gtfs_md5()
    if local_md5:
        save_cached_md5(local_md5)
        print(f"Saved GTFS MD5: {local_md5}")


def run_preprocessing(start_date: str = None):
    """Run Stage 2: Preprocessing (DB extraction)."""
    print("\n" + "=" * 60)
    print("STAGE 2: Preprocessing (Database Extraction)")
    print("=" * 60)

    from application.preprocessing.preprocessing import main as preprocessing_main

    preprocessing_main(start_date=start_date)


def run_vector_processing():
    """Run Stage 3: Vector processing."""
    print("\n" + "=" * 60)
    print("STAGE 3: Vector Processing")
    print("=" * 60)

    from application.preprocessing.vector_processing import process_data

    process_data()


def run_scaling():
    """Run Stage 4: Scaling."""
    print("\n" + "=" * 60)
    print("STAGE 4: Scaling")
    print("=" * 60)

    from application.model.scaling_2 import scaling

    scaling(
        input_parquet=str(PARQUET_DIR / "dataset_lstm_unscaled.parquet"),
        output_parquet=str(PARQUET_DIR / "dataset_lstm_final.parquet"),
        encoder_path=str(PARQUET_DIR / "route_encoder.pkl"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages:
  1. canonical_shape_mapper - Generate canonical route map from GTFS
  2. preprocessing - Extract data from database
  3. vector_processing - Create unscaled training dataset
  4. scaling - Create final scaled training dataset
        """,
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip preprocessing stage (use existing dataset_*.parquet files)",
    )
    parser.add_argument(
        "--force-canonical",
        action="store_true",
        help="Force regeneration of canonical route map",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Only process data from this date onwards (YYYY-MM-DD format)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DATASET PREPARATION PIPELINE")
    print("=" * 60)

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    needs_regen, reason = needs_canonical_regeneration(force=args.force_canonical)
    print(f"\nCanonical map check: {reason}")

    if needs_regen:
        run_canonical_mapper()
    else:
        print("\nSkipping Stage 1: Canonical map is up to date")

    if not args.skip_db:
        run_preprocessing(start_date=args.start_date)
    else:
        print("\nSkipping Stage 2: Using existing dataset_*.parquet files")

    run_vector_processing()

    run_scaling()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {CANONICAL_MAP_PATH}")
    print(f"  - {PARQUET_DIR / 'dataset_lstm_final.parquet'}")
    print(f"  - {PARQUET_DIR / 'route_encoder.pkl'}")
    print(f"  - {PARQUET_DIR / 'route_encoding.json'}")
    print(f"  - {PARQUET_DIR / 'h3_encoding.json'}")


if __name__ == "__main__":
    main()
