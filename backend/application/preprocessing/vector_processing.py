"""Compatibility wrapper for stop-level dataset construction.

The implementation lives in ``backend/prepare_dataset.py``.
"""

from prepare_dataset import (
    Filter,
    build_h3_encoding,
    build_static_map_index,
    build_stop_level_dataset,
    build_traffic_avg_index,
    compute_static_features,
    interpolate_schedule_adherence,
    load_dynamic_data,
    process_single_trip,
    unroll_time,
)


def process_data(start_date: str = None):
    """Build the unscaled stop-level dataset."""
    return build_stop_level_dataset(start_date=start_date)


if __name__ == "__main__":
    process_data()
