"""Compatibility wrapper for prediction-row preprocessing.

The implementation lives in ``backend/prepare_dataset.py``.
"""

from prepare_dataset import (
    extract_historical,
    extract_prediction_rows,
    extract_vehicle_trips,
    generate_synthetic_trip_id,
    get_processed_dates,
    preprocess_prediction_rows,
)


def main(start_date: str = None):
    """Extract and preprocess prediction rows from the database."""
    rows = extract_prediction_rows(start_date=start_date)
    return preprocess_prediction_rows(rows, start_date=start_date)


if __name__ == "__main__":
    main()
