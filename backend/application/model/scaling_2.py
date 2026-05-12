"""Compatibility wrapper for final dataset scaling.

The implementation lives in ``backend/prepare_dataset.py``.
"""

from prepare_dataset import descaling, scale_dataset


def scaling(input_parquet: str, output_parquet: str, encoder_path: str):
    """Scale the unscaled stop-level dataset."""
    return scale_dataset(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        encoder_path=encoder_path,
    )


if __name__ == "__main__":
    from prepare_dataset import FINAL_DATASET_PATH, ROUTE_ENCODER_PATH, UNSCALED_DATASET_PATH

    scaling(
        input_parquet=str(UNSCALED_DATASET_PATH),
        output_parquet=str(FINAL_DATASET_PATH),
        encoder_path=str(ROUTE_ENCODER_PATH),
    )
