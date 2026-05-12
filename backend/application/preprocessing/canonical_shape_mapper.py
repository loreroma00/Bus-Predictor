"""Compatibility wrapper for canonical map generation.

The implementation lives in ``backend/prepare_dataset.py``.
"""

from prepare_dataset import (
    build_canonical_shape_map,
    compute_traffic_averages,
    get_h3_index,
    haversine_np,
    process_stop_route_map,
)


def main():
    """Build the canonical stop-route map."""
    return build_canonical_shape_map()


if __name__ == "__main__":
    main()
