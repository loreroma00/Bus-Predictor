"""
Cache Diagnostic Script - Run this to identify cache hit/miss patterns.

Usage: .\env\Scripts\python.exe scripts\diagnose_cache.py
"""

import pickle
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from application.domain.cities import City, Hexagon
from application.domain.map_info import AsyncGeocodingService
from application.domain import h3_utils


def load_cache(cache_path="city_cache.pkl"):
    """Load and return cache data."""
    if not os.path.exists(cache_path):
        print(f"ERROR: Cache file not found: {cache_path}")
        return None

    with open(cache_path, "rb") as f:
        return pickle.load(f)


def analyze_cache_format(data):
    """Analyze the format of cached data."""
    print("\n=== CACHE FORMAT ANALYSIS ===")
    print(f"Total hexagons: {len(data)}")

    total_streets = 0
    sample_keys = []

    for hex_id, streets in list(data.items())[:5]:
        print(f"\nHexagon: {hex_id}")
        print(f"  Streets count: {len(streets)}")

        if streets:
            first_key = list(streets.keys())[0]
            first_value = streets[first_key]
            print(f"  Key type: {type(first_key)}")
            print(f"  Sample key: {first_key}")
            print(f"  Sample value: {first_value}")
            sample_keys.append((hex_id, first_key))

        total_streets += len(streets)

    print(f"\n  Total street entries: {total_streets}")
    return sample_keys


def test_cache_loading(data):
    """Simulate how main.py loads the cache."""
    print("\n=== SIMULATING CACHE LOAD (as main.py does) ===")

    city = City("Rome")
    restored = 0

    for hex_id, streets in data.items():
        if hex_id not in city.hexagons:
            city.hexagons[hex_id] = Hexagon(hex_id)
        city.hexagons[hex_id].streets = streets
        restored += len(streets)

    print(f"Loaded: {len(city.hexagons)} hexagons, {restored} street entries")
    return city


def test_lookup(city, sample_coords):
    """Test if lookup works for sample coordinates."""
    print("\n=== TESTING LOOKUPS ===")

    for hex_id, coords in sample_coords:
        lat, lon = coords

        # Method 1: Direct hexagon lookup (what process_one uses)
        hexagon = city.get_hexagon(hex_id)
        direct_result = hexagon.get_street_by_coords(lat, lon) if hexagon else None

        # Method 2: Compute hex from coords (what enqueue uses)
        computed_hex = h3_utils.get_h3_index(lat, lon)
        computed_hexagon = city.get_hexagon(computed_hex)
        computed_result = None
        if computed_hexagon:
            computed_result = computed_hexagon.get_street_by_coords(lat, lon)

        # Check for rounding consistency
        rounded_key = (round(lat, 5), round(lon, 5))

        print(f"\nCoords: ({lat}, {lon})")
        print(f"  Stored in hex:   {hex_id}")
        print(f"  Computed hex:    {computed_hex}")
        print(f"  Hex match:       {hex_id == computed_hex}")
        print(f"  Rounded key:     {rounded_key}")
        print(
            f"  Key in streets:  {rounded_key in (hexagon.streets if hexagon else {})}"
        )
        print(f"  Direct lookup:   {direct_result}")
        print(f"  Computed lookup: {computed_result}")


def test_enqueue_dedup(city):
    """Test if enqueue properly skips cached coordinates."""
    print("\n=== TESTING ENQUEUE DEDUPLICATION ===")

    service = AsyncGeocodingService(city)

    # Get a cached coordinate
    for hex_id, hexagon in city.hexagons.items():
        if hexagon.streets:
            coords = list(hexagon.streets.keys())[0]
            lat, lon = coords

            print(f"Testing cached coord: ({lat}, {lon}) in hex {hex_id}")
            print(f"  Before enqueue: queue size = {service.get_queue_size()}")

            # Enqueue should skip this (it's cached)
            service.enqueue(lat, lon, hex_id)
            print(f"  After enqueue:  queue size = {service.get_queue_size()}")

            # Now test with a NEW coordinate
            new_lat, new_lon = lat + 0.001, lon + 0.001
            new_hex = h3_utils.get_h3_index(new_lat, new_lon)
            print(f"\nTesting NEW coord: ({new_lat}, {new_lon}) in hex {new_hex}")
            service.enqueue(new_lat, new_lon, new_hex)
            print(f"  After enqueue:  queue size = {service.get_queue_size()}")

            break


def check_key_format_consistency(data):
    """Check if all keys in cache have consistent format."""
    print("\n=== KEY FORMAT CONSISTENCY CHECK ===")

    issues = []
    for hex_id, streets in data.items():
        for key, street in streets.items():
            if not isinstance(key, tuple) or len(key) != 2:
                issues.append(f"Bad key format in {hex_id}: {key} -> {type(key)}")
            else:
                lat, lon = key
                rounded = (round(lat, 5), round(lon, 5))
                if key != rounded:
                    issues.append(f"Unrounded key in {hex_id}: {key} vs {rounded}")

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("All keys are properly formatted tuples with 5 decimal precision.")


def main():
    print("=" * 60)
    print("GEOCODING CACHE DIAGNOSTIC")
    print("=" * 60)

    # 1. Load cache
    data = load_cache()
    if not data:
        return

    # 2. Analyze format
    sample_coords = analyze_cache_format(data)

    # 3. Check key consistency
    check_key_format_consistency(data)

    # 4. Simulate loading
    city = test_cache_loading(data)

    # 5. Test lookups
    test_lookup(city, sample_coords)

    # 6. Test enqueue dedup
    test_enqueue_dedup(city)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
