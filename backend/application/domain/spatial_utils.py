import logging
import math
import h3


def _derive_distance_moved(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Haversine formula to calculate distance in km
    lat1, lon1 = (
        math.radians(lat1),
        math.radians(lon1),
    )
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    # Calculate the distance in km
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance_km = 6371 * c
    return distance_km


def derive_speed(
    lat1: float, lon1: float, lat2: float, lon2: float, timestamp1, timestamp2
) -> float:
    distance_km = _derive_distance_moved(lat1, lon1, lat2, lon2)

    if distance_km < 0.01:
        return 0

    # Calculate the time difference in hours
    time_diff = (timestamp2 - timestamp1) / 3600

    # Calculate the speed in km/h
    derived_speed = distance_km / time_diff if time_diff > 0 else 0
    return derived_speed


def derive_bearing(lat1: float, lon1: float, lat2: float, lon2: float):
    distance_km = _derive_distance_moved(lat1, lon1, lat2, lon2)

    if distance_km < 0.01:
        return -1

    lat1, lon1 = (
        math.radians(lat1),
        math.radians(lon1),
    )
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    dlon = lon2 - lon1

    x = math.cos(lat2) * math.sin(dlon)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dlon
    )

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def get_cardinal_direction(bearing: float) -> str:
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((bearing + 22.5) / 45.0) % 8
    return directions[index]


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """
    Convert latitude/longitude to tile coordinates at given zoom level.

    Uses Web Mercator projection (EPSG:3857) tile system.
    """
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (x, y)


def tile_to_lat_lon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """
    Convert tile coordinates to latitude/longitude (northwest corner).
    """
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return (lat, lon)


def get_unique_tiles_from_hexagons(
    hex_ids: list[str], zoom: int = 15
) -> list[tuple[int, int]]:
    """
    Dato un elenco di esagoni H3, trova l'insieme MINIMO di tiles TomTom
    necessari per coprirli interamente.
    """
    unique_tiles = set()

    for hex_id in hex_ids:
        try:
            # Otteniamo il contorno (boundary) dell'esagono
            # h3.cell_to_boundary restituisce una lista di (lat, lon)
            boundary = h3.cell_to_boundary(hex_id)

            # Aggiungiamo anche il centro per sicurezza
            center = h3.cell_to_latlng(hex_id)
            points_to_check = [center] + list(boundary)

            for lat, lon in points_to_check:
                # Convertiamo ogni punto nella coordinata Tile corrispondente
                tx, ty = lat_lon_to_tile(lat, lon, zoom)
                unique_tiles.add((tx, ty))

        except Exception as e:
            logging.warning(f"Skipping invalid hex {hex_id}: {e}")
            continue

    return list(unique_tiles)
