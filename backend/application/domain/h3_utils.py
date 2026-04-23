"""H3 utilities — thin wrappers around the H3 library plus city-level hex coverage."""

import h3
import osmnx as ox
import shapely.geometry as geom


def get_h3_index(lat: float, lng: float, resolution: int = 9) -> str:
    """Return the H3 cell id containing ``(lat, lng)`` at the given resolution."""
    return h3.latlng_to_cell(lat, lng, resolution)


def get_coords_from_h3(h3_index: str) -> tuple[float, float]:
    """Return the ``(lat, lng)`` centre of an H3 cell."""
    return h3.cell_to_latlng(h3_index)


def get_neighbours_from_h3(h3_index: str, k: int = 1) -> set:
    """Return the ``k``-ring of H3 cells around ``h3_index`` (inclusive)."""
    return h3.grid_disk(h3_index, k)


def get_hexagons_for_city(city_name: str, resolution: int = 9) -> set[str]:
    """Return the set of H3 cells that tile the polygon of a geocoded city."""
    gdf = ox.geocode_to_gdf(city_name)
    geometry = gdf.geometry.iloc[0]

    polygons = []
    if isinstance(geometry, geom.MultiPolygon):
        polygons = list(geometry.geoms)
    else:
        polygons = [geometry]

    all_hexes = set()
    for poly in polygons:
        geo_json = geom.mapping(poly)
        hexes = h3.polygon_to_cells(geo_json, resolution)
        all_hexes.update(hexes)
    return all_hexes
