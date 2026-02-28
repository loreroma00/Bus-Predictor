import h3
import osmnx as ox
import shapely.geometry as geom


def get_h3_index(lat: float, lng: float, resolution: int = 9) -> str:
    return h3.latlng_to_cell(lat, lng, resolution)


def get_coords_from_h3(h3_index: str) -> tuple[float, float]:
    return h3.cell_to_latlng(h3_index)


def get_neighbours_from_h3(h3_index: str, k: int = 1) -> set:
    return h3.grid_disk(h3_index, k)


def get_hexagons_for_city(city_name: str, resolution: int = 9) -> set[str]:
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
