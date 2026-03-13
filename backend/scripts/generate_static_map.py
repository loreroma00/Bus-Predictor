import sys
import os

# Add project root to sys.path to allow imports from application
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import geopandas as gpd  # noqa: E402
from shapely.geometry import LineString, MultiLineString  # noqa: E402
from application.domain.spatial_utils import derive_bearing  # noqa: E402
from application.domain.h3_utils import get_h3_index  # noqa: E402
from persistence import save_pickle  # noqa: E402

OUTPUT_FILE = "static_bus_lanes_roma.pkl"
INPUT_FILE = "corsie_preferenziali.geojson"


def process_geometry(geometry, static_map):
    """
    Process a geometry (LineString or MultiLineString) and update the static_map.
    """
    if isinstance(geometry, LineString):
        _process_linestring(geometry, static_map)
    elif isinstance(geometry, MultiLineString):
        for line in geometry.geoms:
            _process_linestring(line, static_map)


def _process_linestring(linestring, static_map):
    """
    Process a single LineString.
    Iterates through segments, calculates H3 index and bearing, and updates the map.
    """
    coords = list(linestring.coords)
    if len(coords) < 2:
        return

    for i in range(len(coords) - 1):
        # GeoJSON is (Lon, Lat)
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]

        # Calculate H3 index for the start of the segment
        # get_h3_index expects (lat, lng)
        h3_index = get_h3_index(lat1, lon1, resolution=9)

        # Calculate bearing
        # derive_bearing expects (lat1, lon1, lat2, lon2)
        bearing = derive_bearing(lat1, lon1, lat2, lon2)

        if bearing != -1:
            if h3_index not in static_map:
                static_map[h3_index] = []
            static_map[h3_index].append(bearing)


def generate_static_map():
    print(f"Loading {INPUT_FILE}...")
    try:
        # Load GeoJSON
        gdf = gpd.read_file(os.path.join(project_root, INPUT_FILE))
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        return

    # Filter for active lanes
    if "Attivazione" in gdf.columns:
        print("Filtering for active lanes (Attivazione == 1)...")
        # Ensure Attivazione is treated consistently (string or int)
        # Assuming '1' or 1.
        gdf = gdf[gdf["Attivazione"].astype(str) == "1"]
    else:
        print("Warning: 'Attivazione' column not found. Processing all rows.")

    static_map = {}  # Dict[str, List[float]]

    print(f"Processing {len(gdf)} geometries...")
    for geometry in gdf.geometry:
        process_geometry(geometry, static_map)

    print(f"Generated map with {len(static_map)} H3 indices.")

    # Save to pickle
    print(f"Saving to {OUTPUT_FILE}...")
    save_pickle(static_map, os.path.join(project_root, OUTPUT_FILE))
    print("Done.")


if __name__ == "__main__":
    generate_static_map()
