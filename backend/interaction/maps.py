import folium as fm


def plot_route(coords):
    """
    Plot the route on a map. Coords is a numpy array. Each element is a of the type [latitude, longitude].
    Returns a folium map
    """
    m = fm.Map(location=[coords[0][0], coords[0][1]], zoom_start=12)
    fm.PolyLine(coords, color="blue").add_to(m)
    return m


def plot_stops(coords):
    """
    Plot the stops on a map. Coords is a numpy array. Each element is a of the type [latitude, longitude].
    Returns a folium map
    """
    m = fm.Map(location=[coords[0][0], coords[0][1]], zoom_start=12)
    for coord in coords:
        fm.Marker(coord).add_to(m)
    return m
