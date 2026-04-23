"""GTFS static-data domain objects: fleet enums, VehicleType, Route/Trip/Stop/Shape."""

import numpy as np
import scipy as sp
from scipy.spatial import KDTree
from enum import Enum
from .time_utils import to_unix_time

class FuelType(Enum):
    """Enum of propulsion types used by ATAC fleet."""
    DIESEL = 0
    ELECTRIC_NMC = 1
    CNG = 2
    HYBRID_DIESEL_ELECTRIC = 3
    HYBRID_LTO = 4
    ELECTRIC_NINACL = 5
    ELECTRIC_CATL_LFP = 6
    ELECTRIC_LFP = 7
    ELECTRIC = 8
    DUAL_DIESEL_ELECTRIC = 9

class EuroType(Enum):
    """Euro emission class + ZEV marker."""
    EURO_0 = 0
    EURO_1 = 1
    EURO_2 = 2
    EURO_3 = 3
    EURO_4 = 4
    EURO_5 = 5
    EURO_6 = 6
    EURO_7 = 7
    EURO_8 = 8
    EURO_9 = 9
    ZEV = 10

class Engine:
    """Descriptor for an engine variant (name + fuel + emission class)."""
    def __init__(self,
                 name: str,
                 fuel: FuelType,
                 euro: EuroType):
        """Store engine name, fuel type, and Euro classification."""
        self.name = name
        self.fuel = fuel
        self.euro = euro
    
class VehicleType:
    """Descriptor for a fleet vehicle model (ID ranges, engine, physical dims, capacity)."""
    def __init__(self,
                 name: str,
                 ids: list[list[int]], # Each list is an interval
                 amount: int,
                 active: int, # amount of active vehicles of this model
                 agency: str,
                 deposits: list[str],
                 doors: int, # amount of doors
                 engine: Engine,
                 length: float = None,
                 width: float = None,
                 height: float = None,
                 weight: float = None,
                 capacity_sitting: int = 0,
                 capacity_standing: int = 0,
                 capacity_total: int = 0,
                 construction_year: int = 0,
                 constructors: list[str] = None):
        """Store the full vehicle-type profile and derive total capacity when not supplied."""
        self.name = name
        self.ids = ids
        self.amount = amount
        self.active = active
        self.deposits = deposits
        self.doors = doors
        self.engine = engine
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.capacity_sitting = capacity_sitting
        self.capacity_standing = capacity_standing
        self.capacity_total = (self.capacity_sitting + self.capacity_standing) if capacity_total == 0 else capacity_total
        self.construction_year = construction_year
        self.constructors = constructors        

class Route:
    """GTFS route: id, agency, direction, optional shape and trip list."""
    def __init__(self, id, agency=None, direction=None, shape=None, trips=None):
        """Store the route identity and its optional shape/trips."""
        self.id = id
        self.agency = agency
        self.direction = direction
        self.shape = shape
        self.trips = trips

    @property
    def get_trips(self):  # Returns the list of trips of the route
        """Return the list of trips attached to this route."""
        return self.trips


class Trip:
    """GTFS trip instance: stop times + interpolated time/spatial laws (lazy-built)."""
    def __init__(
        self,
        id,
        route,
        dates,
        direction_id=None,
        shape=None,
        stop_times=None,
        trip_headsign=None,
    ):
        """Store trip metadata; time/spatial laws are built lazily on first access."""
        self.id = id
        self.route = route
        self.direction_id = direction_id
        self.shape = shape  # The argument shape_id actually receives a Shape object now
        self.stop_times = stop_times
        self.direction_name = trip_headsign
        self.dates = dates
        self.time_law = None
        self.spatial_law = None

    def set_stop_times(self, stop_times):
        """Replace the list of stop times for this trip."""
        self.stop_times = stop_times

    def get_stop_times(self):
        """Return the list of stop-time records."""
        return self.stop_times

    def _build_laws(self):
        """Build the time↔distance interpolators (``time_law``/``spatial_law``), monotonising midnight-crossing trips."""
        stop_times = []
        distances = []
        for row in self.stop_times:
            # Parse time string directly to preserve >24h values for night buses.
            # Using get_seconds_since_midnight wraps to 0-86399, breaking
            # interpolation for midnight-crossing trips.
            h, m, s = map(int, row["arrival_time"].split(":"))
            seconds = h * 3600 + m * 60 + s
            stop_times.append(seconds)
            distances.append(float(row["shape_dist_traveled"]))

        # Ensure monotonicity for midnight-crossing trips where times
        # wrap around (e.g., "23:30:00" → "00:05:00" should become 86700).
        for i in range(1, len(stop_times)):
            while stop_times[i] < stop_times[i - 1] - 43200:
                stop_times[i] += 86400

        self.time_law = sp.interpolate.interp1d(stop_times, distances, kind="linear")
        self.spatial_law = sp.interpolate.interp1d(distances, stop_times, kind="linear")

    def get_time_law(self):
        """Return the time→distance interpolator, building it lazily on first call."""
        if self.time_law is None:
            self._build_laws()
        return self.time_law

    def get_spatial_law(self):
        """Return the distance→time interpolator, building it lazily on first call."""
        if self.spatial_law is None:
            self._build_laws()
        return self.spatial_law

    def get_shape(self):
        """Return the Shape object associated with this trip."""
        return self.shape


class Stop:
    """GTFS stop: identifier, human name, and geographic coordinates."""
    def __init__(self, id: int, name: str, latitude: float, longitude: float):
        """Store the stop identity and its lat/lon coordinates."""
        self.id = id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude


class Shape:
    """GTFS shape with a KDTree index and flat-earth projection optimised for Rome."""
    def __init__(self, id: int, points: list[dict[str, float]]):
        """Store polyline points, build a KDTree, and precompute the Rome longitude-scale factor."""
        self.id = id
        # points is a list of {'lat', 'lon', 'dist'}
        # Extract coordinates for KDTree (Lat, Lon)
        self.coords = np.array([[p["lat"], p["lon"]] for p in points])
        self.distances = np.array([p["dist"] for p in points])

        # Build optimized spatial index
        self.tree = KDTree(self.coords)

        # Rome Latitude for Flat Earth Projection (approx 41.9)
        # We pre-calculate scaling factor
        mean_lat = np.radians(41.9)
        self.lon_scale = np.cos(mean_lat)

    def project(self, lat: float, lon: float) -> float:
        """
        Projects a (lat, lon) point onto the shape and returns the distance traveled along the shape (meters).
        Uses KDTree for O(logN) lookup + Vector Projection on nearest segments.
        """
        # 1. Find nearest vertex (Coarse Search)
        _, idx = self.tree.query([lat, lon])  # Returns distance, index

        # 2. Identify Candidate Segments (Previous and Next)
        # We need to check segment (idx-1 -> idx) and (idx -> idx+1)
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, idx))
        if idx < len(self.coords) - 1:
            candidates.append((idx, idx + 1))

        # If no candidates (single point shape?), return point distance
        if not candidates:
            return self.distances[idx]

        best_dist = self.distances[idx]
        min_perp_dist = float("inf")

        # 3. Project onto segments
        P = np.array([lon * self.lon_scale, lat])  # Note: XY order -> Lon, Lat

        for i_start, i_end in candidates:
            # Get Segment Endpoints in Flat Space
            A_lat, A_lon = self.coords[i_start]
            B_lat, B_lon = self.coords[i_end]

            A = np.array([A_lon * self.lon_scale, A_lat])
            B = np.array([B_lon * self.lon_scale, B_lat])

            # Vector A->B and A->P
            AB = B - A
            AP = P - A

            # Length squared of segment
            len_sq = np.dot(AB, AB)

            if len_sq == 0:
                continue  # Zero length segment

            # Project AP onto AB (Scalar projection factor t)
            t = np.dot(AP, AB) / len_sq

            # Clamp t to segment [0, 1]
            t_clamped = max(0, min(1, t))

            # Compute 'Shadow' point on line
            Shadow = A + t_clamped * AB

            # Distance from Point to Shadow (Perpendicular distance)
            perp_vec = P - Shadow
            perp_dist = np.dot(perp_vec, perp_vec)

            # If this is the closest segment so far, update result
            if perp_dist < min_perp_dist:
                min_perp_dist = perp_dist

                # Interpolate shape distance
                d_start = self.distances[i_start]
                d_end = self.distances[i_end]
                best_dist = d_start + t_clamped * (d_end - d_start)

        return float(best_dist)
