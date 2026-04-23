"""
LedgerBuilder - Constructs the ledger from GTFS CSV files.
"""

import os
import pandas as pd

from .static_data import Route, Trip, Shape
from .live_data import Schedule
from .ledgers import TopologyLedger, ScheduleLedger


class LedgerBuilder:
    """Builds TopologyLedger and ScheduleLedger from GTFS DataFrames."""

    def __init__(self):
        """Initialize with empty DataFrame slots for each GTFS file."""
        # Raw DataFrames
        self.stops = None
        self.routes = None
        self.trips = None
        self.shapes = None
        self.stop_times = None
        self.calendar_dates = None

    def read_csvs(self) -> bool:
        """Read GTFS CSV files into DataFrames. Returns True if successful."""
        print("Reading static data (CSVs)...")

        def read_csv_safe(filename):
            """Read a GTFS CSV as strings, or return None if the file is missing."""
            if os.path.exists(filename):
                return pd.read_csv(
                    filename, sep=",", header=0, dtype=str, low_memory=False
                )
            return None

        self.stops = read_csv_safe("stops.txt")
        self.routes = read_csv_safe("routes.txt")
        self.trips = read_csv_safe("trips.txt")
        self.shapes = read_csv_safe("shapes.txt")
        self.stop_times = read_csv_safe("stop_times.txt")
        self.calendar_dates = read_csv_safe("calendar_dates.txt")

        return self.trips is not None

    def build_topology(self) -> TopologyLedger:
        """Build the topology ledger from loaded DataFrames."""
        print("Building Topology Ledger...")

        routes = self._build_routes()
        stops = (
            self.stops.set_index("stop_id").to_dict("index")
            if self.stops is not None
            else {}
        )
        shapes = self._build_shapes()
        trip_schedules = self._build_trip_schedules()

        # _build_trips needs routes, shapes, trip_schedules via an internal dict
        _internal = {
            "routes": routes,
            "shapes": shapes,
            "trip_schedules": trip_schedules,
        }
        trips = self._build_trips(_internal)

        return TopologyLedger(
            routes=routes,
            stops=stops,
            shapes=shapes,
            trips=trips,
        )

    def build_schedule(self, topology: TopologyLedger) -> ScheduleLedger:
        """Build the schedule ledger from the topology."""
        print("Building Schedule Ledger...")
        schedule = Schedule()
        schedule.load(topology.trips)
        return ScheduleLedger(schedule=schedule)

    # ------ private helpers (unchanged) ------

    def _build_routes(self) -> dict[str, Route]:
        """Build route map from routes DataFrame."""
        route_map: dict[str, Route] = {}
        if self.routes is not None:
            for _, row in self.routes.iterrows():
                route_map[row["route_id"]] = Route(id=row["route_id"])
        return route_map

    def _build_shapes(self) -> dict:
        """Build shape map from shapes DataFrame."""
        print("Processing Shapes...")
        if self.shapes is None:
            return {}

        shape_map: dict[str, Shape] = {}
        grouped = self.shapes.sort_values(["shape_id", "shape_pt_sequence"]).groupby(
            "shape_id"
        )

        count = 0
        total = len(grouped)

        for shape_id, group in grouped:
            points = []
            for _, row in group.iterrows():
                points.append(
                    {
                        "lat": float(row["shape_pt_lat"]),
                        "lon": float(row["shape_pt_lon"]),
                        "dist": float(row["shape_dist_traveled"]),
                    }
                )
            shape_map[shape_id] = Shape(shape_id, points)
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count}/{total} shapes...")

        return shape_map

    def _build_trip_schedules(self):
        """Build grouped stop times for efficient lookup."""
        if self.stop_times is None or self.stops is None:
            return None

        # Merge with stops for name lookup
        merged = pd.merge(self.stop_times, self.stops, on="stop_id", how="left")
        return merged.groupby("trip_id")

    def _build_trips(self, ledger: dict) -> dict[str, Trip]:
        """Build trip map from trips DataFrame."""
        if self.trips is None:
            return {}

        # Pre-compute service_id to dates mapping
        service_to_dates = {}
        if self.calendar_dates is not None:
            added = self.calendar_dates[self.calendar_dates["exception_type"] == "1"]
            service_to_dates = added.groupby("service_id")["date"].apply(list).to_dict()

        st_grouped = ledger.get("trip_schedules")
        trip_map = {}

        count = 0
        total = len(self.trips)

        for _, row in self.trips.iterrows():
            t_id = row["trip_id"]
            r_id = row["route_id"]
            s_id = row["service_id"]

            # Get or create Route
            route_obj = ledger["routes"].get(r_id)
            if not route_obj:
                route_obj = Route(id=r_id)
                ledger["routes"][r_id] = route_obj

            # Get dates and shape
            trip_dates = service_to_dates.get(s_id, [])
            shape = ledger["shapes"].get(row["shape_id"])

            trip = Trip(
                t_id,
                route_obj,
                trip_dates,
                row["direction_id"],
                shape,
                trip_headsign=row.get("trip_headsign"),
            )

            # Attach stop times
            if st_grouped is not None and t_id in st_grouped.groups:
                st_df = st_grouped.get_group(t_id)
                trip.set_stop_times(st_df.to_dict("records"))
            else:
                trip.set_stop_times([])

            trip_map[t_id] = trip
            count += 1
            if count % 5000 == 0:
                print(f"Processed {count}/{total} trips...")

        return trip_map
