import logging
from .time_utils import to_readable_time
from .spatial_utils import derive_speed, derive_bearing
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observers import Observer
    from application.domain import Trip
    from application.domain import Autobus


class GPSData:
    """Gpsdata."""
    def __init__(
        self,
        id: int,
        trip: str,
        timestamp,
        latitude: float,
        longitude: float,
        speed: float,
        heading: str,
        next_stop_id=None,
        current_stop_sequence=None,
        current_status=None,
    ):
        """Initialize the instance."""
        self.id = id
        self.trip = trip
        self.timestamp = timestamp  # Timestamp of the GPS data
        self.latitude = latitude  # Current latitude
        self.longitude = longitude  # Current longitude
        self.speed = speed  # Current speed - usually it's 0.0.
        self.heading = heading  # Current heading
        self.next_stop_id = next_stop_id  # ID of the next stop
        self.current_stop_sequence = (
            current_stop_sequence  # Sequence value of the current stop
        )
        self.current_status = current_status  # 1 = STOPPED_AT, 2 = IN_TRANSIT_TO

    def get_longitude(self):
        """Return the longitude."""
        return self.longitude

    def get_latitude(self):
        """Return the latitude."""
        return self.latitude

    def log_data(self):
        """Log data."""
        logging.info("GPS Data ID: " + str(self.id))
        logging.info("Trip ID: " + str(self.trip.id))
        logging.info("Timestamp: " + to_readable_time(self.timestamp))
        logging.info("Latitude: " + str(self.latitude))
        logging.info("Longitude: " + str(self.longitude))
        logging.info("Speed: " + str(self.speed))
        logging.info("Heading: " + str(self.heading))


class Update:  # A combination of vehicleUpdate and tripUpdate to have all update info in one object
    """Update."""
    def __init__(self, autobus: "Autobus", next_stops):
        # State Initialization
        """Initialize the instance."""
        self.autobus: "Autobus" = autobus
        
        # Index real-time updates by stop_sequence for fast lookup
        self.rt_updates = {}
        if next_stops:
            for update in next_stops:
                seq = update.get("stop_sequence")
                if seq is not None:
                    self.rt_updates[int(seq)] = update

        self.has_stop_data = (
            self.autobus.GPSData.current_stop_sequence is not None
            and self.autobus.trip.stop_times
        )
        self.upcoming_stops = self._calculate_upcoming_stops()

    def _calculate_upcoming_stops(self):
        """Calculate upcoming stops."""
        if not self.has_stop_data:
            return []

        # Access via Autobus
        current_seq = self.autobus.GPSData.current_stop_sequence
        stop_times = self.autobus.trip.stop_times

        upcoming = [st for st in stop_times if int(st["stop_sequence"]) >= current_seq]

        formatted_stops = []
        for st in upcoming:
            formatted_stops.append(self._format_stop_info(st))

        return formatted_stops

    def get_next_stop(self):
        """Return the next stop."""
        if self.upcoming_stops:
            return self.upcoming_stops[0]
        return None
    
    def get_delay(self) -> float | None:
        """Returns the delay (in seconds) for the immediate next stop, if available."""
        if not self.upcoming_stops:
            return None
        return self.upcoming_stops[0].get("delay")

    def _format_stop_info(self, st):
        # Normalize time (GTFS can be > 24h, e.g. 25:00:00)
        """Format stop info."""
        time_str = st["arrival_time"]
        stop_name = st.get("stop_name", "Unknown Stop")
        seq = int(st["stop_sequence"])
        
        try:
            h, m, s = map(int, time_str.split(":"))
            if h >= 24:
                h = h % 24
            formatted_time = f"{h:02d}:{m:02d}:{s:02d}"
        except Exception:
            formatted_time = time_str  # Fallback if format is weird

        # Look up real-time delay
        delay = None
        if seq in self.rt_updates:
            delay = self.rt_updates[seq].get("delay")

        return {
            "stop_id": st["stop_id"],
            "stop_name": stop_name,
            "stop_sequence": st["stop_sequence"],
            "formatted_time": formatted_time,
            "delay": delay
        }

    def find_stop_distance(self, stop_id):
        """Find stop distance."""
        stop_dist = self.autobus.trip.stop_times[stop_id]["shape_dist_traveled"]
        return stop_dist

    def get_autobus(self):
        """Return the autobus."""
        return self.autobus

    def log_data(self):
        # Update ID removed from object
        """Log data."""
        logging.info("Autobus: " + str(self.autobus.label))
        logging.info("Route: " + str(self.autobus.trip.route.id))
        logging.info("Trip ID: " + str(self.autobus.trip.id))
        self.autobus.GPSData.log_data()

    def __str__(self):
        """Return a human-readable string representation."""
        output = "BUS INFO:\n"
        output += f"Vehicle: {self.autobus.label}\n"

        output += "RIDE INFO (served by this vehicle RIGHT NOW):\n"
        output += f"Route served: {self.autobus.trip.route.id}\n"
        output += f"Direction: {self.autobus.trip.direction_id}\n"
        output += f"Current trip: {self.autobus.trip.id}\n"
        output += f"Direction Name: {self.autobus.trip.direction_name}\n"

        output += f"Current position: Lat {self.autobus.GPSData.latitude}, Lon {self.autobus.GPSData.longitude}, Speed {self.autobus.GPSData.speed}, Heading {self.autobus.GPSData.heading}\n"
        output += f"Next stop: {self.autobus.GPSData.next_stop_id}\n"

        output += "Stop Sequence:\n"
        if self.has_stop_data:
            for st in self.upcoming_stops:
                output += f"  - {st['stop_id']} ({st['stop_name']}) (Seq: {st['stop_sequence']}, Time: {st['formatted_time']})\n"
        else:
            output += "  (Data unavailable)\n"

        return output


class Autobus:
    """Autobus."""
    def __init__(
        self,
        id: int,
        trip: str,
        old_GPSData: GPSData = None,
        GPSData: GPSData = None,
        # latest_update: Update = None,
        occupancy_status: int = None,
        hexagon_id: str = None,
        location_name: str = None,
        vehicle_type=None,
        label: str = None,
    ):
        """Initialize the instance."""
        self.id = id  # Vehicle ID
        self.label = label if label else str(id)
        self.old_GPSData = old_GPSData
        self.GPSData = GPSData
        self.trip = trip
        # self.latest_update = latest_update
        self.occupancy_status = occupancy_status
        self.observer = None
        self.last_seen_timestamp = time.time()
        self.hexagon_id = hexagon_id
        self.location_name = location_name
        self.vehicle_type = vehicle_type
        # === DERIVED MEASUREMENTS === #
        self.derived_speed: float = 0  # km/h
        self.derived_bearing: float = 0  # degrees
        self.is_in_preferential: bool = False

    def get_label(self) -> str:
        """Return the label."""
        return self.label

    def set_vehicle_type(self, vehicle_type):
        """Set the vehicle type."""
        self.vehicle_type = vehicle_type

    def get_vehicle_type(self):
        """Return the vehicle type."""
        return self.vehicle_type

    def set_latest_update(self, update: "Update"):
        """Set the latest update."""
        self.latest_update = update

    def get_latest_update(self) -> "Update":
        """Return the latest update."""
        return self.latest_update

    def set_observer(self, observer: "Observer"):
        """Set the observer."""
        self.observer = observer

    def get_crowding_level(self) -> str:
        """Returns a readable crowding status."""
        # Mapping based on typical GTFS-RT occupancyStatus enums
        # 0: EMPTY, 1: MANY_SEATS_AVAILABLE, 2: FEW_SEATS_AVAILABLE, 3: STANDING_ROOM_ONLY
        # 4: CRUSHED_STANDING_ROOM_ONLY, 5: FULL, 6: NOT_ACCEPTING_PASSENGERS
        mapping = {
            0: "EMPTY",
            1: "MANY_SEATS",
            2: "FEW_SEATS",
            3: "STANDING_ONLY",
            4: "CRUSHED",
            5: "FULL",
            6: "NOT_ACCEPTING",
        }
        return mapping.get(self.occupancy_status, "UNKNOWN")

    def get_observer(self) -> "Observer":
        """Return the observer."""
        return self.observer

    def set_gpsData(self, GPSData: "GPSData"):
        """Set the gpsdata."""
        self.old_GPSData = self.GPSData
        self.GPSData = GPSData
        self.last_seen_timestamp = time.time()

    def get_gpsData(self) -> "GPSData":
        """Return the gpsdata."""
        return self.GPSData

    def set_location_name(self, location_name: str):
        """Set the location name."""
        self.location_name = location_name

    def get_location_name(self) -> str:
        """Return the location name."""
        return self.location_name

    def get_hexagon_id(self) -> str:
        """Return the hexagon id."""
        return self.hexagon_id

    def set_hexagon_id(self, hexagon_id: str):
        """Set the hexagon id."""
        self.hexagon_id = hexagon_id

    def set_trip(self, trip: "Trip"):
        """Set the trip."""
        self.trip = trip

    def get_trip(self) -> "Trip":
        """Return the trip."""
        return self.trip

    def get_id(self) -> int:
        """Return the id."""
        return self.id

    def set_occupancy_status(self, occupancy_status: int):
        """Set the occupancy status."""
        self.occupancy_status = occupancy_status

    def derive_speed(self) -> float:
        """Derive speed."""
        if self.GPSData is None or self.old_GPSData is None:
            self.derived_speed = 0
            return self.derived_speed

        if not self.GPSData.speed:
            derived_speed = derive_speed(
                self.old_GPSData.latitude,
                self.old_GPSData.longitude,
                self.GPSData.latitude,
                self.GPSData.longitude,
                self.old_GPSData.timestamp,
                self.GPSData.timestamp,
            )
            self.derived_speed = derived_speed if derived_speed > 0 else 0
            return self.derived_speed

        return self.GPSData.speed

    def derive_bearing(self) -> float:
        """Derive bearing."""
        if self.GPSData is None or self.old_GPSData is None:
            return self.derived_bearing if self.derived_bearing else 0

        derived_bearing = derive_bearing(
            self.old_GPSData.latitude,
            self.old_GPSData.longitude,
            self.GPSData.latitude,
            self.GPSData.longitude,
        )
        self.derived_bearing = (
            derived_bearing if derived_bearing != -1 else self.derived_bearing
        )
        return self.derived_bearing

    def set_is_in_preferential(self, is_in_preferential: bool):
        """Set the is in preferential."""
        self.is_in_preferential = is_in_preferential

    def get_is_in_preferential(self) -> bool:
        """Return the is in preferential."""
        return self.is_in_preferential

    def get_bearing(self) -> float:
        """Return the bearing."""
        return self.derived_bearing

    def get_speed(self) -> float:
        """Return the speed."""
        return self.derived_speed


class Schedule:
    """Schedule."""
    def __init__(self):
        """Initialize the instance."""
        self.index = {}  # route -> direction -> date -> [Trip]

    def load(self, trips_map):
        """Load."""
        self._build_index(trips_map)

    def _build_index(self, trips_map):
        """Build the index."""
        count = 0
        for trip in trips_map.values():
            r_id = trip.route.id
            d_id = trip.direction_id

            # For each date this trip is active
            if trip.stop_times:
                start_time = trip.stop_times[0]["arrival_time"]
                for date in trip.dates:
                    if r_id not in self.index:
                        self.index[r_id] = {}
                    if d_id not in self.index[r_id]:
                        self.index[r_id][d_id] = {}
                    if date not in self.index[r_id][d_id]:
                        self.index[r_id][d_id][date] = []

                    self.index[r_id][d_id][date].append(start_time)
                    count += 1
        logging.info(f"Index built: {count} entries.")

    def get(self, route_id, direction_id, date_str):
        """
        Returns a list of scheduled START times (Unix timestamps) for a Route+Direction on a specific Date.
        date_str format: "YYYYMMDD"
        """
        from .time_utils import to_unix_time
        from datetime import datetime

        # 1. Retrieve raw time strings
        time_strings = self._get_time_strings(route_id, direction_id, date_str)

        # 2. Parse Reference Date
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
        except Exception:
            return []

        # 3. Convert to Unix
        timestamps = []
        for t_str in time_strings:
            ts = to_unix_time(t_str, date_ref=dt)
            if ts:
                timestamps.append(ts)

        return sorted(timestamps)

    def _get_time_strings(self, route_id, direction_id, date_str):
        """Returns list of time strings (Internal)."""
        if (
            route_id in self.index
            and direction_id in self.index[route_id]
            and date_str in self.index[route_id][direction_id]
        ):
            return self.index[route_id][direction_id][date_str]
        return []
