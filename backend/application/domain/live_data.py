"""Live GTFS-RT domain objects.

``Vehicle`` is static fleet identity and lives in ``static_data``. ``LiveTrip``
is the runtime aggregate root: it owns the moving state and the measurement
list for the trip currently present in GTFS-RT.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from typing import TYPE_CHECKING

from .spatial_utils import derive_bearing, derive_speed
from .time_utils import (
    get_seconds_since_midnight,
    get_time_sin_cos,
    to_readable_time,
    to_unix_time,
)

if TYPE_CHECKING:
    from .static_data import Trip, Vehicle
    from .weather import Weather


@dataclass
class LiveFeedRecord:
    """Normalized GTFS-RT input consumed by Observatory."""

    vehicle_id: str
    trip_id: str
    latitude: float
    longitude: float
    timestamp: float
    vehicle_label: str | None = None
    route_id: str | None = None
    direction_id: int | None = None
    bearing: float = 0.0
    speed: float = 0.0
    scheduled_start_time: str | None = None
    start_date: str | None = None
    current_stop_sequence: int | None = None
    current_status: int | None = None
    stop_id: str | None = None
    occupancy_status: int | None = None
    stop_updates: list[dict[str, Any]] = field(default_factory=list)


class GPSData:
    """One GPS ping from GTFS-RT vehicle positions."""

    def __init__(
        self,
        id: str,
        trip: "Trip",
        timestamp,
        latitude: float,
        longitude: float,
        speed: float,
        heading,
        next_stop_id=None,
        current_stop_sequence=None,
        current_status=None,
    ):
        """Store raw GPS/update fields."""
        self.id = id
        self.trip = trip
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.speed = speed
        self.heading = heading
        self.next_stop_id = next_stop_id
        self.current_stop_sequence = current_stop_sequence
        self.current_status = current_status

    def get_longitude(self):
        """Return the longitude."""
        return self.longitude

    def get_latitude(self):
        """Return the latitude."""
        return self.latitude

    def log_data(self):
        """Log this GPS ping."""
        logging.info("GPS Data ID: " + str(self.id))
        logging.info("Trip ID: " + str(self.trip.id))
        logging.info("Timestamp: " + to_readable_time(self.timestamp))
        logging.info("Latitude: " + str(self.latitude))
        logging.info("Longitude: " + str(self.longitude))
        logging.info("Speed: " + str(self.speed))
        logging.info("Heading: " + str(self.heading))


class Measurement:
    """A single observed point for a ``LiveTrip``."""

    def __init__(
        self,
        id,
        vehicle_id,
        next_stop,
        next_stop_distance,
        gpsdata: GPSData,
        trip_id,
        weather: "Weather",
        occupancy_status,
        speed_ratio,
        current_speed,
        derived_speed,
        derived_bearing,
        is_in_preferential,
        hexagon_id: str = None,
        traffic_data_pending: bool = False,
        schedule_adherence: float = 0.0,
        bus_type: int = 0,
        door_number: int = 0,
        deposits: list[str] = None,
        scheduled_start_time: str = None,
        measurement_time: float = None,
        delay_genuine: int = 0,
    ):
        """Store the measurement and all raw fields required by preprocessing."""
        self.id = id
        self.vehicle_id = vehicle_id
        self.next_stop = next_stop
        self.next_stop_distance = next_stop_distance
        self.gpsdata = gpsdata
        self.trip_id = trip_id
        self.measurement_time = (
            measurement_time if measurement_time is not None else to_unix_time(time.time())
        )
        self.weather = weather
        self.occupancy_status = occupancy_status
        self.speed_ratio = speed_ratio
        self.current_speed = current_speed
        self.derived_speed = derived_speed
        self.derived_bearing = derived_bearing
        self.is_in_preferential = is_in_preferential
        self.hexagon_id = hexagon_id
        self.traffic_data_pending = traffic_data_pending
        self.schedule_adherence = schedule_adherence
        self.bus_type = bus_type
        self.door_number = door_number
        self.deposits = deposits or []
        self.scheduled_start_time = scheduled_start_time
        self.starting_time_cos: float = 0.0
        self.starting_time_sin: float = 0.0
        self.delay_genuine = delay_genuine

    def update_traffic_data(self, speed_ratio: float, current_speed: float):
        """Update traffic fields after fresh traffic data is fetched."""
        self.speed_ratio = speed_ratio
        self.current_speed = current_speed
        self.traffic_data_pending = False

    def to_dict(self, trip_id):
        """Return a parquet-friendly representation."""
        return {
            "vehicle_id": str(self.vehicle_id),
            "trip_id": str(trip_id),
            "stop_id": str(self.id),
            "next_stop": str(self.next_stop) if self.next_stop else None,
            "next_stop_distance": self.next_stop_distance,
            "lat": self.gpsdata.latitude,
            "lon": self.gpsdata.longitude,
            "speed": self.gpsdata.speed,
            "bearing": self.gpsdata.heading,
            "gps_timestamp": self.gpsdata.timestamp,
            "measurement_time": self.measurement_time,
            "formatted_time": to_readable_time(self.measurement_time),
            "occupancy": self.occupancy_status,
            "weather_code": self.weather.weather_code if self.weather else None,
            "precip_intensity": self.weather.precip_intensity if self.weather else None,
            "temperature": self.weather.temperature if self.weather else None,
            "apparent_temperature": self.weather.apparent_temperature
            if self.weather
            else None,
            "humidity": self.weather.humidity if self.weather else None,
            "wind_speed": self.weather.wind_speed if self.weather else None,
            "speed_ratio": self.speed_ratio,
            "traffic_speed": self.current_speed,
            "derived_speed": self.derived_speed,
            "derived_bearing": self.derived_bearing,
            "is_in_preferential": self.is_in_preferential,
            "hexagon_id": self.hexagon_id,
            "schedule_adherence": self.schedule_adherence,
            "bus_type": self.bus_type,
            "door_number": self.door_number,
            "deposits": self.deposits,
        }

    def __repr__(self):
        """Return a developer-friendly representation."""
        readable_meas = to_readable_time(self.measurement_time)
        occ_map = {
            0: "EMPTY",
            1: "MANY_SEATS",
            2: "FEW_SEATS",
            3: "STANDING_ONLY",
            4: "CRUSHED",
            5: "FULL",
            6: "NOT_ACCEPTING",
        }
        occ_str = occ_map.get(self.occupancy_status, "UNKNOWN")
        dist_str = (
            f"{self.next_stop_distance:.1f}m"
            if self.next_stop_distance is not None
            else "N/A"
        )
        return (
            f"\n{self.id} | {self.trip_id} | {self.next_stop} ({dist_str}) | "
            f"{self.gpsdata.latitude} | {self.gpsdata.longitude} | {occ_str} | "
            f"Taken at: {readable_meas} | Delay: {self.schedule_adherence:.1f}s | "
            f"Type: {self.bus_type}"
        )

    def __str__(self):
        """Return a human-readable representation."""
        return self.__repr__()


class Update:
    """Combined GTFS-RT vehicle position and trip-update stop information."""

    def __init__(self, live_trip: "LiveTrip", next_stops):
        """Bind the update to a live trip and index stop updates."""
        self.live_trip = live_trip
        self.next_stops = list(next_stops or [])
        self.rt_updates = {}
        for update in self.next_stops:
            seq = update.get("stop_sequence")
            if seq is not None:
                self.rt_updates[int(seq)] = update

        self.has_stop_data = (
            self.live_trip.gps_data is not None
            and self.live_trip.gps_data.current_stop_sequence is not None
            and bool(self.live_trip.trip.stop_times)
        )
        self.upcoming_stops = self._calculate_upcoming_stops()

    def _calculate_upcoming_stops(self):
        """Calculate formatted upcoming stops for the current live position."""
        if not self.has_stop_data:
            return []

        current_seq = self.live_trip.gps_data.current_stop_sequence
        upcoming = [
            st for st in self.live_trip.trip.stop_times
            if int(st["stop_sequence"]) >= current_seq
        ]
        return [self._format_stop_info(st) for st in upcoming]

    def get_next_stop(self):
        """Return the immediate next stop if known."""
        if self.upcoming_stops:
            return self.upcoming_stops[0]
        return None

    def get_delay(self) -> float | None:
        """Return real-time delay for the immediate next stop if available."""
        if not self.upcoming_stops:
            return None
        return self.upcoming_stops[0].get("delay")

    def _format_stop_info(self, st):
        """Format static stop-time information with optional real-time delay."""
        time_str = st["arrival_time"]
        stop_name = st.get("stop_name", "Unknown Stop")
        seq = int(st["stop_sequence"])

        try:
            h, m, s = map(int, time_str.split(":"))
            if h >= 24:
                h = h % 24
            formatted_time = f"{h:02d}:{m:02d}:{s:02d}"
        except Exception:
            formatted_time = time_str

        delay = self.rt_updates.get(seq, {}).get("delay")
        return {
            "stop_id": st["stop_id"],
            "stop_name": stop_name,
            "stop_sequence": st["stop_sequence"],
            "formatted_time": formatted_time,
            "delay": delay,
        }

    def find_stop_distance(self, stop_id):
        """Return shape distance for a stop index in the current trip."""
        stop_dist = self.live_trip.trip.stop_times[stop_id]["shape_dist_traveled"]
        return stop_dist

    def get_live_trip(self):
        """Return the target live trip."""
        return self.live_trip

    def log_data(self):
        """Log update data."""
        logging.info("Vehicle: " + str(self.live_trip.vehicle.label))
        logging.info("Route: " + str(self.live_trip.trip.route.id))
        logging.info("Trip ID: " + str(self.live_trip.trip.id))
        self.live_trip.gps_data.log_data()

    def __str__(self):
        """Return a human-readable update summary."""
        output = "LIVE TRIP INFO:\n"
        output += f"Vehicle: {self.live_trip.vehicle.label}\n"
        output += f"Route served: {self.live_trip.trip.route.id}\n"
        output += f"Direction: {self.live_trip.trip.direction_id}\n"
        output += f"Current trip: {self.live_trip.trip.id}\n"
        output += f"Direction Name: {self.live_trip.trip.direction_name}\n"
        if self.live_trip.gps_data:
            gps = self.live_trip.gps_data
            output += (
                f"Current position: Lat {gps.latitude}, Lon {gps.longitude}, "
                f"Speed {gps.speed}, Heading {gps.heading}\n"
            )
            output += f"Next stop: {gps.next_stop_id}\n"
        output += "Stop Sequence:\n"
        if self.has_stop_data:
            for st in self.upcoming_stops:
                output += (
                    f"  - {st['stop_id']} ({st['stop_name']}) "
                    f"(Seq: {st['stop_sequence']}, Time: {st['formatted_time']})\n"
                )
        else:
            output += "  (Data unavailable)\n"
        return output


class LiveTrip:
    """A trip currently present in GTFS-RT and served by one static Vehicle."""

    def __init__(
        self,
        trip: "Trip",
        vehicle: "Vehicle",
        scheduled_start_time: str = None,
    ):
        """Create a live trip aggregate."""
        self.trip = trip
        self.trip_id = trip.id
        self.vehicle = vehicle
        self.vehicle_id = vehicle.id
        self.old_gps_data: GPSData | None = None
        self.gps_data: GPSData | None = None
        self.latest_update: Update | None = None
        self.occupancy_status: int | None = None
        self.last_seen_timestamp = time.time()
        self.hexagon_id: str | None = None
        self.location_name: str | None = None
        self.derived_speed: float = 0
        self.derived_bearing: float = 0
        self.is_in_preferential: bool = False
        self.scheduled_start_time = scheduled_start_time
        self.measurements: list[Measurement] = []
        self.is_finished = False
        self.actual_start_time: float | None = None
        self.starting_time_cos: float = 0.0
        self.starting_time_sin: float = 0.0
        self._actual_start_detected = False

    @property
    def id(self) -> str:
        """Return the moving entity id, keyed by vehicle id in the live feed."""
        return self.vehicle_id

    @property
    def label(self) -> str:
        """Return the serving vehicle label."""
        return self.vehicle.label

    @property
    def vehicle_type(self):
        """Return the serving vehicle type."""
        return self.vehicle.vehicle_type

    def get_id(self) -> str:
        """Return the vehicle id for this live trip."""
        return self.vehicle_id

    def get_trip(self) -> "Trip":
        """Return the static GTFS trip."""
        return self.trip

    def set_gps_data(self, gps_data: GPSData):
        """Update current GPS state."""
        self.old_gps_data = self.gps_data
        self.gps_data = gps_data
        self.last_seen_timestamp = time.time()

    def get_gps_data(self) -> GPSData | None:
        """Return current GPS state."""
        return self.gps_data

    def set_location_name(self, location_name: str):
        """Set resolved street/location name."""
        self.location_name = location_name

    def get_location_name(self) -> str | None:
        """Return resolved street/location name."""
        return self.location_name

    def set_hexagon_id(self, hexagon_id: str):
        """Set current H3 cell."""
        self.hexagon_id = hexagon_id

    def get_hexagon_id(self) -> str | None:
        """Return current H3 cell."""
        return self.hexagon_id

    def set_occupancy_status(self, occupancy_status: int):
        """Set current crowding status."""
        self.occupancy_status = occupancy_status

    def set_latest_update(self, update: Update):
        """Set latest combined update."""
        self.latest_update = update

    def get_latest_update(self) -> Update | None:
        """Return latest combined update."""
        return self.latest_update

    def derive_speed(self) -> float:
        """Derive movement speed when feed speed is missing."""
        if self.gps_data is None or self.old_gps_data is None:
            self.derived_speed = 0
            return self.derived_speed

        if not self.gps_data.speed:
            derived = derive_speed(
                self.old_gps_data.latitude,
                self.old_gps_data.longitude,
                self.gps_data.latitude,
                self.gps_data.longitude,
                self.old_gps_data.timestamp,
                self.gps_data.timestamp,
            )
            self.derived_speed = derived if derived > 0 else 0
            return self.derived_speed
        self.derived_speed = self.gps_data.speed
        return self.derived_speed

    def derive_bearing(self) -> float:
        """Derive movement bearing from consecutive GPS pings."""
        if self.gps_data is None or self.old_gps_data is None:
            return self.derived_bearing if self.derived_bearing else 0

        derived = derive_bearing(
            self.old_gps_data.latitude,
            self.old_gps_data.longitude,
            self.gps_data.latitude,
            self.gps_data.longitude,
        )
        self.derived_bearing = derived if derived != -1 else self.derived_bearing
        return self.derived_bearing

    def set_is_in_preferential(self, is_in_preferential: bool):
        """Set whether this live trip is aligned with a bus lane."""
        self.is_in_preferential = is_in_preferential

    def get_is_in_preferential(self) -> bool:
        """Return preferential-lane status."""
        return self.is_in_preferential

    def get_bearing(self) -> float:
        """Return derived bearing."""
        return self.derived_bearing

    def get_speed(self) -> float:
        """Return derived speed."""
        return self.derived_speed

    def get_crowding_level(self) -> str:
        """Return readable crowding status."""
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

    def apply_update(
        self,
        update: Update,
        next_stop_distance=None,
        speed_ratio: float = None,
        current_speed: float = None,
        traffic_data_pending: bool = False,
        weather: "Weather" = None,
    ):
        """Apply a combined update and append a measurement."""
        self.latest_update = update
        if not self.gps_data:
            return

        schedule_adherence, delay_genuine = self._calculate_schedule_adherence(update)
        self._detect_actual_departure(next_stop_distance)
        self._append_measurement(
            next_stop_distance=next_stop_distance,
            weather=weather,
            speed_ratio=speed_ratio,
            current_speed=current_speed,
            traffic_data_pending=traffic_data_pending,
            schedule_adherence=schedule_adherence,
            delay_genuine=delay_genuine,
        )

    def _detect_actual_departure(self, next_stop_distance=None):
        """Detect actual departure from first stop using stop sequence and distance."""
        if self._actual_start_detected or not self.gps_data:
            return

        stop_sequence = self.gps_data.current_stop_sequence
        if stop_sequence is not None and stop_sequence == 1:
            last_meas = self.get_last_measurement()
            prev_dist = last_meas.next_stop_distance if last_meas else None
            if prev_dist is not None and next_stop_distance is not None:
                distance_change = next_stop_distance - prev_dist
                if distance_change >= 100:
                    self._mark_actual_departure(to_unix_time(time.time()))
        elif stop_sequence is not None and stop_sequence > 1:
            self._mark_actual_departure(to_unix_time(self.gps_data.timestamp))

    def _mark_actual_departure(self, actual_start_time: float):
        self.actual_start_time = actual_start_time
        time_encoding = get_time_sin_cos(actual_start_time)
        if time_encoding:
            self.starting_time_sin, self.starting_time_cos = time_encoding
        self._actual_start_detected = True

    def _calculate_schedule_adherence(self, update: Update) -> tuple[float, int]:
        """Return schedule adherence and whether delay came directly from GTFS-RT."""
        real_time_delay = update.get_delay()
        if real_time_delay is not None:
            raw_delay = real_time_delay
            while real_time_delay > 43200:
                real_time_delay -= 86400
            while real_time_delay < -43200:
                real_time_delay += 86400
            if abs(raw_delay) > 3600:
                logging.warning(
                    "Delay correction: raw=%s corrected=%s vehicle=%s trip=%s",
                    raw_delay,
                    real_time_delay,
                    self.vehicle.label,
                    self.trip.id,
                )
            return float(real_time_delay), 1

        try:
            shape = self.trip.get_shape()
            if not shape or not self.gps_data:
                return -1000.0, 0

            dist_travelled = shape.project(
                self.gps_data.latitude,
                self.gps_data.longitude,
            )
            spatial_law = self.trip.get_spatial_law()
            expected_time_seconds = float(spatial_law(dist_travelled))
            actual_time_seconds = get_seconds_since_midnight(
                to_unix_time(self.gps_data.timestamp)
            )

            if expected_time_seconds > 86400 and actual_time_seconds < 43200:
                actual_time_seconds += 86400

            diff = actual_time_seconds - expected_time_seconds
            if diff < -43200:
                actual_time_seconds += 86400
            elif diff > 43200:
                expected_time_seconds += 86400

            schedule_adherence = actual_time_seconds - expected_time_seconds
            if abs(schedule_adherence) > 3600:
                logging.warning(
                    "Spatial-law adherence %.0fs vehicle=%s trip=%s",
                    schedule_adherence,
                    self.vehicle.label,
                    self.trip.id,
                )
            return schedule_adherence, 0
        except Exception:
            return -1000.0, 0

    def _append_measurement(
        self,
        next_stop_distance,
        weather: "Weather",
        speed_ratio: float,
        current_speed: float,
        traffic_data_pending: bool,
        schedule_adherence: float,
        delay_genuine: int,
    ):
        """Append one measurement to this live trip."""
        vt = self.vehicle.vehicle_type
        bus_type = 0
        door_number = 0
        deposits = []
        if vt:
            if getattr(vt, "engine", None) and getattr(vt.engine, "fuel", None):
                fuel_value = getattr(vt.engine.fuel, "value", 0)
                bus_type = fuel_value[0] if isinstance(fuel_value, tuple) else fuel_value
            door_number = getattr(vt, "doors", 0) or 0
            deposits = getattr(vt, "deposits", []) or []

        measurement = Measurement(
            id=len(self.measurements) + 1,
            vehicle_id=self.vehicle.label,
            next_stop=self.gps_data.next_stop_id,
            next_stop_distance=next_stop_distance,
            gpsdata=self.gps_data,
            trip_id=self.trip.id,
            weather=weather,
            occupancy_status=self.occupancy_status,
            speed_ratio=speed_ratio,
            current_speed=current_speed,
            derived_speed=self.derived_speed,
            derived_bearing=self.derived_bearing,
            is_in_preferential=self.is_in_preferential,
            hexagon_id=self.hexagon_id,
            traffic_data_pending=traffic_data_pending,
            schedule_adherence=schedule_adherence,
            bus_type=bus_type,
            door_number=door_number,
            deposits=deposits,
            scheduled_start_time=self.scheduled_start_time,
            delay_genuine=delay_genuine,
        )
        measurement.starting_time_cos = self.starting_time_cos
        measurement.starting_time_sin = self.starting_time_sin
        self.measurements.append(measurement)

    def get_measurements_amount(self) -> int:
        """Return number of measurements."""
        return len(self.measurements)

    def get_measurement(self, index):
        """Return measurement by index."""
        return self.measurements[index]

    def get_last_measurement(self):
        """Return latest measurement."""
        return self.measurements[-1] if self.measurements else None

    def get_pending_traffic_measurements(self) -> list[Measurement]:
        """Return measurements waiting for traffic correction."""
        return [m for m in self.measurements if m.traffic_data_pending]

    def to_dict_list(self):
        """Return all measurements as dictionaries."""
        return [m.to_dict(self.trip_id) for m in self.measurements]

    def finish(self):
        """Mark the live trip as finished and emit the internal event."""
        if self.is_finished:
            return
        self.is_finished = True
        if self.measurements:
            from .internal_events import LiveTripFinishedEvent, domain_events

            vehicle_type_name = "Unknown"
            if self.vehicle.vehicle_type:
                vehicle_type_name = self.vehicle.vehicle_type.name
            route_id = self.trip.route.id if self.trip and self.trip.route else ""
            domain_events.emit(
                LiveTripFinishedEvent(
                    live_trip=self,
                    route_id=route_id,
                    vehicle_type_name=vehicle_type_name,
                )
            )

    def format_rich(
        self,
        stop_name_resolver=None,
        street_name_resolver=None,
    ) -> str:
        """Format measurements with optional stop and street enrichment."""
        if not self.measurements:
            return f"LiveTrip {self.trip_id} (Empty)"

        if not stop_name_resolver and not street_name_resolver:
            return str(self)

        header = f"LiveTrip: {self.trip_id}\n"
        header += (
            f"{'ID':<4} | {'NEXT STOP':<25} | {'LOCATION':<25} | {'LAT':<10} | "
            f"{'LON':<10} | {'SPEED':<6} | {'OCCUPANCY':<12} | {'TIME'}\n"
        )
        header += "-" * 130

        lines = []
        for m in self.measurements:
            lat = m.gpsdata.latitude
            lon = m.gpsdata.longitude
            speed = m.gpsdata.speed if m.gpsdata.speed else 0.0

            stop_name = m.next_stop
            if stop_name_resolver and m.next_stop:
                resolved = stop_name_resolver(m.next_stop)
                if resolved:
                    stop_name = resolved[:23] + ".." if len(resolved) > 25 else resolved

            street = "N/A"
            if street_name_resolver:
                resolved = street_name_resolver(lat, lon)
                if resolved:
                    street = resolved[:23] + ".." if len(resolved) > 25 else resolved

            occ_map = {
                0: "EMPTY",
                1: "MANY_SEATS",
                2: "FEW_SEATS",
                3: "STANDING",
                4: "CRUSHED",
                5: "FULL",
                6: "NOT_ACCEPT",
            }
            occ_str = occ_map.get(m.occupancy_status, "UNKNOWN")
            time_str = to_readable_time(m.measurement_time)
            lines.append(
                f"\n{m.id:<4} | {str(stop_name):<25} | {street:<25} | "
                f"{lat:<10.5f} | {lon:<10.5f} | {speed:<6.1f} | "
                f"{occ_str:<12} | {time_str}"
            )

        return header + "".join(lines)

    def __str__(self):
        """Return compact measurement list."""
        if not self.measurements:
            return f"LiveTrip {self.trip_id} (Empty)"
        header = f"LiveTrip: {self.trip_id}\n"
        header += "MEASUREMENT ID | TRIP ID | NEXT STOP | LAT | LON | OCCUPANCY | MEASUREMENT TIME\n"
        header += "-" * 75
        return header + "".join([str(m) for m in self.measurements])


class Schedule:
    """Route/direction/date schedule index."""

    def __init__(self):
        """Initialize the index."""
        self.index = {}

    def load(self, trips_map):
        """Load trips into the schedule index."""
        self._build_index(trips_map)

    def _build_index(self, trips_map):
        """Build route -> direction -> date -> start timestamp index."""
        for trip in trips_map.values():
            r_id = trip.route.id
            d_id = trip.direction_id

            if trip.stop_times:
                start_time = trip.stop_times[0]["arrival_time"]
                for date in trip.dates:
                    self.index.setdefault(r_id, {}).setdefault(d_id, {}).setdefault(
                        date, []
                    )

                    timestamp = to_unix_time(start_time, date)
                    if timestamp:
                        self.index[r_id][d_id][date].append(timestamp)

        for routes in self.index.values():
            for directions in routes.values():
                for date_list in directions.values():
                    date_list.sort()

    def get(self, route_id, direction_id, date):
        """Return scheduled start timestamps for route/direction/date."""
        return self.index.get(route_id, {}).get(direction_id, {}).get(date, [])
