from .live_data import Update
from .weather import Weather
from .internal_events import domain_events, DIARY_FINISHED
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
    from .virtual_entities import Observatory
    from application.domain import Autobus, GPSData


from .time_utils import to_unix_time, get_seconds_since_midnight, get_time_sin_cos


class Measurement:
    """A measurement is a record of a bus's location and status at a specific time."""

    def __init__(
        self,
        id,
        autobus_id,
        next_stop,
        next_stop_distance,
        gpsdata: "GPSData",
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
        self.id = id
        self.autobus_id = autobus_id
        self.next_stop = next_stop  # This is the next stop in the trip, by stop_id
        self.next_stop_distance = next_stop_distance
        self.gpsdata = gpsdata
        self.trip_id = trip_id
        self.measurement_time = measurement_time if measurement_time is not None else to_unix_time(time.time())
        self.weather = weather
        self.occupancy_status = occupancy_status
        self.speed_ratio = speed_ratio
        self.current_speed = current_speed
        self.derived_speed = derived_speed
        self.derived_bearing = derived_bearing
        self.is_in_preferential = is_in_preferential
        self.hexagon_id = hexagon_id
        self.traffic_data_pending = traffic_data_pending  # True if correction needed
        self.schedule_adherence = schedule_adherence
        self.bus_type = bus_type
        self.door_number = door_number
        self.deposits = deposits or []
        self.scheduled_start_time = scheduled_start_time
        self.starting_time_cos: float = 0.0
        self.starting_time_sin: float = 0.0
        self.delay_genuine = delay_genuine

    def update_traffic_data(self, speed_ratio: float, current_speed: float):
        """Update ONLY traffic fields after fresh data is fetched."""
        self.speed_ratio = speed_ratio
        self.current_speed = current_speed
        self.traffic_data_pending = False

    def to_dict(self, trip_id):
        from .time_utils import to_readable_time

        # Flatten GPS Data for Parquet
        return {
            "autobus_id": str(self.autobus_id),
            "trip_id": str(trip_id),
            "stop_id": str(self.id),  # Parquet Schema mismatch fix (Bytes vs Int)
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
            # Weather Data
            "weather_code": self.weather.weather_code if self.weather else None,
            "precip_intensity": self.weather.precip_intensity if self.weather else None,
            "temperature": self.weather.temperature if self.weather else None,
            "apparent_temperature": self.weather.apparent_temperature
            if self.weather
            else None,
            "humidity": self.weather.humidity if self.weather else None,
            "wind_speed": self.weather.wind_speed if self.weather else None,
            # Traffic and Derived Data
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
        from .time_utils import to_readable_time

        readable_meas = to_readable_time(self.measurement_time)
        lat = self.gpsdata.latitude
        lon = self.gpsdata.longitude

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
        return f"\n{self.id} | {self.trip_id} | {self.next_stop} ({dist_str}) | {lat} | {lon} | {occ_str} | Taken at: {readable_meas} | Delay: {self.schedule_adherence:.1f}s | Type: {self.bus_type}"

    def __str__(self):
        return self.__repr__()


class Diary:
    def __init__(self, observer, trip_id, scheduled_start_time: str = None):
        self.observer = observer
        self.trip_id = trip_id  # Store the specific Trip ID this diary belongs to
        self.scheduled_start_time = scheduled_start_time  # Format: "HH:MM:SS"
        self.measurements = []
        self.is_finished = False

        # Actual departure detection
        self.actual_start_time: float = (
            None  # Unix timestamp when bus actually departed
        )
        self.starting_time_cos: float = 0.0
        self.starting_time_sin: float = 0.0
        self._actual_start_detected: bool = False

    def add_measurement(self, measurement):
        self.measurements.append(measurement)
        print(f"Added measurement to diary: {measurement}")

    def get_measurement(self, index):
        return self.measurements[index]

    def get_last_measurement(self):
        return self.measurements[-1] if self.measurements else None

    def get_observer(self):
        return self.observer

    def set_observer(self, observer):
        self.observer = observer

    def get_measurements_amount(self):
        return len(self.measurements)

    def get_pending_traffic_measurements(self) -> list:
        """Get measurements waiting for traffic data correction."""
        return [m for m in self.measurements if m.traffic_data_pending]

    def to_dict_list(self):
        # Exports measurements using the stored Trip Context
        return [m.to_dict(self.trip_id) for m in self.measurements]

    def __str__(self):
        if not self.measurements:
            return f"Diary for Trip {self.trip_id} (Empty)"

        header = f"Diary for Trip: {self.trip_id}\n"
        header += "MEASUREMENT ID | TRIP ID | NEXT STOP | LAT | LON | OCCUPANCY | MEASUREMENT TIME\n"
        header += "-" * 75
        return header + "".join([str(m) for m in self.measurements])

    def format_rich(
        self,
        stop_name_resolver=None,
        street_name_resolver=None,
    ) -> str:
        """
        Format diary with enriched data (stop names, street locations).

        Args:
            stop_name_resolver: fn(stop_id) -> str | None
            street_name_resolver: fn(lat, lon) -> str | None

        Returns enriched string. Falls back to __str__ if no resolvers provided.
        """
        if not self.measurements:
            return f"Diary for Trip {self.trip_id} (Empty)"

        if not stop_name_resolver and not street_name_resolver:
            return str(self)

        from .time_utils import to_readable_time

        header = f"Diary for Trip: {self.trip_id}\n"
        header += f"{'ID':<4} | {'NEXT STOP':<25} | {'LOCATION':<25} | {'LAT':<10} | {'LON':<10} | {'SPEED':<6} | {'OCCUPANCY':<12} | {'TIME'}\n"
        header += "-" * 130

        lines = []
        for m in self.measurements:
            lat = m.gpsdata.latitude
            lon = m.gpsdata.longitude
            speed = m.gpsdata.speed if m.gpsdata.speed else 0.0

            # Resolve stop name
            stop_name = m.next_stop
            if stop_name_resolver and m.next_stop:
                resolved = stop_name_resolver(m.next_stop)
                if resolved:
                    stop_name = resolved[:23] + ".." if len(resolved) > 25 else resolved

            # Resolve street name
            street = "N/A"
            if street_name_resolver:
                resolved = street_name_resolver(lat, lon)
                if resolved:
                    street = resolved[:23] + ".." if len(resolved) > 25 else resolved

            # Occupancy
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
                f"\n{m.id:<4} | {str(stop_name):<25} | {street:<25} | {lat:<10.5f} | {lon:<10.5f} | {speed:<6.1f} | {occ_str:<12} | {time_str}"
            )

        return header + "".join(lines)


class Observer:
    """Generic Observer that uses injected Logic strategies."""

    def __init__(
        self, observatory: "Observatory", assignedVehicle: "Autobus", diary: Diary
    ):
        self.assignedVehicle: "Autobus" = assignedVehicle
        self.observatory: "Observatory" = observatory
        self.current_diary: Diary = diary
        self.diary_history: list[Diary] = []

    def archive_current_diary(self):
        if self.current_diary:
            self.current_diary.is_finished = True
            self.diary_history.append(self.current_diary)

            # Get route_id from trip for vectorization
            trip = self.observatory.search_trip(self.current_diary.trip_id)
            route_id = trip.route.id if trip and hasattr(trip, "route") else None

            # Get Vehicle Type Name
            vehicle_type_name = "Unknown"
            if self.assignedVehicle.vehicle_type:
                vehicle_type_name = self.assignedVehicle.vehicle_type.name

            print(
                f"Archived diary for vehicle {self.assignedVehicle.label}: "
                f"Trip {self.current_diary.trip_id}"
            )

            # Emit event for async processing
            if route_id and self.current_diary.measurements:
                domain_events.emit(
                    DIARY_FINISHED,
                    {
                        "diary": self.current_diary,
                        "route_id": route_id,
                        "observatory": self.observatory,
                        "vehicle_type_name": vehicle_type_name,
                    },
                )

            self.current_diary = None

    def start_new_trip(self, new_trip, scheduled_start_time: str = None):
        if self.current_diary:
            self.archive_current_diary()
        self.current_diary = Diary(self, new_trip.id, scheduled_start_time)
        print(
            f"Started new trip on vehicle {self.assignedVehicle.label}: {new_trip.id} - {new_trip.route.id} {new_trip.direction_name}"
        )

    def _record_measurement(
        self,
        gps_data,
        trip_id,
        next_stop_id,
        next_stop_distance,
        weather,
        occupancy_status,
        speed_ratio,
        current_speed,
        derived_speed,
        derived_bearing,
        is_in_preferential,
        hexagon_id: str = None,
        traffic_data_pending: bool = False,
        schedule_adherence: float = 0.0,
        delay_genuine: int = 0,
    ):
        diary = self.current_diary

        # === Actual Departure Detection ===
        stop_sequence = gps_data.current_stop_sequence if gps_data else None
        if not diary._actual_start_detected:
            if stop_sequence is not None and stop_sequence == 1:
                last_meas = diary.get_last_measurement()
                prev_dist = last_meas.next_stop_distance if last_meas else None
                if prev_dist is not None and next_stop_distance is not None:
                    distance_change = next_stop_distance - prev_dist
                    if distance_change >= 100:
                        # Bus has departed from first stop
                        diary.actual_start_time = to_unix_time(time.time())
                        time_encoding = get_time_sin_cos(diary.actual_start_time)
                        if time_encoding:
                            diary.starting_time_sin, diary.starting_time_cos = (
                                time_encoding
                            )
                        diary._actual_start_detected = True
                        print(
                            f"🚌 Detected actual departure for vehicle {self.assignedVehicle.label} at stop_sequence 1"
                        )

        # Extract vehicle info
        bus_type = 0
        door_number = 0
        deposits = []
        if self.assignedVehicle.vehicle_type:
            # Assuming bus_type maps to fuel value or similar
            # Or we can map Engine Name to ID?
            # For now, let's use Fuel Type Value + 1 or similar simple mapping
            # Actually, let's just use 0 if not defined
            # Wait, VehicleType has engine, engine has fuel.
            vt = self.assignedVehicle.vehicle_type
            if hasattr(vt, "engine") and vt.engine:
                if hasattr(vt.engine, "fuel") and hasattr(vt.engine.fuel, "value"):
                    bus_type = (
                        vt.engine.fuel.value[0]
                        if isinstance(vt.engine.fuel.value, tuple)
                        else vt.engine.fuel.value
                    )

            door_number = vt.doors
            deposits = vt.deposits

        # Record
        measurement_id = self.current_diary.get_measurements_amount() + 1

        scheduled_start_time = getattr(self.current_diary, "scheduled_start_time", None)
        starting_time_cos = diary.starting_time_cos
        starting_time_sin = diary.starting_time_sin

        m = Measurement(
            measurement_id,
            self.get_bus().get_label(),
            next_stop_id,
            next_stop_distance,
            gps_data,
            trip_id,
            weather,
            occupancy_status,
            speed_ratio,
            current_speed,
            derived_speed,
            derived_bearing,
            is_in_preferential,
            hexagon_id,
            traffic_data_pending,
            schedule_adherence,
            bus_type,
            door_number,
            deposits,
            scheduled_start_time,
            delay_genuine=delay_genuine,
        )
        m.starting_time_cos = starting_time_cos
        m.starting_time_sin = starting_time_sin

        self.current_diary.add_measurement(m)

        status_map = {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"}
        status_str = status_map.get(
            gps_data.current_status, f"UNKNOWN({gps_data.current_status})"
        )
        print(
            f"A measurement has been recorded in diary for vehicle {self.assignedVehicle.label}: {self.current_diary.trip_id}. Status: {status_str}"
        )

    def get_bus(self):
        return self.assignedVehicle

    def set_bus(self, assignedVehicle: "Autobus"):
        self.assignedVehicle = assignedVehicle

    def updateDiary(
        self,
        update: Update,
        next_stop_distance=None,
        speed_ratio: float = None,
        current_speed: float = None,
        traffic_data_pending: bool = False,
    ):
        if not self.current_diary:
            return

        # REMOVED strict check for has_stop_data to allow recording of raw positions
        # if not update.has_stop_data:
        #    return

        # 2. Record measurement
        bus: "Autobus" = update.get_autobus()
        hexagon_id = getattr(bus, "hexagon_id", None)

        weather = None
        if hexagon_id:
            try:
                weather = self.observatory.get_city("Rome").get_weather(hexagon_id)
            except Exception as e:
                print(f"Warning: Could not get weather for hex {hexagon_id}: {e}")

        # --- SCHEDULE ADHERENCE CALCULATION ---
        schedule_adherence = -1000.0
        delay_genuine = 0
        
        # 1. Try to get real-time delay directly from feed
        real_time_delay = update.get_delay()
        if real_time_delay is not None:
            schedule_adherence = float(real_time_delay)
            delay_genuine = 1
        else:
            # 2. Fallback to Spatial Interpolation
            # print(f"DEBUG: No live data on delay found. Fallback on interpolation...")
            try:
                trip = bus.get_trip()
                if trip and trip.get_shape():
                    shape = trip.get_shape()
                    gps = bus.get_gpsData()

                    # Project current position onto shape
                    dist_travelled = shape.project(gps.latitude, gps.longitude)

                    # Get expected time at this distance
                    spatial_law = trip.get_spatial_law()
                    expected_time_seconds = float(spatial_law(dist_travelled))

                    # Get actual time (from GPS timestamp)
                    # Use timestamp from the update, NOT system time
                    actual_time_seconds = get_seconds_since_midnight(
                        to_unix_time(gps.timestamp)
                    )

                    # --- Midnight Wrap-around Correction ---
                    # Check for massive jumps indicating day boundary crossing
                    diff = actual_time_seconds - expected_time_seconds
                    
                    # Case 1: Actual is early morning (e.g. 00:05 = 300), Expected is late night (e.g. 23:55 = 86100)
                    # Diff would be ~ -85800. We expect diffs within +/- 12 hours.
                    if diff < -43200:
                        actual_time_seconds += 86400
                    
                    # Case 2: Actual is late night, Expected is early morning (Rare in GTFS but possible)
                    elif diff > 43200:
                        expected_time_seconds += 86400

                    schedule_adherence = actual_time_seconds - expected_time_seconds
            except Exception as e:
                # If projection fails or trip data missing, default to -1000.0
                # print(f"Adherence calc failed: {e}")
                pass
        # -------------------------------------

        self._record_measurement(
            update.get_autobus().get_gpsData(),
            update.get_autobus().get_trip().id,
            update.get_autobus().get_gpsData().next_stop_id,
            next_stop_distance,
            weather,
            bus.occupancy_status,
            speed_ratio,
            current_speed,
            bus.derived_speed,
            bus.derived_bearing,
            bus.is_in_preferential,
            hexagon_id,
            traffic_data_pending,
            schedule_adherence,
            delay_genuine,
        )
