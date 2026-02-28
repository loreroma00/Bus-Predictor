import uuid
import logging
from datetime import datetime
from application.domain.time_utils import (
    get_timestamp_components,
    get_time_sin_cos,
    get_seconds_since_midnight,
    get_time_sin_cos_from_str,
)
from application.domain.interfaces import Vector, Vectorizer, LabelVector

from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from application.domain import Measurement, Route, Shape, Trip, Weather


class DayType:
    WEEKDAY = 0
    SATURDAY = 1
    SUNDAY = 2


class BusType:
    pass


class PredictionLabel(LabelVector):
    def __init__(
        self,
        id: int,
        time: float,
        time_seconds: int,
        occupancy_status: int,
    ):
        self.id = id
        self.time = time
        self.time_seconds = time_seconds
        self.occupancy_status = occupancy_status


class PredictionVector(Vector):
    def __init__(
        self,
        id: int,
        trip_id: str,
        route_id: int,
        direction_id: int,
        stop_sequence: int,
        bus_type: int,
        door_number: int,
        shape_dist_travelled: float,
        distance_to_next_stop: float,
        day_type: DayType,
        time: float,
        schedule_adherence: float,
        speed_ratio: float,
        current_traffic_speed: float,
        current_speed: float,
        precipitation: float,
        weather_code: int,
        served_ratio: float,
        deposit_grottarossa: int,
        deposit_magliana: int,
        deposit_tor_sapienza: int,
        deposit_portonaccio: int,
        deposit_monte_sacro: int,
        deposit_tor_pagnotta: int,
        deposit_tor_cervara: int,
        deposit_maglianella: int,
        deposit_costi: int,
        deposit_trastevere: int,
        deposit_acilia: int,
        deposit_tor_vergata: int,
        deposit_porta_maggiore: int,
        sch_starting_time_cos: float = 0.0,
        sch_starting_time_sin: float = 0.0,
        starting_time_cos: float = 0.0,
        starting_time_sin: float = 0.0,
        delay_genuine: int = 0,
    ):
        self.id = id
        self.trip_id = trip_id

        ### ======================= ###
        ###        Route Data       ###
        ### ======================= ###

        self.route_id: int = route_id
        self.direction_id: int = direction_id
        self.stop_sequence: int = stop_sequence
        self.bus_type: int = bus_type
        self.door_number: int = door_number
        self.deposit_grottarossa: int = deposit_grottarossa
        self.deposit_magliana: int = deposit_magliana
        self.deposit_tor_sapienza: int = deposit_tor_sapienza
        self.deposit_portonaccio: int = deposit_portonaccio
        self.deposit_monte_sacro: int = deposit_monte_sacro
        self.deposit_tor_pagnotta: int = deposit_tor_pagnotta
        self.deposit_tor_cervara: int = deposit_tor_cervara
        self.deposit_maglianella: int = deposit_maglianella
        self.deposit_costi: int = deposit_costi
        self.deposit_trastevere: int = deposit_trastevere
        self.deposit_acilia: int = deposit_acilia
        self.deposit_tor_vergata: int = deposit_tor_vergata
        self.deposit_porta_maggiore: int = deposit_porta_maggiore
        self.shape_dist_travelled: float = shape_dist_travelled
        self.distance_to_next_stop: float = distance_to_next_stop
        self.far_status: bool = True if distance_to_next_stop > 250 else False

        ### ======================= ###
        ###         Time Data       ###
        ### ======================= ###

        self.day_type: DayType = day_type
        self.rush_hour_status: int = self._is_rush_hour(time)
        self.time_encoding: tuple[float, float] = get_time_sin_cos(time)
        self.sch_starting_time_cos: float = sch_starting_time_cos
        self.sch_starting_time_sin: float = sch_starting_time_sin
        self.starting_time_cos: float = starting_time_cos
        self.starting_time_sin: float = starting_time_sin
        self.delay_genuine: int = delay_genuine

        ### ======================= ###
        ###        Trip Data        ###
        ### ======================= ###

        self.schedule_adherence: float = schedule_adherence

        self.speed_ratio: float = speed_ratio  # TRAFFIC (Context)
        self.current_traffic_speed: float = current_traffic_speed  # TRAFFIC (Context)
        self.current_speed: float = current_speed
        self.precipitation: float = precipitation  # in mm/h
        self.weather_code: int = weather_code

        self.served_ratio = served_ratio

    def _is_rush_hour(self, time: float) -> int:
        _, _, time_str = get_timestamp_components(time)
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        match time_obj.hour:
            case 7 | 8 | 9:
                return 1
            case 17 | 18:
                return 1
            case 19:
                if time_obj.minute > 30:
                    return 0
                return 1
            case _:
                return 0


class PredictionVectorizer(Vectorizer):
    def __init__(
        self,
        measurement: "Measurement",
        route: "Route",
        shape: "Shape",
        trip: "Trip",
        served_ratio: float,
    ):
        self.measurement = measurement
        self.route = route
        self.shape = shape
        self.trip = trip
        self.served_ratio = served_ratio

    def vectorize(self) -> tuple[PredictionVector, PredictionLabel]:
        """
        Transforms a Measurement into a Vector and Label.
        """
        # Generate common ID
        vector_id = str(uuid.uuid4())

        # === Route Data ===
        route_id: int = self.route.id
        direction_id: int = self.trip.direction_id
        stop_sequence: int = self.measurement.gpsdata.current_stop_sequence or 0
        try:
            shape_dist_travelled: float = self.shape.project(
                self.measurement.gpsdata.latitude, self.measurement.gpsdata.longitude
            )
        except Exception as e:
            logger.error(
                f"❌ Projection failed for measurement {self.measurement.id} "
                f"at {self.measurement.gpsdata.latitude}, "
                f"{self.measurement.gpsdata.longitude}: {e}"
            )
            raise e
        distance_to_next_stop: float = self.measurement.next_stop_distance or 0.0

        # === Time Data ===
        measurement_time: float = self.measurement.measurement_time
        day_type: DayType = self._get_day_type(measurement_time)
        time_seconds = get_seconds_since_midnight(measurement_time)

        # === Bus Type === #

        bus_type = getattr(self.measurement, "bus_type", -1)
        door_number = getattr(self.measurement, "door_number", 0)
        deposits = getattr(self.measurement, "deposits", [])

        # === Schedule Adherence (Context) ===
        schedule_adherence = getattr(self.measurement, "schedule_adherence", 0.0)

        # === Targets ===
        occupancy_status: int = self.measurement.occupancy_status or 0
        served_ratio: float = self.served_ratio

        # === Traffic Data (Context) ===
        speed_ratio: float = self.measurement.speed_ratio or 1.0
        current_traffic_speed: float = self.measurement.current_speed or 0.0

        # Current speed: prefer GPS speed, fall back to derived
        gps_speed: float = self.measurement.gpsdata.speed
        current_speed: float = (
            gps_speed if gps_speed else (self.measurement.derived_speed or 0.0)
        )

        # === Weather Data ===
        weather: Weather = self.measurement.weather
        precipitation: float = weather.precip_intensity if weather else 0.0
        weather_code: int = weather.weather_code if weather else 0

        # === Scheduled Start Time Encoding ===
        scheduled_start_time = getattr(self.measurement, "scheduled_start_time", None)
        sch_starting_time_cos = 0.0
        sch_starting_time_sin = 0.0
        if scheduled_start_time:
            time_encoding = get_time_sin_cos_from_str(scheduled_start_time)
            if time_encoding:
                sch_starting_time_sin, sch_starting_time_cos = time_encoding

        # === Actual Start Time Encoding ===
        starting_time_cos = getattr(self.measurement, "starting_time_cos", 0.0) or 0.0
        starting_time_sin = getattr(self.measurement, "starting_time_sin", 0.0) or 0.0
        
        delay_genuine = getattr(self.measurement, "delay_genuine", 0)

        vector = PredictionVector(
            id=vector_id,
            trip_id=str(self.trip.id),
            route_id=route_id,
            direction_id=direction_id,
            stop_sequence=stop_sequence,
            bus_type=bus_type,
            door_number=door_number,
            deposit_grottarossa=1 if "Grottarossa" in deposits else 0,
            deposit_magliana=1 if "Magliana" in deposits else 0,
            deposit_tor_sapienza=1 if "Tor Sapienza" in deposits else 0,
            deposit_portonaccio=1 if "Portonaccio" in deposits else 0,
            deposit_monte_sacro=1 if "Monte Sacro" in deposits else 0,
            deposit_tor_pagnotta=1 if "Tor Pagnotta" in deposits else 0,
            deposit_tor_cervara=1 if "Tor Cervara" in deposits else 0,
            deposit_maglianella=1 if "Maglianella" in deposits else 0,
            deposit_costi=1 if "Costi" in deposits else 0,
            deposit_trastevere=1 if "Trastevere" in deposits else 0,
            deposit_acilia=1 if "Acilia" in deposits else 0,
            deposit_tor_vergata=1 if "Tor Vergata" in deposits else 0,
            deposit_porta_maggiore=1 if "Porta Maggiore" in deposits else 0,
            shape_dist_travelled=shape_dist_travelled,
            distance_to_next_stop=distance_to_next_stop,
            day_type=day_type,
            time=measurement_time,
            schedule_adherence=schedule_adherence,
            speed_ratio=speed_ratio,
            current_traffic_speed=current_traffic_speed,
            current_speed=current_speed,
            precipitation=precipitation,
            weather_code=weather_code,
            served_ratio=served_ratio,
            sch_starting_time_cos=sch_starting_time_cos,
            sch_starting_time_sin=sch_starting_time_sin,
            starting_time_cos=starting_time_cos,
            starting_time_sin=starting_time_sin,
            delay_genuine=delay_genuine,
        )

        label = PredictionLabel(
            id=vector_id,
            time=measurement_time,
            time_seconds=time_seconds,
            occupancy_status=occupancy_status,
        )

        return vector, label

    def _get_day_type(self, unix_timestamp: float) -> DayType:
        """Determines if the timestamp falls on a weekday, Saturday, or Sunday."""
        date_str, day_name, _ = get_timestamp_components(unix_timestamp)
        if day_name == "Saturday":
            return DayType.SATURDAY
        elif day_name == "Sunday":
            return DayType.SUNDAY
        else:
            return DayType.WEEKDAY


class TrafficLabel(LabelVector):
    def __init__(
        self,
        id: str,
        time: float,
        speed_ratio: float,
        current_traffic_speed: float,
    ):
        self.id = id
        self.time = time
        self.speed_ratio = speed_ratio
        self.current_traffic_speed = current_traffic_speed


class TrafficVector(Vector):
    def __init__(
        self,
        id: str,
        trip_id: str,
        day_type: DayType,
        time: float,
        hexagon_id: str = None,
    ):
        self.id = id
        self.trip_id = trip_id
        self.time = time
        self.hexagon_id = hexagon_id

        ### ======================= ###
        ###         Time Data       ###
        ### ======================= ###

        self.day_type: DayType = day_type
        self.rush_hour_status: int = self._is_rush_hour(time)
        self.time_encoding: tuple[float, float] = get_time_sin_cos(time)

    def _is_rush_hour(self, time: float) -> int:
        _, _, time_str = get_timestamp_components(time)
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        match time_obj.hour:
            case 7 | 8 | 9:
                return 1
            case 17 | 18:
                return 1
            case 19:
                if time_obj.minute > 30:
                    return 0
                return 1
            case _:
                return 0


class TrafficVectorizer(Vectorizer):
    def __init__(
        self,
        measurement: "Measurement",
    ):
        self.measurement = measurement

    def vectorize(self) -> tuple[TrafficVector, TrafficLabel]:
        """
        Transforms a Measurement into a TrafficVector and TrafficLabel.
        """
        vector_id = str(uuid.uuid4())

        # === Time Data ===
        measurement_time: float = self.measurement.measurement_time
        day_type: DayType = self._get_day_type(measurement_time)

        # === Traffic Data (Targets) ===
        speed_ratio: float = self.measurement.speed_ratio or 1.0
        current_traffic_speed: float = self.measurement.current_speed or 0.0

        hexagon_id: str = self.measurement.hexagon_id

        vector = TrafficVector(
            id=vector_id,
            trip_id=str(self.measurement.trip_id),
            day_type=day_type,
            time=measurement_time,
            hexagon_id=hexagon_id,
        )

        label = TrafficLabel(
            id=vector_id,
            time=measurement_time,
            speed_ratio=speed_ratio,
            current_traffic_speed=current_traffic_speed,
        )

        return vector, label

    def _get_day_type(self, unix_timestamp: float) -> DayType:
        """Determines if the timestamp falls on a weekday, Saturday, or Sunday."""
        date_str, day_name, _ = get_timestamp_components(unix_timestamp)
        if day_name == "Saturday":
            return DayType.SATURDAY
        elif day_name == "Sunday":
            return DayType.SUNDAY
        else:
            return DayType.WEEKDAY


class VehicleLabel(LabelVector):
    def __init__(self, id: str, vehicle_type: str, time: float):
        self.id = id
        self.vehicle_type = vehicle_type
        self.time = time


class VehicleVector(Vector):
    def __init__(
        self,
        id: str,
        route_id: str,
        trip_id: str,
        direction_id: int,
        timestamp: float,
    ):
        self.id = id
        self.route_id = route_id
        self.trip_id = trip_id
        self.direction_id = direction_id
        self.timestamp = timestamp


class VehicleVectorizer(Vectorizer):
    def __init__(
        self,
        measurement: "Measurement",
        trip: "Trip",
        vehicle_type_name: str = "Unknown",
    ):
        self.measurement = measurement
        self.trip = trip
        self.vehicle_type_name = vehicle_type_name

    def vectorize(self) -> tuple[VehicleVector, VehicleLabel]:
        vector_id = str(uuid.uuid4())

        vector = VehicleVector(
            id=vector_id,
            route_id=str(self.trip.route.id),  # Ensure string
            trip_id=str(self.trip.id),  # Ensure string
            direction_id=int(self.trip.direction_id)
            if self.trip.direction_id is not None
            else 0,
            timestamp=self.measurement.measurement_time,
        )

        label = VehicleLabel(
            id=vector_id,
            vehicle_type=self.vehicle_type_name,
            time=self.measurement.measurement_time,
        )

        return vector, label
