"""
LiveFeedFetcher - Fetches and parses GTFS-RT feeds.
"""

import logging
import pandas as pd
from google.transit import gtfs_realtime_pb2 as grt
import requests as rq

from application.domain.live_data import LiveFeedRecord


HEADERS: dict[str, str] = {
    "Referer": "https://romamobilita.it/sistemi-e-tecnologie/open-data/",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}


class LiveFeedFetcher:
    """Fetches and parses GTFS-RT vehicle positions and trip updates."""

    def __init__(self, vehicles_url=None, trips_url=None):
        """Store the GTFS-RT endpoints; warns if either URL is missing."""
        self.vehicles_url = vehicles_url
        self.trips_url = trips_url
        
        if not self.vehicles_url:
            logging.warning("LiveFeedFetcher initialized without vehicles_url")
        if not self.trips_url:
            logging.warning("LiveFeedFetcher initialized without trips_url")

    def fetch(self) -> list[LiveFeedRecord]:
        """Fetch and normalize GTFS-RT feeds into domain-level records."""
        return self._records_from_frame(self.fetch_frame())

    def fetch_frame(self) -> pd.DataFrame:
        """
        Fetch vehicle positions and trip updates, merge them.
        Returns merged DataFrame or empty DataFrame on error.
        """
        # 1. Fetch Vehicle Positions (Required)
        if not self.vehicles_url:
            logging.error("Cannot fetch: vehicles_url is missing")
            return pd.DataFrame()

        df_vp = pd.DataFrame()
        try:
            resp_vp = rq.get(self.vehicles_url, headers=HEADERS, timeout=15)
            if resp_vp.status_code == 200:
                vp_raw = grt.FeedMessage()
                try:
                    vp_raw.ParseFromString(resp_vp.content)
                    df_vp = self._parse_vehicle_positions(vp_raw)
                except Exception as e:
                    logging.error(f"Error parsing VehiclePositions pbuf: {e}")
            else:
                logging.warning(f"VehiclePositions feed unavailable: HTTP {resp_vp.status_code}")
        except Exception as e:
            logging.error(f"Error fetching VehiclePositions: {e}")

        # If no vehicles, we can't do anything
        if df_vp.empty:
            return pd.DataFrame()

        # 2. Fetch Trip Updates (Optional)
        df_tu = pd.DataFrame()
        try:
            resp_tu = rq.get(self.trips_url, headers=HEADERS, timeout=15)
            if resp_tu.status_code == 200:
                trip_raw = grt.FeedMessage()
                try:
                    trip_raw.ParseFromString(resp_tu.content)
                    df_tu = self._parse_trip_updates(trip_raw)
                except Exception as e:
                    logging.error(f"Error parsing TripUpdates pbuf: {e}")
            else:
                logging.warning(f"TripUpdates feed unavailable: HTTP {resp_tu.status_code}")
        except Exception as e:
            logging.error(f"Error fetching TripUpdates: {e}")

        logging.info(f"Fetched {len(df_vp)} vehicles and {len(df_tu)} trip updates.")

        # 3. Merge
        if not df_tu.empty:
            merged = pd.merge(
                df_vp,
                df_tu,
                on="tripId",
                how="left",
                suffixes=("", "_tu"),
            )
        else:
            merged = df_vp

        return merged
    def _records_from_frame(self, frame: pd.DataFrame) -> list[LiveFeedRecord]:
        """Convert the merged feed DataFrame into LiveFeedRecord objects."""
        if frame.empty:
            return []

        records: list[LiveFeedRecord] = []
        for _, row in frame.iterrows():
            vehicle_id = self._clean(row.get("vehicleId"))
            trip_id = self._clean(row.get("tripId"))
            latitude = self._clean(row.get("latitude"))
            longitude = self._clean(row.get("longitude"))
            timestamp = self._clean(row.get("timestamp"))

            if not vehicle_id or not trip_id or latitude is None or longitude is None:
                continue

            records.append(
                LiveFeedRecord(
                    vehicle_id=str(vehicle_id),
                    vehicle_label=self._clean(row.get("vehicleLabel")),
                    trip_id=str(trip_id),
                    route_id=self._clean(row.get("routeId")),
                    direction_id=self._to_int(row.get("directionId")),
                    latitude=float(latitude),
                    longitude=float(longitude),
                    bearing=float(self._clean(row.get("bearing"), 0.0) or 0.0),
                    speed=float(self._clean(row.get("speed"), 0.0) or 0.0),
                    timestamp=float(timestamp or 0),
                    scheduled_start_time=self._clean(row.get("startTime")),
                    start_date=self._clean(row.get("startDate")),
                    current_stop_sequence=self._to_int(
                        row.get("currentStopSequence")
                    ),
                    current_status=self._to_int(row.get("currentStatus")),
                    stop_id=self._clean(row.get("stopId")),
                    occupancy_status=self._to_int(row.get("occupancyStatus")),
                    stop_updates=self._clean_stop_updates(row.get("stopUpdates")),
                )
            )

        return records

    @staticmethod
    def _clean(value, default=None):
        """Return None/default for pandas missing values without touching lists."""
        if isinstance(value, list):
            return value
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        return value

    @classmethod
    def _to_int(cls, value):
        """Best-effort integer conversion for optional feed fields."""
        value = cls._clean(value)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _clean_stop_updates(cls, value) -> list[dict]:
        """Return stop updates as a list of dictionaries."""
        value = cls._clean(value, [])
        return value if isinstance(value, list) else []

    def _parse_vehicle_positions(self, feed: grt.FeedMessage) -> pd.DataFrame:
        """Parse vehicle positions from protobuf feed."""
        positions = []

        for entity in feed.entity:
            if entity.HasField("vehicle"):
                v = entity.vehicle
                real_id = v.vehicle.label if v.vehicle.label else 0
                positions.append(
                    {
                        "vehicleId": v.vehicle.id,
                        "vehicleLabel": real_id,
                        "tripId": v.trip.trip_id,
                        "routeId": v.trip.route_id,
                        "directionId": v.trip.direction_id,
                        "latitude": v.position.latitude,
                        "longitude": v.position.longitude,
                        "bearing": v.position.bearing,
                        "speed": v.position.speed,
                        "timestamp": v.timestamp,
                        "startTime": v.trip.start_time,
                        "startDate": v.trip.start_date,
                        "currentStopSequence": v.current_stop_sequence,
                        "currentStatus": v.current_status,
                        "stopId": v.stop_id,
                        "occupancyStatus": v.occupancy_status,
                    }
                )

        return pd.DataFrame(positions)

    def _parse_trip_updates(self, feed: grt.FeedMessage) -> pd.DataFrame:
        """Parse trip updates from protobuf feed."""
        updates = []

        for entity in feed.entity:
            if entity.HasField("trip_update"):
                tu = entity.trip_update

                stop_updates = []
                for stu in tu.stop_time_update:
                    stop_updates.append(
                        {
                            "stop_sequence": stu.stop_sequence,
                            "stop_id": stu.stop_id,
                            "arrival_time": stu.arrival.time,
                            "departure_time": stu.departure.time,
                            "delay": stu.arrival.delay,
                        }
                    )

                updates.append(
                    {
                        "tripId": tu.trip.trip_id,
                        "routeId": tu.trip.route_id,
                        "vehicleId": tu.vehicle.id,
                        "stopUpdates": stop_updates,
                    }
                )

        return pd.DataFrame(updates)
