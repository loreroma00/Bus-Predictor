"""
LiveFeedFetcher - Fetches and parses GTFS-RT feeds.
"""

import logging
import pandas as pd
from google.transit import gtfs_realtime_pb2 as grt
import requests as rq


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

    def fetch(self) -> pd.DataFrame:
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
