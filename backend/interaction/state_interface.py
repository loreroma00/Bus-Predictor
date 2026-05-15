"""
State Interface - Read-only interface for GUI to access application state.

This module provides a clean abstraction layer between the GUI and application internals.
The GUI only depends on this interface, ensuring low coupling and respecting top-down imports.
"""

import io
import logging
import time
from typing import Optional
from application.domain import h3_utils


class StateInterface:
    """
    Read-only interface to application state for the debugging GUI.

    Wraps the Observatory and provides structured access to all data
    without exposing internal implementation details.
    """

    def __init__(self, observatory, predictor=None):
        """
        Initialize the state interface with an Observatory instance.

        Args:
            observatory: The main Observatory facade from the application.
        """
        self._observatory = observatory
        self._predictor = predictor
        self._last_feed_timestamp: float = 0.0
        self._command_registry: dict = {}
        self._last_command_output: str = ""

    # ============================================================
    # City & Hexagon Access
    # ============================================================

    def get_city_names(self) -> list[str]:
        """Get list of all city names."""
        return list(self._observatory.observed_cities.keys())

    def get_city_summary(self, city_name: str) -> Optional[dict]:
        """Get summary info about a city."""
        city = self._observatory.get_city(city_name)
        if not city:
            return None
        return {
            "name": city.name,
            "hexagon_count": len(city.hexagons),
            "active_buses": len(city.live_trip_index),
            "deposit_buses": len(city.live_trip_deposit),
        }

    def get_hexagons(self, city_name: str) -> list[dict]:
        """Get all hexagons in a city with their data."""
        city = self._observatory.get_city(city_name)
        if not city:
            return []

        hexagons = []
        for hex_id, hexagon in city.hexagons.items():
            hexagons.append(
                {
                    "hex_id": hex_id,
                    "bus_count": len(hexagon.mobile_agents["live_trips"]),
                    "street_count": len(hexagon.streets),
                    "temperature": hexagon.get_temperature(),
                    "humidity": hexagon.get_humidity(),
                    "wind_speed": hexagon.get_wind_speed(),
                    "precip_intensity": hexagon.get_precip_intensity(),
                    "preferentials": hexagon.get_preferentials(),
                    "current_speed": hexagon.get_current_speed(),
                    "flow_speed": hexagon.get_flow_speed(),
                    "speed_ratio": hexagon.get_speed_ratio(),
                }
            )
        return hexagons

    def get_hexagon_buses(self, city_name: str, hex_id: str) -> list[dict]:
        """Get all buses in a specific hexagon."""
        city = self._observatory.get_city(city_name)
        if not city:
            return []
        hexagon = city.hexagons.get(hex_id)
        if not hexagon:
            return []

        live_trips = []
        for _, live_trip in hexagon.mobile_agents["live_trips"].items():
            live_trips.append(self._live_trip_to_dict(live_trip))
        return live_trips

    def get_hexagon_details(self, city_name: str, hex_id: str) -> Optional[dict]:
        """Get detailed info about a specific hexagon."""
        city = self._observatory.get_city(city_name)
        if not city:
            return None

        hexagon = city.hexagons.get(hex_id)
        if not hexagon:
            return None

        # Coordinates
        lat, lon = h3_utils.get_coords_from_h3(hex_id)

        # Live trips
        live_trips_list = []
        for _, live_trip in hexagon.mobile_agents["live_trips"].items():
            live_trips_list.append(self._live_trip_to_dict(live_trip))

        # Streets
        streets = list(set(hexagon.streets.values()))

        return {
            "hex_id": hex_id,
            "center": {"lat": lat, "lon": lon},
            "streets": streets,
            "preferentials": hexagon.get_preferentials(),
            "buses": live_trips_list,
            # Aggregate traffic (averages)
            "current_speed": hexagon.get_current_speed(),
            "flow_speed": hexagon.get_flow_speed(),
            "speed_ratio": hexagon.get_speed_ratio(),
            # Per-direction traffic
            "traffic_by_direction": {
                d: {
                    "current_speed": hexagon.get_traffic().get_current_speed(d),
                    "flow_speed": hexagon.get_traffic().get_flow_speed(d),
                    "speed_ratio": hexagon.get_traffic().get_speed_ratio(d),
                }
                for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            },
            "weather": {
                "temperature": hexagon.get_temperature(),
                "precip_intensity": hexagon.get_precip_intensity(),
                "diffraction": 0,
            },
        }

    # ============================================================
    # Bus Access
    # ============================================================

    def get_all_buses(self, city_name: str) -> list[dict]:
        """Get all active buses in a city."""
        city = self._observatory.get_city(city_name)
        if not city:
            return []

        live_trips = []
        for live_trip_id in city.live_trip_index:
            live_trip = city.get_live_trip(live_trip_id)
            if live_trip:
                live_trips.append(self._live_trip_to_dict(live_trip))
        return live_trips

    def get_deposit_buses(self, city_name: str) -> list[dict]:
        """Get all buses currently in deposit."""
        city = self._observatory.get_city(city_name)
        if not city:
            return []

        buses = []
        for _, live_trip in city.live_trip_deposit.items():
            data = self._live_trip_to_dict(live_trip)
            data["status"] = "DEPOSIT"
            buses.append(data)
        return buses

    def get_bus(self, city_name: str, bus_id: str) -> Optional[dict]:
        """Get detailed info about a specific active live trip by vehicle id."""
        city = self._observatory.get_city(city_name)
        live_trip = city.get_live_trip(bus_id) if city else None
        if not live_trip:
            return None
        return self._live_trip_to_dict(live_trip, detailed=True)

    def _live_trip_to_dict(self, live_trip, detailed: bool = False) -> dict:
        """Convert a live trip object to a dictionary."""
        trip = live_trip.trip
        gps = live_trip.gps_data

        city = self._observatory.get_city("Rome")
        is_deposit = city.is_live_trip_in_deposit(live_trip.id) if city else False

        data = {
            "id": live_trip.id,
            "label": live_trip.label,
            "trip_id": trip.id if trip else None,
            "route_id": trip.route.id if trip and trip.route else None,
            "direction": trip.direction_name if trip else None,
            "location_name": live_trip.location_name,
            "hexagon_id": live_trip.hexagon_id,
            "last_seen": live_trip.last_seen_timestamp,
            "status": "DEPOSIT" if is_deposit else "ACTIVE",
        }

        if gps:
            data["gps"] = {
                "latitude": gps.latitude,
                "longitude": gps.longitude,
                "speed": gps.speed,
                "heading": gps.heading,
                "timestamp": gps.timestamp,
                "next_stop_id": gps.next_stop_id,
                "current_stop_sequence": gps.current_stop_sequence,
                "current_status": gps.current_status,
            }
            data["derived_speed"] = live_trip.derived_speed
            data["derived_bearing"] = live_trip.derived_bearing

        if detailed:
            data["occupancy_status"] = live_trip.occupancy_status
            data["crowding_level"] = live_trip.get_crowding_level()
            data["live_trip"] = {
                "measurement_count": len(live_trip.measurements),
                "is_finished": live_trip.is_finished,
            }

        return data

    # ============================================================
    # LiveTrip & Measurement Access
    # ============================================================

    def get_observers(self) -> list[dict]:
        """Compatibility surface: return active live trips with summary info."""
        live_trips = self._observatory.get_active_live_trips()
        if not live_trips:
            return []

        result = []
        for live_trip_id, live_trip in live_trips.items():
            trip = live_trip.trip
            result.append(
                {
                    "bus_id": live_trip.id,
                    "trip_id": trip.id if trip else None,
                    "route_id": trip.route.id if trip and trip.route else None,
                    "measurement_count": len(live_trip.measurements),
                    "archived_count": len(self._observatory.completed_live_trips),
                }
            )
        return result

    def get_observer_diary(self, bus_id: str) -> Optional[dict]:
        """Compatibility surface: get measurement info for one live trip."""
        live_trip = self._observatory.get_active_live_trips().get(bus_id)
        if not live_trip:
            return None

        measurements = [
            m.to_dict(live_trip.trip_id) for m in live_trip.measurements[-20:]
        ]
        return {
            "trip_id": live_trip.trip_id,
            "is_finished": live_trip.is_finished,
            "total_measurements": len(live_trip.measurements),
            "recent_measurements": measurements,
        }

    # ============================================================
    # Tracking Summary (Ingestion View)
    # ============================================================

    def get_tracking_summary(self) -> list[dict]:
        """
        Get a summary of all tracked live trips for the ingestion view.
        Similar to print_tracking_summary but returns structured data.
        """
        live_trips = self._observatory.get_active_live_trips()
        if not live_trips:
            return []

        rows = []
        for live_trip in live_trips.values():
            trip = live_trip.trip

            # Vehicle Type
            vehicle_type = "Unknown"
            if live_trip.vehicle_type:
                 vehicle_type = live_trip.vehicle_type.name

            # Basic info
            row = {
                "bus_id": live_trip.label,
                "vehicle_type": vehicle_type,
                "trip_id": trip.id if trip else "N/A",
                "route_id": trip.route.id if trip and trip.route else "N/A",
                "headsign": trip.direction_name[:20]
                if trip and trip.direction_name
                else "N/A",
                "location": live_trip.get_location_name() or "Resolving...",
                "samples": len(live_trip.measurements),
            }

            # Speed
            if live_trip.gps_data and live_trip.gps_data.speed:
                row["speed"] = f"{live_trip.gps_data.speed:.1f}"
            elif live_trip.derived_speed:
                row["speed"] = f"{live_trip.derived_speed:.1f}"
            else:
                row["speed"] = "0.0"

            # Status
            city = self._observatory.get_city("Rome")
            if city and city.is_live_trip_in_deposit(live_trip.id):
                row["status"] = "DEPOSIT"
            else:
                row["status"] = "ACTIVE"

            # Last seen
            delta_min = int((time.time() - live_trip.last_seen_timestamp) / 60)
            row["last_seen"] = f"{delta_min}m ago" if delta_min > 0 else "Just now"

            # Weather
            try:
                last_meas = live_trip.get_last_measurement()
                if last_meas and last_meas.weather:
                    w = last_meas.weather
                    row["weather"] = f"{w.description} ({w.precip_intensity}mm/h)"
                else:
                    row["weather"] = "N/A"
            except Exception:
                row["weather"] = "N/A"

            rows.append(row)

        # Sort by route ID
        rows.sort(key=lambda x: x["route_id"])
        return rows

    # ============================================================
    # Metrics & Stats
    # ============================================================

    def get_geocoding_stats(self) -> dict:
        """Get geocoding queue statistics."""
        if self._observatory._geocoding:
            return {
                "resolved_this_cycle": self._observatory._geocoding.get_and_reset_resolved_count(),
                "pending": self._observatory._geocoding.get_queue_size(),
                "enabled": True,
            }
        return {"enabled": False, "resolved_this_cycle": 0, "pending": 0}

    def set_feed_timestamp(self, timestamp: float):
        """Update the last feed timestamp (called by data module)."""
        self._last_feed_timestamp = timestamp

    def get_feed_timestamp(self) -> float:
        """Get the last feed timestamp.

        Derives from the most recent GPS timestamp across all active live trips,
        falling back to the manually-set value.
        """
        latest = self._last_feed_timestamp
        for city in self._observatory.observed_cities.values():
            for live_trip_id, hex_id in city.live_trip_index.items():
                hexagon = city.hexagons.get(hex_id)
                if not hexagon:
                    continue
                live_trip = hexagon.get_live_trip(live_trip_id)
                if live_trip and live_trip.gps_data and live_trip.gps_data.timestamp:
                    ts = float(live_trip.gps_data.timestamp)
                    if ts > latest:
                        latest = ts
        return latest

    def get_system_stats(self) -> dict:
        """Get overall system statistics."""
        live_trips = self._observatory.get_active_live_trips() or {}
        cities = self._observatory.observed_cities or {}

        total_active = 0
        total_deposit = 0
        total_hexagons = 0

        for city_name, city in cities.items():
            total_active += len(city.live_trip_index)
            total_deposit += len(city.live_trip_deposit)
            total_hexagons += len(city.hexagons)

        return {
            "observer_count": len(live_trips),
            "active_buses": total_active,
            "deposit_buses": total_deposit,
            "hexagon_count": total_hexagons,
            "city_count": len(cities),
        }

    def get_traffic_stats(self, city_name: str) -> dict:
        """Get traffic data statistics for a city."""
        city = self._observatory.get_city(city_name)
        if not city:
            return {"total_hexagons": 0, "with_traffic": 0}

        with_traffic = sum(1 for h in city.hexagons.values() if h.get_speed_ratio() > 0)
        return {
            "total_hexagons": len(city.hexagons),
            "with_traffic": with_traffic,
        }

    def get_ledger_stats(self) -> dict:
        """Get statistics about the loaded ledgers."""
        topology = self._observatory.topology
        if not topology:
            return {"loaded": False}

        return {
            "loaded": True,
            "trips_count": len(topology.trips),
            "routes_count": len(topology.routes),
            "stops_count": len(topology.stops),
            "shapes_count": len(topology.shapes),
            "current_md5": self._observatory.current_md5,
        }

    # ============================================================
    # Map Data (for Dashboard GUI)
    # ============================================================

    def get_bus_positions(self, city_name: str) -> list[dict]:
        """Get all active bus positions with GPS coordinates for map rendering."""
        city = self._observatory.get_city(city_name)
        if not city:
            return []

        positions = []
        for live_trip_id, hex_id in city.live_trip_index.items():
            hexagon = city.hexagons.get(hex_id)
            if not hexagon:
                continue
            live_trip = hexagon.get_live_trip(live_trip_id)
            if not live_trip or not live_trip.gps_data:
                continue

            trip = live_trip.trip
            speed = live_trip.gps_data.speed or live_trip.derived_speed or 0.0

            positions.append({
                "id": live_trip.id,
                "label": live_trip.label,
                "lat": live_trip.gps_data.latitude,
                "lon": live_trip.gps_data.longitude,
                "route_id": trip.route.id if trip and trip.route else "?",
                "direction": (trip.direction_name or "")[:25] if trip else "",
                "speed": round(speed, 1),
                "status": "ACTIVE",
            })
        return positions

    def get_traffic_hexagons(self, city_name: str) -> list[dict]:
        """Get hexagons with traffic data, including polygon boundaries for map overlay."""
        import h3

        city = self._observatory.get_city(city_name)
        if not city:
            return []

        result = []
        for hex_id, hexagon in city.hexagons.items():
            speed_ratio = hexagon.get_speed_ratio()
            if speed_ratio <= 0:
                continue

            boundary = h3.cell_to_boundary(hex_id)
            boundary_coords = [[lat, lon] for lat, lon in boundary]

            result.append({
                "hex_id": hex_id,
                "boundary": boundary_coords,
                "speed_ratio": round(speed_ratio, 3),
                "bus_count": len(hexagon.mobile_agents["live_trips"]),
                "current_speed": round(hexagon.get_current_speed(), 1),
            })
        return result

    def get_city_center(self, city_name: str) -> tuple[float, float]:
        """Get the center coordinates for a city. Defaults to Rome."""
        city = self._observatory.get_city(city_name)
        if city:
            bbox = city.get_bounding_box()
            if bbox:
                min_lat, min_lon, max_lat, max_lon = bbox
                return ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
        return (41.9028, 12.4964)

    # ============================================================
    # Ledger Info (for Dashboard GUI)
    # ============================================================

    def get_all_ledger_info(self) -> dict:
        """Get combined info from all ledger types, including recent content samples."""
        obs = self._observatory

        topology_info = {"loaded": False, "sample_routes": [], "sample_trips": []}
        if obs.topology:
            route_ids = sorted(obs.topology.routes.keys())
            topology_info = {
                "loaded": True,
                "trips": len(obs.topology.trips),
                "routes": len(obs.topology.routes),
                "stops": len(obs.topology.stops),
                "shapes": len(obs.topology.shapes),
                "md5": obs.current_md5 or "N/A",
                "sample_routes": route_ids[:20],
            }

        schedule_info = {"loaded": obs.schedule_ledger is not None, "sample_entries": []}
        if obs.schedule_ledger and obs.schedule_ledger.schedule:
            index = obs.schedule_ledger.schedule.index
            schedule_info["routes_indexed"] = len(index)
            # Sample: first 10 routes with their direction count
            for route_id in sorted(index.keys())[:10]:
                directions = list(index[route_id].keys())
                schedule_info["sample_entries"].append({
                    "route_id": route_id,
                    "directions": len(directions),
                })

        historical_info = {
            "type": "database-backed",
            "table": obs.historical._table if obs.historical else "N/A",
        }

        prediction_ledger = getattr(self._predictor, "predicted", None)
        predicted_info = {
            "type": "database-backed",
            "owner": "Predictor",
            "table": prediction_ledger._table if prediction_ledger else "N/A",
        }

        vehicle_table = "N/A"
        for vehicle in obs.vehicles.values():
            ledger = getattr(vehicle, "_history_ledger", None)
            if ledger:
                vehicle_table = ledger._table
                break
        vehicle_info = {
            "type": "database-backed",
            "owner": "Vehicle",
            "table": vehicle_table,
        }

        return {
            "topology": topology_info,
            "schedule": schedule_info,
            "historical": historical_info,
            "predicted": predicted_info,
            "vehicle": vehicle_info,
        }

    def resolve_direction_name(self, route_id: str, direction_id: int) -> str:
        """Resolve direction_id to its textual headsign from the topology."""
        topology = self._observatory.topology
        if not topology:
            return str(direction_id)
        for trip in topology.trips.values():
            if trip.route.id == route_id and trip.direction_id == direction_id:
                return trip.direction_name or f"Direction {direction_id}"
        return str(direction_id)

    def get_recent_predictions(self) -> list[dict]:
        """Return all predictions from in-memory buffer (no DB query)."""
        prediction_ledger = getattr(self._predictor, "predicted", None)
        if not prediction_ledger:
            return []
        return prediction_ledger.get_today_predictions()

    def get_recent_vehicle_trips(self) -> list[dict]:
        """Return all vehicle trip records from in-memory buffer (no DB query)."""
        obs = self._observatory
        records = []
        for vehicle in obs.vehicles.values():
            records.extend(vehicle.get_today_vehicle_trips())
        return records

    def get_prediction_stops(self, route_id: str, direction_id: int, scheduled_start: str) -> list[dict]:
        """Return per-stop prediction data for a specific trip (for detail popup)."""
        prediction_ledger = getattr(self._predictor, "predicted", None)
        if not prediction_ledger:
            return []
        return prediction_ledger.get_trip_stops(route_id, direction_id, scheduled_start)

    def get_trip_measurements(self, trip_id: str) -> list[dict]:
        """Return per-measurement data for a specific trip (for detail popup)."""
        obs = self._observatory
        if not obs.historical:
            return []
        return obs.historical.get_trip_measurements(trip_id)

    # ============================================================
    # Model Info (for Dashboard GUI)
    # ============================================================

    def set_predictor_info(self, model_name: str):
        """Store model name for GUI display."""
        self._model_name = model_name

    def get_model_info(self) -> dict:
        """Get model loading status and info."""
        model_name = getattr(self, "_model_name", None)
        return {
            "loaded": model_name is not None,
            "model_name": model_name or "Not loaded",
        }

    # ============================================================
    # Service Thread Status (for Dashboard GUI)
    # ============================================================

    # ============================================================
    # Command Execution (for Dashboard GUI)
    # ============================================================

    def set_command_registry(self, registry: dict):
        """Store reference to the console command registry."""
        self._command_registry = registry

    def get_available_commands(self) -> list[dict]:
        """Get list of available commands with their help text."""
        # Commands known to need arguments (from their help text patterns)
        _ARGS_COMMANDS = {
            "print hex", "print diary", "fetch data", "print diaries vehicle",
            "pause traffic service", "validate", "validate live",
            "weather strategy", "trip validation chart",
        }
        commands = []
        for cmd_name, cmd_instance in self._command_registry.items():
            commands.append({
                "name": cmd_name,
                "help": cmd_name,
                "needs_args": cmd_name in _ARGS_COMMANDS,
            })
        return commands

    def execute_command(self, cmd_name: str, args: str = "") -> str:
        """Execute a registered command and capture its log output."""
        cmd = self._command_registry.get(cmd_name)
        if not cmd:
            return f"Unknown command: {cmd_name}"

        # Capture log output during execution
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            cmd.execute(args)
        except Exception as e:
            buf.write(f"Error: {e}\n")
        finally:
            logger.removeHandler(handler)

        output = buf.getvalue().strip()
        self._last_command_output = output
        return output or "(command produced no output)"

    def get_last_command_output(self) -> str:
        """Get the output from the last executed command."""
        return self._last_command_output

    def get_service_thread_status(self) -> dict:
        """Get status of all background service threads."""
        from . import services

        threads = {
            "Collection": services.COLLECTION_THREAD,
            "Saving": services.SAVING_THREAD,
            "Weather": services.WEATHER_THREAD,
            "Traffic": services.TRAFFIC_THREAD,
            "Geocoding": services.GEOCODING_THREAD,
            "Uptime": services.UPTIME_THREAD,
            "GUI": services.GUI_THREAD,
        }

        status = {}
        for name, thread in threads.items():
            if thread is not None and thread.is_alive():
                status[name] = "running"
            elif thread is not None:
                status[name] = "stopped"
            else:
                status[name] = "not started"

        # Add validation status
        validation = services.get_validation_status()
        status["Batch Validation"] = validation.get("batch", "idle")
        status["Live Validation"] = validation.get("live", "idle")

        return status
