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

    def __init__(self, observatory):
        """
        Initialize the state interface with an Observatory instance.

        Args:
            observatory: The main Observatory facade from the application.
        """
        self._observatory = observatory
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
            "active_buses": len(city.bus_index),
            "deposit_buses": len(city.bus_deposit),
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
                    "bus_count": len(hexagon.mobile_agents["buses"]),
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

        buses = []
        for bus_id, bus in hexagon.mobile_agents.items():
            buses.append(self._bus_to_dict(bus))
        return buses

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

        # Buses
        buses_list = []
        for bus_id, bus in hexagon.mobile_agents["buses"].items():
            buses_list.append(self._bus_to_dict(bus))

        # Streets
        streets = list(set(hexagon.streets.values()))

        return {
            "hex_id": hex_id,
            "center": {"lat": lat, "lon": lon},
            "streets": streets,
            "preferentials": hexagon.get_preferentials(),
            "buses": buses_list,
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

        buses = []
        for bus_id in city.bus_index:
            bus = city.get_bus(bus_id)
            if bus:
                buses.append(self._bus_to_dict(bus))
        return buses

    def get_deposit_buses(self, city_name: str) -> list[dict]:
        """Get all buses currently in deposit."""
        city = self._observatory.get_city(city_name)
        if not city:
            return []

        buses = []
        for bus_id, bus in city.bus_deposit.items():
            data = self._bus_to_dict(bus)
            data["status"] = "DEPOSIT"
            buses.append(data)
        return buses

    def get_bus(self, city_name: str, bus_id: str) -> Optional[dict]:
        """Get detailed info about a specific bus."""
        bus = self._observatory.get_bus(city_name, bus_id)
        if not bus:
            return None
        return self._bus_to_dict(bus, detailed=True)

    def _bus_to_dict(self, bus, detailed: bool = False) -> dict:
        """Convert a bus object to a dictionary."""
        trip = bus.trip
        gps = bus.GPSData

        data = {
            "id": bus.id,
            "label": bus.label,
            "trip_id": trip.id if trip else None,
            "route_id": trip.route.id if trip and trip.route else None,
            "direction": trip.direction_name if trip else None,
            "location_name": bus.location_name,
            "hexagon_id": bus.hexagon_id,
            "last_seen": bus.last_seen_timestamp,
            "status": "DEPOSIT"
            if self._observatory.is_bus_in_deposit("Rome", bus.id)
            else "ACTIVE",
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
            data["derived_speed"] = bus.derived_speed
            data["derived_bearing"] = bus.derived_bearing

        if detailed:
            data["occupancy_status"] = bus.occupancy_status
            data["crowding_level"] = bus.get_crowding_level()

            observer = bus.observer
            if observer:
                data["observer"] = {
                    "has_diary": observer.current_diary is not None,
                    "measurement_count": len(observer.current_diary.measurements)
                    if observer.current_diary
                    else 0,
                    "archived_diary_count": len(observer.diary_history),
                }

        return data

    # ============================================================
    # Observer & Diary Access
    # ============================================================

    def get_observers(self) -> list[dict]:
        """Get all observers with summary info."""
        observers = self._observatory.get_observers()
        if not observers:
            return []

        result = []
        for obs_id, obs in observers.items():
            bus = obs.assignedVehicle
            diary = obs.current_diary
            result.append(
                {
                    "bus_id": bus.id if bus else None,
                    "trip_id": bus.trip.id if bus and bus.trip else None,
                    "route_id": bus.trip.route.id
                    if bus and bus.trip and bus.trip.route
                    else None,
                    "measurement_count": len(diary.measurements) if diary else 0,
                    "archived_count": len(obs.diary_history),
                }
            )
        return result

    def get_observer_diary(self, bus_id: str) -> Optional[dict]:
        """Get detailed diary info for a specific observer."""
        observers = self._observatory.get_observers()
        if not observers:
            return None

        for obs_id, obs in observers.items():
            if obs.assignedVehicle and obs.assignedVehicle.id == bus_id:
                diary = obs.current_diary
                if not diary:
                    return None

                measurements = []
                for m in diary.measurements[-20:]:  # Last 20 measurements
                    measurements.append(m.to_dict(diary.trip_id))

                return {
                    "trip_id": diary.trip_id,
                    "is_finished": diary.is_finished,
                    "total_measurements": len(diary.measurements),
                    "recent_measurements": measurements,
                }
        return None

    # ============================================================
    # Tracking Summary (Ingestion View)
    # ============================================================

    def get_tracking_summary(self) -> list[dict]:
        """
        Get a summary of all tracked buses for the ingestion view.
        Similar to print_tracking_summary but returns structured data.
        """
        observers = self._observatory.get_observers()
        if not observers:
            return []

        rows = []
        for obs_id, obs in observers.items():
            bus = obs.assignedVehicle
            trip = bus.trip if bus else None
            diary = obs.current_diary

            # Vehicle Type
            vehicle_type = "Unknown"
            if bus and bus.vehicle_type:
                 vehicle_type = bus.vehicle_type.name

            # Basic info
            row = {
                "bus_id": bus.label if bus else "N/A",
                "vehicle_type": vehicle_type,
                "trip_id": trip.id if trip else "N/A",
                "route_id": trip.route.id if trip and trip.route else "N/A",
                "headsign": trip.direction_name[:20]
                if trip and trip.direction_name
                else "N/A",
                "location": bus.get_location_name() or "Resolving..." if bus else "N/A",
                "samples": len(diary.measurements) if diary else 0,
            }

            # Speed
            if bus and bus.GPSData and bus.GPSData.speed:
                row["speed"] = f"{bus.GPSData.speed:.1f}"
            elif bus and bus.derived_speed:
                row["speed"] = f"{bus.derived_speed:.1f}"
            else:
                row["speed"] = "0.0"

            # Status
            if bus and self._observatory.is_bus_in_deposit("Rome", bus.id):
                row["status"] = "DEPOSIT"
            else:
                row["status"] = "ACTIVE"

            # Last seen
            if bus:
                delta_min = int((time.time() - bus.last_seen_timestamp) / 60)
                row["last_seen"] = f"{delta_min}m ago" if delta_min > 0 else "Just now"
            else:
                row["last_seen"] = "N/A"

            # Weather
            try:
                if diary:
                    last_meas = diary.get_last_measurement()
                    if last_meas and last_meas.weather:
                        w = last_meas.weather
                        row["weather"] = f"{w.description} ({w.precip_intensity}mm/h)"
                    else:
                        row["weather"] = "N/A"
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

        Derives from the most recent GPS timestamp across all active buses,
        falling back to the manually-set value.
        """
        latest = self._last_feed_timestamp
        for city in self._observatory.observed_cities.values():
            for bus_id, hex_id in city.bus_index.items():
                hexagon = city.hexagons.get(hex_id)
                if not hexagon:
                    continue
                bus = hexagon.get_bus(bus_id)
                if bus and bus.GPSData and bus.GPSData.timestamp:
                    ts = float(bus.GPSData.timestamp)
                    if ts > latest:
                        latest = ts
        return latest

    def get_system_stats(self) -> dict:
        """Get overall system statistics."""
        observers = self._observatory.get_observers() or {}
        cities = self._observatory.observed_cities or {}

        total_active = 0
        total_deposit = 0
        total_hexagons = 0

        for city_name, city in cities.items():
            total_active += len(city.bus_index)
            total_deposit += len(city.bus_deposit)
            total_hexagons += len(city.hexagons)

        return {
            "observer_count": len(observers),
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
        for bus_id, hex_id in city.bus_index.items():
            hexagon = city.hexagons.get(hex_id)
            if not hexagon:
                continue
            bus = hexagon.get_bus(bus_id)
            if not bus or not bus.GPSData:
                continue

            trip = bus.trip
            speed = bus.GPSData.speed or bus.derived_speed or 0.0

            positions.append({
                "id": bus.id,
                "label": bus.label,
                "lat": bus.GPSData.latitude,
                "lon": bus.GPSData.longitude,
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
                "bus_count": len(hexagon.mobile_agents["buses"]),
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

        predicted_info = {
            "type": "database-backed",
            "table": obs.predicted._table if obs.predicted else "N/A",
        }

        vehicle_info = {
            "type": "database-backed",
            "table": obs.vehicle_ledger._table if obs.vehicle_ledger else "N/A",
        }

        return {
            "topology": topology_info,
            "schedule": schedule_info,
            "historical": historical_info,
            "predicted": predicted_info,
            "vehicle": vehicle_info,
        }

    def get_recent_predictions(self) -> list[dict]:
        """Return all predictions from in-memory buffer (no DB query)."""
        obs = self._observatory
        if not obs.predicted:
            return []
        return obs.predicted.get_today_predictions()

    def get_recent_vehicle_trips(self) -> list[dict]:
        """Return all vehicle trip records from in-memory buffer (no DB query)."""
        obs = self._observatory
        if not obs.vehicle_ledger:
            return []
        return obs.vehicle_ledger.get_today_vehicle_trips()

    def get_prediction_stops(self, route_id: str, direction_id: int, scheduled_start: str) -> list[dict]:
        """Return per-stop prediction data for a specific trip (for detail popup)."""
        obs = self._observatory
        if not obs.predicted:
            return []
        return obs.predicted.get_trip_stops(route_id, direction_id, scheduled_start)

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
            "weather strategy", "fotoromanzo",
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
