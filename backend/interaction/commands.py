"""
Commands - User interaction commands.

Commands receive dependencies via injection, not global imports.
Control flow uses events (shutdown, start, stop).
"""

import logging
from abc import ABC, abstractmethod
import pandas as pd
import time

from .events import console_events
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.domain.observers import Observatory


class Command(ABC):
    """Base command interface."""

    command_name: str

    @abstractmethod
    def execute(self, args):
        pass

    @staticmethod
    @abstractmethod
    def help():
        pass


# ============================================================
# Commands that need Observatory (injected)
# ============================================================


class print_hex(Command):
    command_name = "print hex"

    def __init__(self, observatory: "Observatory"):
        self._obs = observatory

    def execute(self, args):
        hex_id = args.strip()
        city = self._obs.get_city("Rome")  # Default to Rome for now
        if not city:
            logging.warning("City Rome not found")
            return

        hexagon = city.hexagons.get(hex_id)
        if not hexagon:
            logging.warning(f"Hexagon {hex_id} not found")
            logging.info(f"Available hexagons: {len(city.hexagons)}")
            return

        logging.info(f"--- Hexagon {hex_id} ---")
        logging.info(f"Streets: {len(hexagon.streets)}")
        logging.info(f"Buses: {len(hexagon.mobile_agents['buses'])}")

        # Traffic Data - per direction
        logging.info("\n🚗 Traffic Status (per direction):")
        traffic = hexagon.get_traffic()
        for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            spd = traffic.get_current_speed(d)
            ratio = traffic.get_speed_ratio(d)
            logging.info(f"  {d}: speed={spd:.1f} kph, ratio={ratio:.2f}")

        # Aggregate
        avg_speed = hexagon.get_current_speed()
        avg_ratio = hexagon.get_speed_ratio()
        logging.info(f"\n  OVERALL: speed={avg_speed:.1f} kph, ratio={avg_ratio:.2f}")
        logging.info(f"  Congestion: {(1 - avg_ratio) * 100:.1f}%")

        # Weather
        w = hexagon.weather
        if w:
            logging.info("\n☁️ Weather:")
            logging.info(f"  Temp: {w.temperature:.1f}°C")
            logging.info(f"  Precip: {w.precip_intensity} mm/h")

    @staticmethod
    def help():
        logging.info("print hex <hex_id>: Show details for a specific hexagon")


class debug_traffic(Command):
    command_name = "debug traffic"

    def __init__(self, observatory: "Observatory"):
        self._obs = observatory

    def execute(self, args):
        from application.live import data as live_data

        city = self._obs.get_city("Rome")
        if not city:
            logging.warning("City Rome not found")
            return

        # Print object IDs for comparison
        logging.info(f"Observatory ID: {id(self._obs)}")
        logging.info(f"City ID: {id(city)}")
        logging.info(f"data.OBSERVATORY ID: {id(live_data.OBSERVATORY)}")
        logging.info(
            f"data.TRAFFIC_SERVICE city ID: {id(live_data.TRAFFIC_SERVICE._city) if live_data.TRAFFIC_SERVICE else 'N/A'}"
        )

        # Find hexagons with traffic data
        traffic_hexes = []
        for hex_id, hexagon in city.hexagons.items():
            ratio = hexagon.get_speed_ratio()
            if ratio > 0:
                traffic_hexes.append((hex_id, ratio))

        logging.info(f"\nTotal hexagons: {len(city.hexagons)}")
        logging.info(f"Hexagons with traffic data: {len(traffic_hexes)}")

        if traffic_hexes:
            logging.info("\nTop 10 with traffic:")
            for hid, ratio in sorted(traffic_hexes, key=lambda x: x[1])[:10]:
                congestion = (1 - ratio) * 100
                logging.info(
                    f"  {hid}: ratio={ratio:.2f}, congestion={congestion:.1f}%"
                )

    @staticmethod
    def help():
        logging.info(
            "debug traffic: Show hexagons with traffic data and verify object references"
        )


class print_diary(Command):
    command_name = "print diary"

    def __init__(self, observatory: "Observatory"):
        self._obs = observatory

    def execute(self, args):
        t_id = args.strip()
        found = False

        # Create resolver functions
        def resolve_stop_name(stop_id):
            try:
                ledger: dict[str, dict] = self._obs.get_ledger()
                if ledger and "stops" in ledger:
                    stop = ledger["stops"].get(stop_id)
                    # Stops are dicts with 'stop_name' key, not objects with .name
                return stop.get("stop_name") if stop else None
                return None
            except Exception as e:
                logging.error(f"Error in resolve_stop_name: {e}")
                return f"ERROR:{e}"

        def resolve_street_name(lat, lon):
            try:
                city = self._obs.get_city("Rome")
                if city and self._obs._geocoding:
                    return self._obs._geocoding.get_street(lat, lon)
                return None
            except Exception as e:
                logging.error(f"Error in resolve_street_name: {e}")
                import traceback

                logging.exception("Traceback:")
                return f"ERROR:{e}"

        logging.info(f"--- Active Diary for Trip {t_id} ---")
        diary = self._obs.search_diary(t_id)
        if diary:
            logging.info(
                diary.format_rich(
                    stop_name_resolver=resolve_stop_name,
                    street_name_resolver=resolve_street_name,
                )
            )
            found = True

        diary = self._obs.search_history(t_id)
        if diary:
            logging.info(
                diary.format_rich(
                    stop_name_resolver=resolve_stop_name,
                    street_name_resolver=resolve_street_name,
                )
            )
            found = True

        if not found:
            logging.warning(f"No diary found for Trip ID {t_id}")

    @staticmethod
    def help():
        logging.info("print diary <trip_id>: Show recorded stops for a specific trip")


class fetch_data(
    Command
):  # TO REDO: Static data to be inferred from position; debating if useful or not...
    command_name = "fetch data"

    def __init__(self, observatory, time_formatter):
        self._obs = observatory
        self._format_time = time_formatter

    def execute(self, args):
        t_id = args.strip()
        diary = self._obs.search_diary(t_id)

        logging.info(f"\n--- TRIP {t_id} STATUS ---")
        logging.info(
            f"{'STOP NAME':<30} | {'SEQ':<3} | {'TIME':<8} | {'SCHED':<8} | {'STATUS'}"
        )
        logging.info("-" * 75)

        if diary:
            for m in diary.measurements:
                s_info = next(
                    (s for s in diary.scheduledTimes if s["stop_id"] == m.id), None
                )
                s_name = s_info["stop_name"] if s_info else m.id
                t_str = self._format_time(m.actual_time)
                s_str = m.scheduled_time if m.scheduled_time else "N/A"
                logging.info(
                    f"{s_name[:30]:<30} | {m.sequence:<3} | {t_str:<8} | {s_str:<8} | RECORDED"
                )

        update = None
        if diary and diary.observer:
            bus = diary.observer.get_bus()
            if bus:
                update = bus.get_latest_update()

                # Show traffic info for current location
                if bus.hexagon_id:
                    city = self._obs.get_city("Rome")
                    if city:
                        hexagon = city.hexagons.get(bus.hexagon_id)
                        if hexagon:
                            spd = getattr(hexagon, "current_speed", 0)
                            ratio = hexagon.get_speed_ratio()
                            logging.info(
                                f"\n📍 Location Traffic: {spd:.1f} kph (Speed Ratio: {ratio:.0%})"
                            )

        if update:
            for stu in update.next_stops:
                s_id = stu["stop_id"]
                static_match = next(
                    (s for s in update.upcoming_stops if s["stop_id"] == s_id), None
                )
                s_name = static_match["stop_name"] if static_match else s_id
                s_str = static_match["formatted_time"] if static_match else "N/A"
                t_str = self._format_time(stu["arrival_time"])
                delta_min = int((stu["arrival_time"] - time.time()) / 60)
                t_display = f"{t_str} - {delta_min} min"
                logging.info(
                    f"{s_name[:30]:<30} | {stu['stop_sequence']:<3} | {t_display:<16} | {s_str:<8} | PREDICTED"
                )

        if not diary and not update:
            logging.warning("No active diary or live data found for this trip.")

    @staticmethod
    def help():
        logging.info("fetch data <trip_id>: Show static schedule + live vehicle status")


class print_all_diaries(Command):
    command_name = "print diaries"

    def __init__(self, observatory):
        self._obs = observatory

    def execute(self, args):
        diaries, count_diaries, count_observers = self._obs.get_all_current_diaries()
        for d in diaries:
            logging.info(d)
        logging.info(f"Total Observers: {count_observers}")
        logging.info(f"Total Diaries: {count_diaries}")

    @staticmethod
    def help():
        logging.info("print diaries: Show all active diaries")


# ============================================================
# Commands that emit events (no direct coupling to services)
# ============================================================


class command_quit(Command):
    command_name = "quit"

    def __init__(self):
        pass  # No dependencies needed

    def execute(self, args):
        logging.info("Shutting Down...")
        console_events.emit("shutdown_requested")

    @staticmethod
    def help():
        logging.info("quit: Stop the observer and save data")


class stop_observers(Command):
    command_name = "stop observers"

    def __init__(self):
        pass

    def execute(self, args):
        console_events.emit("services_stop")

    @staticmethod
    def help():
        logging.info(
            "stop observers: Stop collection and auto-save (triggers final save)"
        )


class start_observers(Command):
    command_name = "start observers"

    def __init__(self):
        pass

    def execute(self, args):
        console_events.emit("services_start")

    @staticmethod
    def help():
        logging.info("start observers: Start/Restart collection and auto-save loops")


# ============================================================
# Commands that need complex dependencies (normalize)
# ============================================================


class normalize_diaries(Command):
    command_name = "normalize diaries"

    def __init__(self, normalizer_fn, observatory_factory, parquet_writer):
        """
        Args:
            normalizer_fn: Function to normalize trip stops
            observatory_factory: Callable that returns an Observatory
            parquet_writer: Function to write parquet files
        """
        self._normalize = normalizer_fn
        self._get_observatory = observatory_factory
        self._write_parquet = parquet_writer

    def execute(self, args):
        # Import here to avoid circular deps - this is a batch operation
        from application.post_processing import normalize_diary as nd

        logging.info("=== Diary Normalizer ===")

        diaries = nd.load_all_diaries()
        if diaries is None:
            return

        obs = self._get_observatory()
        ledger = obs.get_ledger()
        if ledger is None:
            return

        logging.info(f"Loaded {len(diaries)} raw measurements.")

        normalized_dfs: list[pd.DataFrame] = []
        grouped = diaries.groupby("trip_id")

        logging.info(f"Processing {len(grouped)} trips...")
        for trip_id, group_df in grouped:
            trip = ledger["trips"].get(trip_id)
            if trip:
                norm_df = nd.normalize_trip_stops(group_df, trip)
                normalized_dfs.append(norm_df)

        final_df = pd.concat(normalized_dfs, ignore_index=True)
        logging.info(
            f"Normalization Complete. Records: {len(diaries)} -> {len(final_df)}"
        )

        logging.info(f"Saving to {nd.OUTPUT_FILE}...")
        self._write_parquet(final_df, nd.OUTPUT_FILE)
        logging.info("Done.")

    @staticmethod
    def help():
        logging.info("normalize diaries: Normalize all diaries")


class command_help(Command):
    command_name = "help"

    def __init__(self):
        pass

    def execute(self, args):
        logging.info("\nAvailable Commands:")
        for subclass in Command.__subclasses__():
            try:
                subclass.help()
            except Exception as e:
                logging.error(f"Error showing help for {subclass.__name__}: {e}")

    @staticmethod
    def help():
        logging.info("help: Show this help message")


class pause_traffic_service(Command):
    command_name = "pause traffic service"

    def __init__(self):
        pass

    def execute(self, args):
        try:
            seconds = int(args.strip())
        except ValueError:
            logging.warning("Usage: pause traffic service <seconds>")
            return

        from application.live import data as live_data

        if live_data.TRAFFIC_SERVICE:
            live_data.TRAFFIC_SERVICE.pause(seconds)
        else:
            logging.warning("Traffic Service is not active/initialized.")

    @staticmethod
    def help():
        logging.info(
            "pause traffic service <seconds>: Pause traffic updates for the specified duration"
        )


# ============================================================
# Validation Commands
# ============================================================


class validate_date(Command):
    command_name = "validate"

    def __init__(self, predictor, observatory):
        self._predictor = predictor
        self._obs = observatory

    def execute(self, args):
        if self._predictor is None:
            logging.warning("No predictor loaded. Start with 'serve' mode or load a model first.")
            return

        date_str = args.strip()
        if not date_str:
            logging.warning("Usage: validate <DD-MM-YYYY>")
            return

        from . import services
        services.start_batch_validation(date_str, self._predictor, self._obs)

    @staticmethod
    def help():
        logging.info("validate <DD-MM-YYYY>: Run batch validation for a date (background thread)")


class validate_live(Command):
    command_name = "validate live"

    def __init__(self, predictor, observatory, bus_type_predictor=None):
        self._predictor = predictor
        self._obs = observatory
        self._bus_type_predictor = bus_type_predictor

    def execute(self, args):
        if self._predictor is None:
            logging.warning("No predictor loaded. Start with 'serve' mode or load a model first.")
            return

        date_str = args.strip()
        if not date_str:
            logging.warning("Usage: validate live <DD-MM-YYYY>")
            return

        from . import services
        services.start_live_validation(
            date_str, self._predictor, self._obs,
            bus_type_predictor=self._bus_type_predictor,
        )

    @staticmethod
    def help():
        logging.info("validate live <DD-MM-YYYY>: Start live validation session (background thread)")


class stop_validation(Command):
    command_name = "stop validation"

    def __init__(self):
        pass

    def execute(self, args):
        from . import services
        services.stop_live_validation()

    @staticmethod
    def help():
        logging.info("stop validation: Stop any running live validation session")


class validation_status(Command):
    command_name = "validation status"

    def __init__(self):
        pass

    def execute(self, args):
        from . import services
        status = services.get_validation_status()

        logging.info("--- Validation Status ---")
        logging.info(f"Batch: {status['batch']}")
        logging.info(f"Live:  {status['live']}")

        if status["live"] != "idle":
            logging.info(f"  Predicted:  {status.get('live_predicted', 0)}")
            logging.info(f"  Validated:  {status.get('live_validated', 0)}")
            logging.info(f"  Pending:    {status.get('live_pending', 0)}")
            logging.info(f"  Discarded:  {status.get('live_discarded', 0)}")
            rmse = status.get('live_median_rmse', 0)
            if rmse > 0:
                logging.info(f"  Median RMSE: {rmse:.2f}s")

    @staticmethod
    def help():
        logging.info("validation status: Show status of running validators")


class print_all_diaries_vehicle(Command):
    command_name = "print diaries vehicle"

    def __init__(self, observatory):
        self._obs = observatory

    def execute(self, args):
        if not args:
            logging.warning("Usage: print diaries vehicle <vehicle_label>")
            return
        vehicle_label = args.strip()
        
        # Try to resolve label to ID
        vehicle_id = self._obs.get_id_by_label(vehicle_label)
        if not vehicle_id:
            # Fallback: maybe the user provided the internal ID directly
            vehicle_id = vehicle_label

        diaries = self._obs.get_vehicle_diaries(vehicle_id)
        if not diaries:
            logging.warning(f"No diaries found for vehicle {vehicle_label} (ID: {vehicle_id})")
            return

        logging.info(f"--- Diaries for Vehicle {vehicle_label} ({len(diaries)}) ---")
        for diary in diaries:
            logging.info(diary)

    @staticmethod
    def help():
        logging.info(
            "print diaries vehicle <vehicle_label>: Show all diaries for a specific vehicle"
        )
