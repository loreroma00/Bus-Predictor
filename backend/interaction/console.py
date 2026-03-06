"""
Console - Interactive command loop.

Commands are registered with their dependencies at startup.
"""

import logging
import traceback
from application.live import data
from application import domain
from persistence import writeParquet
from . import commands


# ============================================================
# Command Registry - instantiated commands with dependencies
# ============================================================
_command_registry = {}


def register_commands(observatory, predictor=None, weather_service=None, bus_type_predictor=None):
    """
    Register all commands with their dependencies.
    Called once at application startup.
    """
    global _command_registry

    _command_registry = {
        # Commands that need observatory
        "debug traffic": commands.debug_traffic(observatory),
        "print hex": commands.print_hex(observatory),
        "print diary": commands.print_diary(observatory),
        "fetch data": commands.fetch_data(observatory, domain.to_readable_time),
        "print diaries vehicle": commands.print_all_diaries_vehicle(observatory),
        "print diaries": commands.print_all_diaries(observatory),
        # Commands that emit events (no deps)
        "quit": commands.command_quit(),
        "stop observers": commands.stop_observers(),
        "start observers": commands.start_observers(),
        "stop validation": commands.stop_validation(),
        "validation status": commands.validation_status(),
        "help": commands.command_help(),
        "pause traffic service": commands.pause_traffic_service(),
        # Validation commands (longer prefix first for correct matching)
        "validate live": commands.validate_live(predictor, observatory, weather_service, bus_type_predictor),
        "validate": commands.validate_date(predictor, observatory, weather_service),
        # Complex commands
        "normalize diaries": commands.normalize_diaries(
            normalizer_fn=None,  # Uses internal import
            observatory_factory=lambda: domain.Observatory(),
            parquet_writer=writeParquet,
        ),
    }


def run_console_loop():
    """Main thread function for user interaction."""
    logging.info("\n--- INTERACTIVE CONSOLE ---")
    logging.info("Commands: 'print diary <trip_id>', 'fetch data <trip_id>', 'quit'")

    while not data.SHUTDOWN_EVENT.is_set():
        try:
            cmd = input("Command> ").strip()
            if cmd == "":
                continue

            # Find matching command
            matched = False
            for cmd_name, command_instance in _command_registry.items():
                if cmd.lower().startswith(cmd_name):
                    args = cmd[len(cmd_name) :].strip()
                    command_instance.execute(args)
                    matched = True
                    break

            if not matched:
                logging.warning("Unknown command. Type 'help' for available commands.")

        except EOFError:
            data.SHUTDOWN_EVENT.set()
            break
        except Exception as e:
            logging.error(f"Console Error: {e}")
            logging.exception("Traceback:")
