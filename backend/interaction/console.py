"""
Console - Interactive command loop.

Commands are registered with their dependencies at startup.
"""

import logging
from application import domain
from . import commands

# ============================================================
# Command Registry - instantiated commands with dependencies
# ============================================================
_command_registry = {}


def register_commands(observatory, predictor=None, bus_type_predictor=None):
    """
    Register all commands with their dependencies.
    Called once at application startup.
    """
    global _command_registry

    _command_registry = {
        # Commands that need observatory
        "debug traffic": commands.debug_traffic(observatory),
        "print hex": commands.print_hex(observatory),
        "print live trip": commands.print_live_trip(observatory),
        "fetch data": commands.fetch_data(observatory, domain.to_readable_time),
        "print vehicle trips": commands.print_vehicle_live_trips(observatory),
        "print live trips": commands.print_all_live_trips(observatory),
        # Commands that emit events (no deps)
        "quit": commands.command_quit(),
        "stop services": commands.stop_services(),
        "start services": commands.start_services(),
        "stop validation": commands.stop_validation(),
        "validation status": commands.validation_status(),
        "help": commands.command_help(),
        "pause traffic service": commands.pause_traffic_service(observatory),
        "weather strategy": commands.weather_strategy_cmd(observatory),
        "trip validation chart": commands.trip_validation_chart(observatory, predictor),
        # Validation commands (longer prefix first for correct matching)
        "validate_live": commands.validate_live(
            predictor, observatory, bus_type_predictor
        ),
        "validate": commands.validate_date(predictor, observatory),
    }


def run_console_loop(shutdown_event=None):
    """Main thread function for user interaction."""
    logging.info("\n--- INTERACTIVE CONSOLE ---")
    logging.info("Commands: 'print live trip <trip_id>', 'fetch data <trip_id>', 'quit'")

    while shutdown_event is None or not shutdown_event.is_set():
        try:
            cmd = input("Command> ").strip()
            if cmd == "":
                continue

            # Find matching command
            matched = False
            for cmd_name in sorted(_command_registry.keys(), key=len, reverse=True):
                command_instance = _command_registry[cmd_name]
                if cmd.lower().startswith(cmd_name):
                    args = cmd[len(cmd_name) :].strip()
                    command_instance.execute(args)
                    matched = True
                    break

            if not matched:
                logging.warning("Unknown command. Type 'help' for available commands.")

        except EOFError:
            if shutdown_event is not None:
                shutdown_event.set()
            break
        except Exception as e:
            logging.error(f"Console Error: {e}")
            logging.exception("Traceback:")
