"""Collect-mode entry point for the interactive backend."""

import logging

from bootstrapper import (
    build_runtime_context,
    configure_logging,
    shutdown_runtime,
    start_collection_services,
    wire_state_interface,
)
from . import console


def main(debug_mode: bool = False, lenient_pipeline: bool = False):
    """Entry point for ``collect``: bootstraps logging, services, and the observation loop (plus debug GUI)."""
    configure_logging(debug_mode=debug_mode)
    context = None

    try:
        context = build_runtime_context(
            lenient_pipeline=lenient_pipeline,
        )

        wire_state_interface(context)
        start_collection_services(context)

        # Start Interactive Console (Main Thread)
        console.run_console_loop()
    except KeyboardInterrupt:
        logging.info("\nStopping Observer (User Interrupt)...")
    except Exception as e:
        logging.exception(f"Unexpected Error: {e}")
        logging.error("Attempting failsafe save...")
    finally:
        logging.info("Archiving Diaries...")
        shutdown_runtime(context, save_diaries=True, join_services=True)


if __name__ == "__main__":
    main()
