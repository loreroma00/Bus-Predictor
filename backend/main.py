#!/usr/bin/env python
"""Unified backend entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional


def run_collect(debug_mode: bool, lenient_pipeline: bool):
    """Run the GTFS data collection pipeline."""
    from interaction import main as interaction_main

    interaction_main.main(
        debug_mode=debug_mode,
        lenient_pipeline=lenient_pipeline,
    )


def run_serve(
    time_model_name: Optional[str],
    crowd_model_name: Optional[str],
    host: Optional[str],
    port: Optional[int],
    lenient_pipeline: bool = False,
):
    """Create and run the FastAPI server."""
    import uvicorn
    from api import create_app
    from application.runtime import ApplicationContext

    context = ApplicationContext()
    app = create_app(
        time_model_name=time_model_name,
        crowd_model_name=crowd_model_name,
        lenient_pipeline=lenient_pipeline,
        context=context,
    )
    uvicorn.run(
        app,
        host=host or context.config.api.host,
        port=port or context.config.api.port,
    )


def main():
    """Parse CLI arguments and dispatch to the selected backend mode."""
    parser = argparse.ArgumentParser(
        description="ATAC Bus Delay Prediction Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect    Run GTFS real-time data collection pipeline
  serve      Start the FastAPI prediction server
  test-db    Test database connections
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    collect_parser = subparsers.add_parser(
        "collect",
        help="Run data collection pipeline",
    )
    collect_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging",
    )
    collect_parser.add_argument(
        "--lenient-pipeline",
        action="store_true",
        help="Use lenient data cleaning",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start prediction API server with integrated data collection",
    )
    serve_parser.add_argument(
        "--time-model",
        type=str,
        help="TIME model filename (e.g. bus_model_TIME_mse_0.pth)",
    )
    serve_parser.add_argument(
        "--crowd-model",
        type=str,
        help="CROWD model filename (e.g. bus_model_CROWD_mse_0.pth)",
    )
    serve_parser.add_argument("--host", type=str, help="Host to bind")
    serve_parser.add_argument("--port", type=int, help="Port to bind")
    serve_parser.add_argument(
        "--lenient-pipeline",
        action="store_true",
        help="Use lenient data cleaning",
    )

    subparsers.add_parser("test-db", help="Test database connections")

    args = parser.parse_args()

    if args.command == "collect":
        run_collect(
            debug_mode=args.debug,
            lenient_pipeline=args.lenient_pipeline,
        )
    elif args.command == "serve":
        run_serve(
            time_model_name=args.time_model,
            crowd_model_name=args.crowd_model,
            host=args.host,
            port=args.port,
            lenient_pipeline=args.lenient_pipeline,
        )
    elif args.command == "test-db":
        try:
            from application.runtime import ApplicationContext
            from persistence.database import test_database_connection

            context = ApplicationContext()
            asyncio.run(test_database_connection(context.config))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Test failed with error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
