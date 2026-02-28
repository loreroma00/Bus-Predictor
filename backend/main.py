#!/usr/bin/env python
"""
ATAC Bus Delay Prediction Backend

Unified entry point for:
  - Data collection (GTFS real-time)
  - API server (predictions)
  - Database testing

Usage:
  python main.py collect [--debug] [--lenient-pipeline]
  python main.py serve [--model NAME] [--host HOST] [--port PORT]
  python main.py test-db
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent


def run_collect(debug_mode: bool, lenient_pipeline: bool):
    """Run the GTFS data collection pipeline."""
    from interaction import main

    main.main(debug_mode=debug_mode, lenient_pipeline=lenient_pipeline)


async def test_database_connection():
    """Tests database connections defined in config."""
    from config import Prediction, Traffic, Vehicle, load_config, _CONFIG_PATH
    from persistence.database import TimescaleDBConnection
    from datetime import datetime
    import uuid
    from urllib.parse import urlparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("test_db")

    if not _CONFIG_PATH.exists():
        logger.warning(f"Config file not found at: {_CONFIG_PATH}")
    else:
        logger.info(f"Loading config from: {_CONFIG_PATH}")

    config = load_config()

    def log_conn_details(uri):
        try:
            p = urlparse(uri)
            logger.info(
                f"   Target: {p.scheme}://{p.username}:***@{p.hostname}:{p.port}{p.path}"
            )
        except Exception:
            logger.info(f"   Target: [Could not parse connection string]")

    async def test_table_operations(db_conn, table_type, table_name):
        if not await db_conn.connect():
            logger.error(f"Failed to connect for {table_name}")
            return

        logger.info(f"Connection successful for {table_name}")
        pool = await db_conn.get_valid_pool()
        if not pool:
            logger.error(f"Failed to get pool for {table_name}")
            await db_conn.close()
            return

        try:
            async with pool.acquire() as conn:
                mock_id = str(uuid.uuid4())
                ts = datetime.now()

                if table_type == "pred_vec":
                    query = f"""
                        INSERT INTO {table_name} (
                            id, ts, route_id, direction_id, stop_sequence,
                            shape_dist_travelled, distance_to_next_stop, far_status,
                            day_type, rush_hour_status, time_feat, time_sin, time_cos,
                            schedule_adherence, speed_ratio, current_traffic_speed, 
                            current_speed, precipitation, weather_code, bus_type, door_number,
                            deposit_grottarossa, deposit_magliana, deposit_tor_sapienza, deposit_portonaccio,
                            deposit_monte_sacro, deposit_tor_pagnotta, deposit_tor_cervara, deposit_maglianella,
                            deposit_costi, deposit_trastevere, deposit_acilia, deposit_tor_vergata,
                            deposit_porta_maggiore,
                            served_ratio,
                            trip_id,
                            sch_starting_time_cos, sch_starting_time_sin,
                            starting_time_cos, starting_time_sin,
                            delay_genuine
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 
                            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                            $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35,
                            $36, $37, $38, $39, $40, $41
                        )
                    """
                    values = (
                        mock_id,
                        ts,
                        "TEST",
                        0,
                        0,
                        0.0,
                        0.0,
                        False,
                        0,
                        False,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.0,
                        "TEST_TRIP",
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0,
                    )
                elif table_type == "pred_lbl":
                    query = f"""
                        INSERT INTO {table_name} (id, ts, time_seconds, occupancy_status)
                        VALUES ($1, $2, $3, $4)
                    """
                    values = (mock_id, ts, 0, 0)
                elif table_type == "traf_vec":
                    query = f"""
                        INSERT INTO {table_name} (id, ts, day_type, rush_hour_status, time_sin, time_cos, hexagon_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """
                    values = (mock_id, ts, 0, False, 0.0, 0.0, "TEST_HEX")
                elif table_type == "traf_lbl":
                    query = f"""
                        INSERT INTO {table_name} (id, ts, speed_ratio, current_traffic_speed)
                        VALUES ($1, $2, $3, $4)
                    """
                    values = (mock_id, ts, 0.0, 0.0)
                elif table_type == "veh_vec":
                    query = f"""
                        INSERT INTO {table_name} (id, ts, route_id, trip_id, direction_id)
                        VALUES ($1, $2, $3, $4, $5)
                    """
                    values = (mock_id, ts, "TEST_ROUTE", "TEST_TRIP", 0)
                elif table_type == "veh_lbl":
                    query = f"""
                        INSERT INTO {table_name} (id, ts, vehicle_type)
                        VALUES ($1, $2, $3)
                    """
                    values = (mock_id, ts, "TEST_TYPE")
                else:
                    logger.warning(f"Unknown table type {table_type}")
                    return

                try:
                    await conn.execute(query, *values)
                    logger.info(f"   Mock insertion successful")
                except Exception as e:
                    logger.error(f"   Insertion failed: {e}")
                    return

                del_query = f"DELETE FROM {table_name} WHERE id = $1"
                try:
                    await conn.execute(del_query, mock_id)
                    logger.info(f"   Mock deletion successful")
                except Exception as e:
                    logger.error(f"   Deletion failed: {e}")

        except Exception as e:
            logger.error(f"Unexpected error during test op: {e}")
        finally:
            await db_conn.close()

    if Prediction.ENABLED:
        logger.info("\nTesting Prediction Database...")
        pred_section = config.get("prediction", {})
        logger.info(f"   Loaded Keys: {list(pred_section.keys())}")
        if "vector_db_connection" in pred_section:
            log_conn_details(pred_section["vector_db_connection"])
        else:
            logger.warning(
                "   'vector_db_connection' NOT found in [prediction] section!"
            )

        conn_str = Prediction.VECTOR_DB_CONNECTION
        table_vec = Prediction.VECTOR_TABLE
        table_lbl = Prediction.LABEL_TABLE

        logger.info(f"   Resolved Connection String:")
        log_conn_details(conn_str)

        if conn_str:
            if table_vec:
                logger.info(f"   Vector Table: '{table_vec}'")
                db_vec = TimescaleDBConnection(conn_str, table_vec)
                await test_table_operations(db_vec, "pred_vec", table_vec)
            else:
                logger.warning("Prediction vector table not defined")

            if table_lbl:
                logger.info(f"   Label Table: '{table_lbl}'")
                db_lbl = TimescaleDBConnection(conn_str, table_lbl)
                await test_table_operations(db_lbl, "pred_lbl", table_lbl)
            else:
                logger.warning("Prediction label table not defined")
    else:
        logger.info("\nPrediction pipeline disabled.")

    if Traffic.ENABLED:
        logger.info("\nTesting Traffic Database...")
        conn_str = Traffic.VECTOR_DB_CONNECTION
        table_vec = Traffic.VECTOR_TABLE
        table_lbl = Traffic.LABEL_TABLE
        log_conn_details(conn_str)

        if conn_str:
            if table_vec:
                logger.info(f"   Vector Table: '{table_vec}'")
                db_vec = TimescaleDBConnection(conn_str, table_vec)
                await test_table_operations(db_vec, "traf_vec", table_vec)
            else:
                logger.warning("Traffic vector table not defined")

            if table_lbl:
                logger.info(f"   Label Table: '{table_lbl}'")
                db_lbl = TimescaleDBConnection(conn_str, table_lbl)
                await test_table_operations(db_lbl, "traf_lbl", table_lbl)
            else:
                logger.warning("Traffic label table not defined")
    else:
        logger.info("\nTraffic pipeline disabled.")

    if Vehicle.ENABLED:
        logger.info("\nTesting Vehicle Database...")
        conn_str = Vehicle.VECTOR_DB_CONNECTION
        table_vec = Vehicle.VECTOR_TABLE
        table_lbl = Vehicle.LABEL_TABLE
        log_conn_details(conn_str)

        if conn_str:
            if table_vec:
                logger.info(f"   Vector Table: '{table_vec}'")
                db_vec = TimescaleDBConnection(conn_str, table_vec)
                await test_table_operations(db_vec, "veh_vec", table_vec)
            else:
                logger.warning("Vehicle vector table not defined")

            if table_lbl:
                logger.info(f"   Label Table: '{table_lbl}'")
                db_lbl = TimescaleDBConnection(conn_str, table_lbl)
                await test_table_operations(db_lbl, "veh_lbl", table_lbl)
            else:
                logger.warning("Vehicle label table not defined")
    else:
        logger.info("\nVehicle pipeline disabled.")


def run_serve(model_name: Optional[str], host: Optional[str], port: Optional[int]):
    """Run the FastAPI prediction server."""
    import configparser
    import glob
    from datetime import datetime
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, field_validator

    sys.path.insert(0, str(PROJECT_ROOT / "application" / "model"))
    from predictor import Predictor, StopPrediction, TripForecast

    MODEL_DIR = PROJECT_ROOT / "application" / "model"
    CONFIG_PATH = PROJECT_ROOT / "config.ini"

    class PredictRequest(BaseModel):
        route_id: str
        direction_id: int
        start_date: str
        start_time: str
        weather_code: int
        bus_type: int

        @field_validator("start_date")
        @classmethod
        def validate_date(cls, v):
            try:
                datetime.strptime(v, "%d-%m-%Y")
                return v
            except ValueError:
                raise ValueError("start_date must be in DD-MM-YYYY format")

        @field_validator("start_time")
        @classmethod
        def validate_time(cls, v):
            try:
                datetime.strptime(v, "%H:%M")
                return v
            except ValueError:
                raise ValueError("start_time must be in HH:MM format")

    class StopPredictionResponse(BaseModel):
        stop_sequence: int
        distance_m: float
        cumulative_delay_sec: float
        delay_formatted: str
        expected_arrival: str
        crowd_level: int

    class PredictResponse(BaseModel):
        route_id: str
        direction_id: int
        trip_date: str
        scheduled_start: str
        stops: list[StopPredictionResponse]

    class ModelInfo(BaseModel):
        filename: str
        path: str

    predictor: Optional[Predictor] = None
    available_models: list[ModelInfo] = []

    def load_config_api() -> dict:
        config = configparser.ConfigParser()
        if CONFIG_PATH.exists():
            config.read(CONFIG_PATH)
        return {
            "frontend_url": config.get(
                "api", "frontend_url", fallback="http://localhost:3000"
            ),
            "host": config.get("api", "host", fallback="0.0.0.0"),
            "port": config.getint("api", "port", fallback=8000),
        }

    def discover_models() -> list[ModelInfo]:
        pattern = str(MODEL_DIR / "bus_model_*.pth")
        model_files = glob.glob(pattern)
        models = []
        for path in sorted(model_files):
            models.append(ModelInfo(filename=os.path.basename(path), path=path))
        return models

    def load_model_by_name(model_name_arg: str) -> bool:
        nonlocal predictor
        model_path = MODEL_DIR / model_name_arg
        if not model_path.exists():
            print(f"[ERROR] Model file not found: {model_name_arg}")
            return False

        config_filename = model_name_arg.replace(
            "bus_model_", "hyperparameters_"
        ).replace(".pth", ".json")
        config_path = MODEL_DIR / config_filename

        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_filename}")
            return False

        predictor = Predictor(
            weights_path=str(model_path), config_path=str(config_path)
        )
        return True

    def interactive_model_selection():
        nonlocal predictor
        print("\n" + "=" * 50)
        print("ATAC Bus Delay Prediction - Model Selection")
        print("=" * 50)
        print("\nAvailable models:")
        for i, m in enumerate(available_models):
            print(f"  [{i}] {m.filename}")

        while True:
            try:
                choice = input("\nSelect model number: ").strip()
                choice_idx = int(choice)
                if 0 <= choice_idx < len(available_models):
                    break
                print(
                    f"Please enter a number between 0 and {len(available_models) - 1}"
                )
            except (ValueError, EOFError):
                print("Invalid input. Please enter a number.")

        selected_model = available_models[choice_idx]
        return load_model_by_name(selected_model.filename)

    api_config = load_config_api()

    app = FastAPI(
        title="ATAC Bus Delay Prediction API",
        description="Predict bus delays for routes in Rome",
        version="1.0.0",
    )

    frontend_url = api_config["frontend_url"]
    allow_origins = ["*"] if frontend_url == "*" else [frontend_url]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        nonlocal predictor, available_models
        available_models = discover_models()

        if not available_models:
            print("\n[ERROR] No trained models found in application/model/")
            print("Expected files matching: bus_model_*.pth")
            sys.exit(1)

        cli_model = model_name or os.environ.get("MODEL_NAME")
        if cli_model:
            if not load_model_by_name(cli_model):
                sys.exit(1)
        else:
            if not interactive_model_selection():
                sys.exit(1)

        print(f"\nCORS enabled for: {api_config['frontend_url']}")
        print(f"Server ready. API documentation at: /docs")
        print("=" * 50 + "\n")

    @app.get("/models", response_model=list[ModelInfo])
    async def list_models():
        return available_models

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            forecast: TripForecast = predictor.get_trip_forecast(
                route_id=request.route_id,
                direction_id=request.direction_id,
                start_date=request.start_date,
                start_time=request.start_time,
                weather_code=request.weather_code,
                bus_type=request.bus_type,
            )

            stops = [
                StopPredictionResponse(
                    stop_sequence=s.stop_sequence,
                    distance_m=s.distance_m,
                    cumulative_delay_sec=s.cumulative_delay_sec,
                    delay_formatted=s.delay_formatted,
                    expected_arrival=s.expected_arrival,
                    crowd_level=s.crowd_level,
                )
                for s in forecast.stops
            ]

            return PredictResponse(
                route_id=forecast.route_id,
                direction_id=forecast.direction_id,
                trip_date=forecast.trip_date,
                scheduled_start=forecast.scheduled_start,
                stops=stops,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    @app.get("/health")
    async def health():
        return {"status": "healthy", "model_loaded": predictor is not None}

    import uvicorn

    final_host = host or api_config["host"]
    final_port = port or api_config["port"]
    uvicorn.run(app, host=final_host, port=final_port)


def main():
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
        "collect", help="Run data collection pipeline"
    )
    collect_parser.add_argument(
        "--debug", action="store_true", help="Enable verbose logging"
    )
    collect_parser.add_argument(
        "--lenient-pipeline", action="store_true", help="Use lenient data cleaning"
    )

    serve_parser = subparsers.add_parser("serve", help="Start prediction API server")
    serve_parser.add_argument("--model", type=str, help="Model filename to load")
    serve_parser.add_argument("--host", type=str, help="Host to bind")
    serve_parser.add_argument("--port", type=int, help="Port to bind")

    subparsers.add_parser("test-db", help="Test database connections")

    args = parser.parse_args()

    if args.command == "collect":
        run_collect(debug_mode=args.debug, lenient_pipeline=args.lenient_pipeline)
    elif args.command == "serve":
        run_serve(model_name=args.model, host=args.host, port=args.port)
    elif args.command == "test-db":
        try:
            asyncio.run(test_database_connection())
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
