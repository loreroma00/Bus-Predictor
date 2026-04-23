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
        """Log conn details."""
        try:
            p = urlparse(uri)
            logger.info(
                f"   Target: {p.scheme}://{p.username}:***@{p.hostname}:{p.port}{p.path}"
            )
        except Exception:
            logger.info(f"   Target: [Could not parse connection string]")

    async def test_table_operations(db_conn, table_type, table_name):
        """Test: table operations."""
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


def run_serve(time_model_name: Optional[str], crowd_model_name: Optional[str], host: Optional[str], port: Optional[int], lenient_pipeline: bool = False):
    """Run the FastAPI prediction server with integrated data collection."""
    import configparser
    import glob
    import asyncio
    import uuid
    from datetime import datetime
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, field_validator

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    sys.path.insert(0, str(PROJECT_ROOT / "application" / "model"))
    from predictor import Predictor, StopPrediction, TripForecast

    MODEL_DIR = PROJECT_ROOT / "application" / "model"
    CONFIG_PATH = PROJECT_ROOT / "config.ini"
    PARQUET_DIR = PROJECT_ROOT / "parquets"

    class PredictRequest(BaseModel):
        """Predictrequest."""
        route_id: str
        direction_id: int
        start_date: str
        start_time: str
        weather_code: int
        bus_type: int

        @field_validator("start_date")
        @classmethod
        def validate_date(cls, v):
            """Validate date."""
            try:
                datetime.strptime(v, "%d-%m-%Y")
                return v
            except ValueError:
                raise ValueError("start_date must be in DD-MM-YYYY format")

        @field_validator("start_time")
        @classmethod
        def validate_time(cls, v):
            """Validate time."""
            try:
                datetime.strptime(v, "%H:%M")
                return v
            except ValueError:
                raise ValueError("start_time must be in HH:MM format")

    class StopPredictionResponse(BaseModel):
        """Stoppredictionresponse."""
        stop_sequence: int
        distance_m: float
        cumulative_delay_sec: float
        delay_formatted: str
        expected_arrival: str
        crowd_level: int

    class PredictResponse(BaseModel):
        """Predictresponse."""
        route_id: str
        direction_id: int
        trip_date: str
        scheduled_start: str
        stops: list[StopPredictionResponse]

    class StopPredictionWithInfo(BaseModel):
        """Stoppredictionwithinfo."""
        stop_sequence: int
        stop_id: str
        stop_name: str
        stop_lat: float
        stop_lon: float
        predicted_arrival: str
        delay_seconds: float
        confidence_rating: Optional[float] = None

    class PredictedTrip(BaseModel):
        """Predictedtrip."""
        trip_id: str
        route_id: str
        direction_id: int
        trip_date: str
        scheduled_start: str
        weather_code: int
        bus_type: int
        stop_sequence: dict[int, StopPredictionWithInfo]

    class BatchPredictRequest(BaseModel):
        """Batchpredictrequest."""
        trips: list[PredictRequest]

    class FailedTrip(BaseModel):
        """Failedtrip."""
        index: int
        route_id: str
        direction_id: int
        start_time: str
        error: str

    class BatchPredictResponse(BaseModel):
        """Batchpredictresponse."""
        successful: list[PredictedTrip]
        failed: list[FailedTrip]
        total_requested: int
        total_successful: int
        total_failed: int

    class ModelInfo(BaseModel):
        """Modelinfo."""
        filename: str
        path: str

    class WeatherResponse(BaseModel):
        """Weatherresponse."""
        temperature: float
        apparent_temperature: float
        humidity: float
        precip_intensity: float
        wind_speed: float
        weather_code: int
        description: str

    class RouteSummary(BaseModel):
        """Routesummary."""
        route_id: str

    class RoutesResponse(BaseModel):
        """Routesresponse."""
        routes: list[RouteSummary]

    class DirectionInfo(BaseModel):
        """Directioninfo."""
        direction_id: int
        trip_headsign: str

    class DirectionsResponse(BaseModel):
        """Directionsresponse."""
        route_id: str
        directions: list[DirectionInfo]

    class ShapePoint(BaseModel):
        """Shapepoint."""
        lat: float
        lon: float
        dist: float

    class StopInfo(BaseModel):
        """Stopinfo."""
        stop_id: str
        stop_name: str
        stop_lat: float
        stop_lon: float
        stop_sequence: int
        shape_dist_traveled: Optional[float] = None

    class TripSchedule(BaseModel):
        """Tripschedule."""
        trip_id: str
        start_time: str

    class RouteDirectionResponse(BaseModel):
        """Routedirectionresponse."""
        route_id: str
        direction_id: int
        trip_headsign: str
        shape: list[ShapePoint]
        stops: list[StopInfo]
        schedule: list[TripSchedule]

    class ValidateRequest(BaseModel):
        """Validaterequest."""
        date: str

        @field_validator("date")
        @classmethod
        def validate_date(cls, v):
            """Validate date."""
            try:
                datetime.strptime(v, "%d-%m-%Y")
                return v
            except ValueError:
                raise ValueError("date must be in DD-MM-YYYY format")

    class TripValidationSummary(BaseModel):
        """Tripvalidationsummary."""
        trip_id: str
        route_id: str
        direction_id: int
        scheduled_start: str
        mse: float
        rmse: float
        n_measurements: int
        error: Optional[str] = None

    class ValidateResponse(BaseModel):
        """Validateresponse."""
        date: str
        total_scheduled_trips: int
        total_trips_with_ground_truth: int
        total_trips_predicted: int
        total_trips_validated: int
        total_measurements: int
        median_mse: float
        median_rmse: float
        min_mse: float
        max_mse: float
        min_rmse: float
        max_rmse: float
        occupancy_confusion_matrix: list[list[int]]
        trips: list[TripValidationSummary]
        log_file: str
        report_file: str

    class LiveValidateScheduleRequest(BaseModel):
        """Livevalidateschedulerequest."""
        date: str

        @field_validator("date")
        @classmethod
        def validate_date(cls, v):
            """Validate date."""
            try:
                datetime.strptime(v, "%d-%m-%Y")
                return v
            except ValueError:
                raise ValueError("date must be in DD-MM-YYYY format")

    class LiveValidateScheduleResponse(BaseModel):
        """Livevalidatescheduleresponse."""
        session_id: str
        date: str
        total_scheduled: int
        total_predicted: int
        status: str
        started_at: str
        stops_at: Optional[str] = None

    class LiveValidateStatusResponse(BaseModel):
        """Livevalidatestatusresponse."""
        session_id: Optional[str] = None
        date: Optional[str] = None
        status: Optional[str] = None
        total_scheduled: int = 0
        total_predicted: int = 0
        total_validated: int = 0
        total_pending: int = 0
        total_discarded: int = 0
        median_mse: float = 0.0
        median_rmse: float = 0.0
        min_mse: float = 0.0
        max_mse: float = 0.0
        min_rmse: float = 0.0
        max_rmse: float = 0.0
        started_at: Optional[str] = None
        stops_at: Optional[str] = None
        log_file: Optional[str] = None
        report_file: Optional[str] = None

    predictor: Optional[Predictor] = None
    available_models: list[ModelInfo] = []
    observatory = None
    city = None

    current_live_session = None
    bus_type_predictor = None

    def _record_predictions(forecast: TripForecast, stops_map: dict):
        """Record a forecast into the PredictedLedger (fire-and-forget)."""
        if observatory is None:
            return
        try:
            import time as _time
            from application.domain.ledgers import StopPredictionRecord

            now = _time.time()
            records = []
            for sp in forecast.stops:
                stop_info = stops_map.get(sp.stop_sequence, {})
                records.append(
                    StopPredictionRecord(
                        route_id=forecast.route_id,
                        direction_id=forecast.direction_id,
                        trip_date=forecast.trip_date,
                        scheduled_start=forecast.scheduled_start,
                        stop_id=stop_info.get("stop_id", ""),
                        stop_sequence=sp.stop_sequence,
                        predicted_arrival=sp.expected_arrival,
                        predicted_delay_sec=sp.cumulative_delay_sec,
                        predicted_crowd_level=sp.crowd_level,
                        prediction_timestamp=now,
                    )
                )
            observatory.predicted.record_predictions(records)
        except Exception as e:
            logging.warning(f"Failed to record predictions: {e}")

    def _get_current_weather_code() -> int:
        """Best-effort: grab the weather code from any city hexagon."""
        try:
            if city is None:
                return 0
            for hexagon in city.hexagons.values():
                w = getattr(hexagon, "weather", None)
                if w is not None:
                    return int(w.weather_code)
        except Exception:
            pass
        return 0

    def _reconstruct_from_ledger(
        df,
        topology,
        route_id: str,
        direction_id: int,
        weather_code: int,
        bus_type: int,
    ) -> "PredictedTrip":
        """Build a PredictedTrip response from cached ledger rows + topology."""
        stops_map = topology.build_stops_map(route_id, direction_id)
        stop_sequence = {}
        for _, row in df.iterrows():
            seq = int(row["stop_sequence"])
            stop_info = stops_map.get(seq, {})
            stop_sequence[seq] = StopPredictionWithInfo(
                stop_sequence=seq,
                stop_id=str(row.get("stop_id") or stop_info.get("stop_id", "")),
                stop_name=stop_info.get("stop_name", ""),
                stop_lat=stop_info.get("stop_lat", 0),
                stop_lon=stop_info.get("stop_lon", 0),
                predicted_arrival=str(row["predicted_arrival"]),
                delay_seconds=float(row["predicted_delay_sec"]),
                confidence_rating=None,
            )
        first = df.iloc[0]
        # trip_date comes back from DB as a date object — reformat to DD-MM-YYYY
        td = first["trip_date"]
        if hasattr(td, "strftime"):
            trip_date_str = td.strftime("%d-%m-%Y")
        else:
            trip_date_str = str(td)
        return PredictedTrip(
            route_id=str(first["route_id"]),
            direction_id=int(first["direction_id"]),
            trip_date=trip_date_str,
            scheduled_start=str(first["scheduled_start"]),
            weather_code=weather_code,
            bus_type=bus_type,
            stop_sequence=stop_sequence,
        )

    async def generate_daily_predictions(date_yyyymmdd: str):
        """Pre-generate and cache predictions for every trip on the given date.

        Runs entirely in the background — failures per trip are silently skipped.
        date_yyyymmdd: 'YYYYMMDD' (matches the format used in Trip.dates).
        """
        if predictor is None or observatory is None:
            return

        from datetime import datetime as _dt
        date_obj = _dt.strptime(date_yyyymmdd, "%Y%m%d").date()
        start_date_fmt = date_obj.strftime("%d-%m-%Y")    # predictor format
        trip_date_iso  = date_obj.strftime("%Y-%m-%d")    # SQL format

        topology = observatory.get_topology()
        weather_code = _get_current_weather_code()

        # Lazy-init bus_type predictor
        nonlocal bus_type_predictor
        if bus_type_predictor is None:
            try:
                from application.services.bus_type_predictor import BusTypePredictor
                bus_type_predictor = BusTypePredictor()
            except Exception:
                pass

        # Pre-fetch already-cached trips so we can skip them
        try:
            cached_df = observatory.predicted.query(trip_date=start_date_fmt)
            already_cached: set[tuple] = set()
            if not cached_df.empty:
                for _, row in cached_df.iterrows():
                    key = (str(row["route_id"]), int(row["direction_id"]),
                           str(row["scheduled_start"])[:5])
                    already_cached.add(key)
        except Exception:
            already_cached = set()

        # Collect unique (route_id, direction_id, HH:MM) for this date
        seen: set[tuple] = set()
        trips_to_run: list[tuple] = []
        for trip in list(topology.trips.values()):
            if date_yyyymmdd not in (trip.dates or []):
                continue
            if not trip.stop_times:
                continue
            arrival = trip.stop_times[0].get("arrival_time", "")
            if not arrival:
                continue
            start_time = arrival[:5]       # "HH:MM"
            route_id   = trip.route.id
            direction_id = int(trip.direction_id or 0)
            key = (route_id, direction_id, start_time)
            if key in seen or key in already_cached:
                continue
            seen.add(key)
            trips_to_run.append(key)

        logging.info(
            f"[pre-gen] {date_yyyymmdd}: {len(trips_to_run)} trips to predict "
            f"({len(already_cached)} already cached)"
        )

        generated = errors = 0
        for route_id, direction_id, start_time in trips_to_run:
            try:
                bt = 0
                if bus_type_predictor is not None:
                    try:
                        bt = int(bus_type_predictor.predict(
                            route_id=route_id,
                            start_time=start_time,
                            trip_date=date_obj,
                        ) or 0)
                    except Exception:
                        pass

                forecast = await asyncio.to_thread(
                    predictor.get_trip_forecast,
                    route_id, direction_id, start_date_fmt, start_time,
                    weather_code, bt,
                )
                stops_map = topology.build_stops_map(route_id, direction_id)
                _record_predictions(forecast, stops_map)
                generated += 1
            except Exception as e:
                errors += 1
                logging.debug(
                    f"[pre-gen] skip {route_id}/{direction_id}/{start_time}: {e}"
                )

            # Yield every 20 trips so we don't starve the event loop
            if (generated + errors) % 20 == 0:
                await asyncio.sleep(0)

        logging.info(f"[pre-gen] done: {generated} generated, {errors} skipped")

    def load_config_api() -> dict:
        """Load the config api."""
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
        """Discover paired TIME+CROWD model sets from bus_model_TIME_*.pth files."""
        pattern = str(MODEL_DIR / "bus_model_TIME_*.pth")
        time_files = glob.glob(pattern)
        models = []
        for time_path in sorted(time_files):
            time_filename = os.path.basename(time_path)
            # Derive experiment id: bus_model_TIME_mse_0.pth -> mse_0
            exp_id = time_filename.replace("bus_model_TIME_", "").replace(".pth", "")
            crowd_filename = f"bus_model_CROWD_{exp_id}.pth"
            crowd_path = MODEL_DIR / crowd_filename
            config_filename = f"hyperparameters_DUAL_{exp_id}.json"
            config_path = MODEL_DIR / config_filename
            if crowd_path.exists() and config_path.exists():
                models.append(ModelInfo(filename=exp_id, path=time_path))
            else:
                missing = []
                if not crowd_path.exists(): missing.append(crowd_filename)
                if not config_path.exists(): missing.append(config_filename)
                print(f"[WARN] Skipping {time_filename}: missing {', '.join(missing)}")
        return models

    _loaded_model_name = None  # Track which model is loaded for GUI display

    def load_model_pair(time_name_arg: str, crowd_name_arg: str) -> bool:
        """Load a TIME + CROWD model pair into the predictor."""
        nonlocal predictor, _loaded_model_name
        time_path = MODEL_DIR / time_name_arg
        crowd_path = MODEL_DIR / crowd_name_arg
        if not time_path.exists():
            print(f"[ERROR] TIME model not found: {time_name_arg}")
            return False
        if not crowd_path.exists():
            print(f"[ERROR] CROWD model not found: {crowd_name_arg}")
            return False

        # Derive config from the TIME model filename
        exp_id = time_name_arg.replace("bus_model_TIME_", "").replace(".pth", "")
        config_filename = f"hyperparameters_DUAL_{exp_id}.json"
        config_path = MODEL_DIR / config_filename
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_filename}")
            return False

        predictor = Predictor(
            config_path=str(config_path),
            time_weights_path=str(time_path),
            crowd_weights_path=str(crowd_path),
        )
        _loaded_model_name = exp_id
        return True

    def load_model_by_exp_id(exp_id: str) -> bool:
        """Load a model pair given an experiment id (e.g. 'mse_0')."""
        return load_model_pair(
            f"bus_model_TIME_{exp_id}.pth",
            f"bus_model_CROWD_{exp_id}.pth",
        )

    def interactive_model_selection():
        """Interactive model selection."""
        nonlocal predictor
        print("\n" + "=" * 50)
        print("ATAC Bus Delay Prediction - Model Selection")
        print("=" * 50)
        print("\nAvailable model pairs:")
        for i, m in enumerate(available_models):
            print(f"  [{i}] {m.filename}")
            print(f"       TIME:  bus_model_TIME_{m.filename}.pth")
            print(f"       CROWD: bus_model_CROWD_{m.filename}.pth")

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

        selected = available_models[choice_idx]
        return load_model_by_exp_id(selected.filename)

    def generate_canonical_map():
        """Generate canonical map."""
        if not (PARQUET_DIR / "stop_route_map.parquet").exists():
            print("Generating stop_route_map.parquet...")
            from application.preprocessing.canonical_shape_mapper import (
                main as gen_main,
            )

            gen_main()

    async def periodic_ledger_check():
        """Periodic ledger check."""
        from application.services.shared_state import check_for_updates
        from datetime import datetime as _dt

        while True:
            await asyncio.sleep(3600)
            if check_for_updates():
                print("Ledger updated with new GTFS data")
                today = _dt.now().strftime("%Y%m%d")
                asyncio.create_task(generate_daily_predictions(today))

    api_config = load_config_api()

    app = FastAPI(
        title="ATAC Bus Delay Prediction API",
        description="Predict bus delays for routes in Rome",
        version="2.0.0",
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
        """Startup event."""
        nonlocal predictor, available_models, observatory, city

        print("\n" + "=" * 50)
        print("Initializing ATAC Backend...")
        print("=" * 50)

        print("\n[1/6] Initializing collection pipeline (Observatory, City, services)...")
        from interaction.main import initialize_collection
        from interaction import services as ingestor_services

        observatory, city, collection_config = initialize_collection(
            lenient_pipeline=lenient_pipeline,
        )

        print("\n[2/6] Generating canonical route map (if needed)...")
        generate_canonical_map()

        print("\n[4/6] Loading ML model...")
        available_models = discover_models()

        if not available_models:
            print("\n[ERROR] No trained model pairs found in application/model/")
            print("Expected files matching: bus_model_TIME_*.pth + bus_model_CROWD_*.pth + hyperparameters_DUAL_*.json")
            sys.exit(1)

        # Resolve model pair: CLI flags > env vars > interactive selection
        cli_time  = time_model_name  or os.environ.get("TIME_MODEL_NAME")
        cli_crowd = crowd_model_name or os.environ.get("CROWD_MODEL_NAME")
        if cli_time and cli_crowd:
            if not load_model_pair(cli_time, cli_crowd):
                sys.exit(1)
        elif cli_time:
            # Derive crowd model from time model name
            exp_id = cli_time.replace("bus_model_TIME_", "").replace(".pth", "")
            if not load_model_by_exp_id(exp_id):
                sys.exit(1)
        else:
            if not interactive_model_selection():
                sys.exit(1)

        print("\n[5/7] Starting data collection services...")
        from interaction.state_interface import StateInterface
        state_interface = StateInterface(observatory)
        if predictor is not None and _loaded_model_name:
            state_interface.set_predictor_info(_loaded_model_name)
        ingestor_services.set_state_interface(state_interface)
        ingestor_services.start_services(observatory, collection_config)

        print("\n[6/7] Starting background tasks...")
        asyncio.create_task(periodic_ledger_check())

        from datetime import datetime as _dt
        today = _dt.now().strftime("%Y%m%d")
        asyncio.create_task(generate_daily_predictions(today))

        print("\n[7/7] Starting interactive console...")
        from interaction import console
        import threading

        console.register_commands(
            observatory,
            predictor=predictor,
            bus_type_predictor=bus_type_predictor,
        )
        state_interface.set_command_registry(console._command_registry)
        threading.Thread(
            target=console.run_console_loop,
            daemon=True,
        ).start()

        print(f"\nCORS enabled for: {api_config['frontend_url']}")
        print(f"Server ready with integrated data collection.")
        print(f"API documentation at: /docs")
        print("=" * 50 + "\n")

    @app.get("/models", response_model=list[ModelInfo])
    async def list_models():
        """List models."""
        return available_models

    @app.get("/weather", response_model=WeatherResponse)
    async def get_weather(lat: float = None, lon: float = None, hex_id: str = None):
        """Get current weather from hexagons (updated every 15 min by weather thread)."""
        try:
            if city is None:
                raise HTTPException(status_code=503, detail="City not initialized")

            hexagon = None
            if hex_id:
                hexagon = city.get_hexagon(hex_id)
            elif lat is not None and lon is not None:
                target_hex = city.get_hex_id(lat, lon)
                hexagon = city.get_hexagon(target_hex)

            # Get weather from target hexagon or fall back to any with weather
            weather = None
            if hexagon:
                weather = hexagon.get_weather()
            if weather is None:
                for h in city.hexagons.values():
                    weather = h.get_weather()
                    if weather and weather.temperature is not None:
                        break

            if weather is None:
                raise HTTPException(status_code=503, detail="No weather data available yet")

            return WeatherResponse(
                temperature=weather.temperature,
                apparent_temperature=weather.apparent_temperature,
                humidity=weather.humidity,
                precip_intensity=weather.precip_intensity,
                wind_speed=weather.wind_speed,
                weather_code=weather.weather_code,
                description=weather.description,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Weather error: {str(e)}")

    @app.get("/routes", response_model=RoutesResponse)
    async def list_routes():
        """List all available routes."""
        topology = observatory.get_topology()
        routes = [{"route_id": rid} for rid in topology.routes.keys()]
        return RoutesResponse(routes=routes)

    @app.get("/routes/{route_id}/directions", response_model=DirectionsResponse)
    async def get_directions(route_id: str):
        """List all directions for a route."""
        topology = observatory.get_topology()
        directions = {}
        for trip_id, trip in list(topology.trips.items()):
            if trip.route.id == route_id:
                dir_id = trip.direction_id
                if dir_id not in directions:
                    directions[dir_id] = trip.direction_name or f"Direction {dir_id}"

        if not directions:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        return DirectionsResponse(
            route_id=route_id,
            directions=[
                DirectionInfo(direction_id=d, trip_headsign=h)
                for d, h in sorted(directions.items())
            ],
        )

    @app.get(
        "/routes/{route_id}/directions/{direction_id}",
        response_model=RouteDirectionResponse,
    )
    async def get_route_info(route_id: str, direction_id: int):
        """Get full info for route+direction: shape, stops, schedule."""
        topology = observatory.get_topology()

        shape_counts = {}
        canonical_trip = None
        all_trips = []

        for trip_id, trip in list(topology.trips.items()):
            if trip.route.id == route_id and trip.direction_id == direction_id:
                all_trips.append(trip)
                if trip.shape:
                    sid = trip.shape.id
                    shape_counts[sid] = shape_counts.get(sid, 0) + 1
                    current_count = shape_counts[sid]
                    best_count = shape_counts.get(
                        canonical_trip.shape.id
                        if canonical_trip and canonical_trip.shape
                        else "",
                        0,
                    )
                    if canonical_trip is None or current_count > best_count:
                        canonical_trip = trip

        if not canonical_trip:
            raise HTTPException(
                status_code=404,
                detail=f"Route {route_id} direction {direction_id} not found",
            )

        shape_points = []
        if canonical_trip.shape:
            for i in range(len(canonical_trip.shape.coords)):
                shape_points.append(
                    ShapePoint(
                        lat=float(canonical_trip.shape.coords[i][0]),
                        lon=float(canonical_trip.shape.coords[i][1]),
                        dist=float(canonical_trip.shape.distances[i]),
                    )
                )

        stops = []
        stop_times = canonical_trip.get_stop_times() or []
        for st in stop_times:
            stop_id = st.get("stop_id")
            stop_info = topology.stops.get(stop_id, {})
            stops.append(
                StopInfo(
                    stop_id=stop_id or "",
                    stop_name=stop_info.get("stop_name", ""),
                    stop_lat=float(stop_info.get("stop_lat", 0) or 0),
                    stop_lon=float(stop_info.get("stop_lon", 0) or 0),
                    stop_sequence=int(st.get("stop_sequence", 0) or 0),
                    shape_dist_traveled=float(st.get("shape_dist_traveled", 0))
                    if st.get("shape_dist_traveled")
                    else None,
                )
            )

        schedule = []
        for trip in all_trips:
            first_stop = (trip.get_stop_times() or [{}])[0]
            schedule.append(
                TripSchedule(
                    trip_id=trip.id,
                    start_time=first_stop.get("arrival_time", ""),
                )
            )

        return RouteDirectionResponse(
            route_id=route_id,
            direction_id=direction_id,
            trip_headsign=canonical_trip.direction_name or "",
            shape=shape_points,
            stops=stops,
            schedule=schedule,
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(request: BatchPredictRequest):
        """Predict bus delays for multiple trips in a single batched inference."""
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        topology = observatory.get_topology()
        valid_trips_data = []
        failed = []

        valid_route_directions = set()
        for trip in list(topology.trips.values()):
            valid_route_directions.add((trip.route.id, trip.direction_id))

        for idx, trip in enumerate(request.trips):
            key = (trip.route_id, trip.direction_id)
            if key in valid_route_directions:
                valid_trips_data.append(
                    {
                        "orig_idx": idx,
                        "route_id": trip.route_id,
                        "direction_id": trip.direction_id,
                        "start_date": trip.start_date,
                        "start_time": trip.start_time,
                        "weather_code": trip.weather_code,
                        "bus_type": trip.bus_type,
                    }
                )
            else:
                failed.append(
                    FailedTrip(
                        index=idx,
                        route_id=trip.route_id,
                        direction_id=trip.direction_id,
                        start_time=trip.start_time,
                        error=f"Route {trip.route_id} direction {trip.direction_id} not found",
                    )
                )

        successful = []

        if valid_trips_data:
            try:
                forecasts = predictor.get_batch_forecast(
                    [
                        {k: v for k, v in t.items() if k != "orig_idx"}
                        for t in valid_trips_data
                    ]
                )

                stops_cache = {}
                trip_data_by_key = {
                    (t["route_id"], t["direction_id"], t["start_time"]): t
                    for t in valid_trips_data
                }

                for forecast in forecasts:
                    start_time = (
                        forecast.scheduled_start[:-3]
                        if forecast.scheduled_start.endswith(":00")
                        else forecast.scheduled_start
                    )
                    key = (
                        forecast.route_id,
                        forecast.direction_id,
                        start_time,
                    )
                    trip_data = trip_data_by_key.get(key)
                    if not trip_data:
                        for k, v in trip_data_by_key.items():
                            if (
                                k[0] == forecast.route_id
                                and k[1] == forecast.direction_id
                            ):
                                trip_data = v
                                break

                    if not trip_data:
                        continue

                    cache_key = (forecast.route_id, forecast.direction_id)
                    if cache_key not in stops_cache:
                        stops_map = {}
                        for ledger_trip in list(topology.trips.values()):
                            if (
                                ledger_trip.route.id == forecast.route_id
                                and ledger_trip.direction_id == forecast.direction_id
                            ):
                                for st in ledger_trip.get_stop_times() or []:
                                    seq = int(st.get("stop_sequence", 0) or 0)
                                    if seq not in stops_map:
                                        stop_id = st.get("stop_id")
                                        stop_info = topology.stops.get(stop_id, {})
                                        stops_map[seq] = {
                                            "stop_id": stop_id or "",
                                            "stop_name": stop_info.get("stop_name", ""),
                                            "stop_lat": float(
                                                stop_info.get("stop_lat", 0) or 0
                                            ),
                                            "stop_lon": float(
                                                stop_info.get("stop_lon", 0) or 0
                                            ),
                                        }
                                break
                        stops_cache[cache_key] = stops_map
                    else:
                        stops_map = stops_cache[cache_key]

                    stop_sequence = {}
                    for pred in forecast.stops:
                        stop_info = stops_map.get(pred.stop_sequence, {})
                        stop_sequence[pred.stop_sequence] = StopPredictionWithInfo(
                            stop_sequence=pred.stop_sequence,
                            stop_id=stop_info.get("stop_id", ""),
                            stop_name=stop_info.get("stop_name", ""),
                            stop_lat=stop_info.get("stop_lat", 0),
                            stop_lon=stop_info.get("stop_lon", 0),
                            predicted_arrival=pred.expected_arrival,
                            delay_seconds=pred.cumulative_delay_sec,
                            confidence_rating=None,
                        )

                    _record_predictions(forecast, stops_map)

                    successful.append(
                        PredictedTrip(
                            trip_id=f"{forecast.route_id}_{forecast.scheduled_start}",
                            route_id=forecast.route_id,
                            direction_id=forecast.direction_id,
                            trip_date=forecast.trip_date,
                            scheduled_start=forecast.scheduled_start,
                            weather_code=trip_data["weather_code"],
                            bus_type=trip_data["bus_type"],
                            stop_sequence=stop_sequence,
                        )
                    )

            except Exception as e:
                for trip_data in valid_trips_data:
                    failed.append(
                        FailedTrip(
                            index=trip_data["orig_idx"],
                            route_id=trip_data["route_id"],
                            direction_id=trip_data["direction_id"],
                            start_time=trip_data["start_time"],
                            error=str(e),
                        )
                    )

        return BatchPredictResponse(
            successful=successful,
            failed=failed,
            total_requested=len(request.trips),
            total_successful=len(successful),
            total_failed=len(failed),
        )

    @app.post("/predict", response_model=PredictedTrip)
    async def predict(request: PredictRequest):
        """Predict bus delays with full stop information.

        Returns a cached prediction from the ledger if one exists for this
        exact trip, otherwise runs the model and caches the result.
        """
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        topology = observatory.get_topology()

        # --- Ledger cache check ---
        try:
            cached = observatory.predicted.query_trip(
                route_id=request.route_id,
                direction_id=request.direction_id,
                trip_date=request.start_date,
                scheduled_start=request.start_time,
            )
            if not cached.empty:
                return _reconstruct_from_ledger(
                    cached, topology,
                    request.route_id, request.direction_id,
                    request.weather_code, request.bus_type,
                )
        except Exception as e:
            logging.debug(f"Ledger cache lookup failed (will predict fresh): {e}")

        # --- Fresh prediction ---
        try:
            forecast: TripForecast = await asyncio.to_thread(
                predictor.get_trip_forecast,
                request.route_id, request.direction_id,
                request.start_date, request.start_time,
                request.weather_code, request.bus_type,
            )

            stops_map = topology.build_stops_map(request.route_id, request.direction_id)

            stop_sequence = {}
            for pred in forecast.stops:
                seq = pred.stop_sequence
                stop_info = stops_map.get(seq, {})
                stop_sequence[seq] = StopPredictionWithInfo(
                    stop_sequence=seq,
                    stop_id=stop_info.get("stop_id", ""),
                    stop_name=stop_info.get("stop_name", ""),
                    stop_lat=stop_info.get("stop_lat", 0),
                    stop_lon=stop_info.get("stop_lon", 0),
                    predicted_arrival=pred.expected_arrival,
                    delay_seconds=pred.cumulative_delay_sec,
                    confidence_rating=None,
                )

            _record_predictions(forecast, stops_map)

            return PredictedTrip(
                route_id=forecast.route_id,
                direction_id=forecast.direction_id,
                trip_date=forecast.trip_date,
                scheduled_start=forecast.scheduled_start,
                weather_code=request.weather_code,
                bus_type=request.bus_type,
                stop_sequence=stop_sequence,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    @app.post("/validate", response_model=ValidateResponse)
    async def validate(request: ValidateRequest):
        """Validate model predictions against ground truth for a specific date."""
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        if observatory is None:
            raise HTTPException(status_code=503, detail="Observatory not loaded")

        from application.services.validator import Validator

        validator = Validator(
            predictor=predictor,
            observatory=observatory,
        )

        try:
            report = validator.validate_date(request.date)

            trip_summaries = [
                TripValidationSummary(
                    trip_id=t.trip_id,
                    route_id=t.route_id,
                    direction_id=t.direction_id,
                    scheduled_start=t.scheduled_start,
                    mse=t.mse,
                    rmse=t.rmse,
                    n_measurements=t.n_measurements,
                    error=t.error,
                )
                for t in report.trips
            ]

            return ValidateResponse(
                date=report.date,
                total_scheduled_trips=report.total_scheduled_trips,
                total_trips_with_ground_truth=report.total_trips_with_ground_truth,
                total_trips_predicted=report.total_trips_predicted,
                total_trips_validated=report.total_trips_validated,
                total_measurements=report.total_measurements,
                median_mse=report.median_mse,
                median_rmse=report.median_rmse,
                min_mse=report.min_mse,
                max_mse=report.max_mse,
                min_rmse=report.min_rmse,
                max_rmse=report.max_rmse,
                occupancy_confusion_matrix=report.occupancy_confusion_matrix,
                trips=trip_summaries,
                log_file=report.log_file,
                report_file=report.report_file,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

    @app.post("/validate/live/schedule", response_model=LiveValidateScheduleResponse)
    async def schedule_live_validation(request: LiveValidateScheduleRequest):
        """Schedule and start a live validation session for a date."""
        nonlocal current_live_session, bus_type_predictor

        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        if observatory is None:
            raise HTTPException(status_code=503, detail="Observatory not loaded")

        if current_live_session is not None and current_live_session.status in [
            "predicting",
            "monitoring",
        ]:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Session already running",
                    "session_id": current_live_session.session_id,
                    "date": current_live_session.target_date,
                    "status": current_live_session.status,
                },
            )

        from application.services.live_validator import LiveValidationSession
        from application.services.bus_type_predictor import BusTypePredictor

        if bus_type_predictor is None:
            try:
                bus_type_predictor = BusTypePredictor()
            except FileNotFoundError:
                pass

        session_id = str(uuid.uuid4())

        session = LiveValidationSession(
            session_id=session_id,
            target_date=request.date,
            predictor=predictor,
            observatory=observatory,
            bus_type_predictor=bus_type_predictor,
        )

        success = await session.start()

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to start live validation session"
            )

        current_live_session = session

        status = session.get_status()

        return LiveValidateScheduleResponse(
            session_id=session_id,
            date=request.date,
            total_scheduled=status.total_scheduled,
            total_predicted=status.total_predicted,
            status=status.status,
            started_at=status.started_at,
            stops_at=status.stops_at,
        )

    @app.post("/validate/live/stop")
    async def stop_live_validation():
        """Stop the current live validation session."""
        nonlocal current_live_session

        if current_live_session is None:
            raise HTTPException(status_code=404, detail="No active session")

        await current_live_session.stop()

        status = current_live_session.get_status()

        return {
            "session_id": current_live_session.session_id,
            "status": status.status,
            "total_validated": status.total_validated,
        }

    @app.get("/validate/live/status", response_model=LiveValidateStatusResponse)
    async def get_live_validation_status():
        """Get the current live validation session status."""
        if current_live_session is None:
            return LiveValidateStatusResponse()

        status = current_live_session.get_status()

        return LiveValidateStatusResponse(
            session_id=status.session_id,
            date=status.date,
            status=status.status,
            total_scheduled=status.total_scheduled,
            total_predicted=status.total_predicted,
            total_validated=status.total_validated,
            total_pending=status.total_pending,
            total_discarded=status.total_discarded,
            median_mse=status.median_mse,
            median_rmse=status.median_rmse,
            min_mse=status.min_mse,
            max_mse=status.max_mse,
            min_rmse=status.min_rmse,
            max_rmse=status.max_rmse,
            started_at=status.started_at,
            stops_at=status.stops_at,
            log_file=status.log_file,
            report_file=status.report_file,
        )

    @app.websocket("/validate/live/ws/{session_id}")
    async def live_validation_websocket(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for real-time validation updates."""
        if (
            current_live_session is None
            or current_live_session.session_id != session_id
        ):
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()

        await current_live_session.add_websocket(websocket)

        try:
            while current_live_session.status in ["predicting", "monitoring"]:
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass
        finally:
            current_live_session.remove_websocket(websocket)

    @app.get("/health")
    async def health():
        """Health."""
        return {
            "status": "healthy",
            "model_loaded": predictor is not None,
            "ledger_loaded": observatory is not None,
            "city_loaded": city is not None,
        }

    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event."""
        print("\nShutting down data collection services...")
        from interaction import services as ingestor_services
        from interaction.main import last_save
        from application.live import data as live_data
        from persistence.database import shutdown_database

        ingestor_services.stop_services()
        if live_data.OBSERVATORY:
            last_save(live_data.OBSERVATORY)

        # Close database pools and shared DB event loop
        shutdown_database()

    import uvicorn

    final_host = host or api_config["host"]
    final_port = port or api_config["port"]
    uvicorn.run(app, host=final_host, port=final_port)


def main():
    """Main."""
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

    serve_parser = subparsers.add_parser("serve", help="Start prediction API server with integrated data collection")
    serve_parser.add_argument("--time-model",  type=str, help="TIME model filename (e.g. bus_model_TIME_mse_0.pth)")
    serve_parser.add_argument("--crowd-model", type=str, help="CROWD model filename (e.g. bus_model_CROWD_mse_0.pth)")
    serve_parser.add_argument("--host", type=str, help="Host to bind")
    serve_parser.add_argument("--port", type=int, help="Port to bind")
    serve_parser.add_argument("--lenient-pipeline", action="store_true", help="Use lenient data cleaning")

    subparsers.add_parser("test-db", help="Test database connections")

    args = parser.parse_args()

    if args.command == "collect":
        run_collect(debug_mode=args.debug, lenient_pipeline=args.lenient_pipeline)
    elif args.command == "serve":
        run_serve(time_model_name=args.time_model, crowd_model_name=args.crowd_model, host=args.host, port=args.port, lenient_pipeline=args.lenient_pipeline)
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
