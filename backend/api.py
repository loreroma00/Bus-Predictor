"""FastAPI boundary for the backend process."""

from __future__ import annotations

import asyncio
import configparser
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from application.model.model_loader import ModelCandidate, ModelLoader
from application.model.predictor import Predictor, TripForecast


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"
PARQUET_DIR = PROJECT_ROOT / "parquets"


class PredictRequest(BaseModel):
    """Prediction request at the HTTP boundary."""

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


class StopPredictionWithInfo(BaseModel):
    """Prediction enriched with stop metadata."""

    stop_sequence: int
    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    predicted_arrival: str
    delay_seconds: float
    confidence_rating: Optional[float] = None


class PredictedTrip(BaseModel):
    """HTTP response for one predicted trip."""

    trip_id: str
    route_id: str
    direction_id: int
    trip_date: str
    scheduled_start: str
    weather_code: int
    bus_type: int
    stop_sequence: dict[int, StopPredictionWithInfo]


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    trips: list[PredictRequest]


class FailedTrip(BaseModel):
    """A trip rejected or failed during batch prediction."""

    index: int
    route_id: str
    direction_id: int
    start_time: str
    error: str


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    successful: list[PredictedTrip]
    failed: list[FailedTrip]
    total_requested: int
    total_successful: int
    total_failed: int


class ModelInfo(BaseModel):
    """Public model card for `/models`."""

    filename: str
    path: str


class WeatherResponse(BaseModel):
    """Current weather snapshot."""

    temperature: float
    apparent_temperature: float
    humidity: float
    precip_intensity: float
    wind_speed: float
    weather_code: int
    description: str


class RouteSummary(BaseModel):
    """Route list item."""

    route_id: str


class RoutesResponse(BaseModel):
    """Route list response."""

    routes: list[RouteSummary]


class DirectionInfo(BaseModel):
    """Direction list item."""

    direction_id: int
    trip_headsign: str


class DirectionsResponse(BaseModel):
    """Direction list response."""

    route_id: str
    directions: list[DirectionInfo]


class ShapePoint(BaseModel):
    """A point on a route shape."""

    lat: float
    lon: float
    dist: float


class StopInfo(BaseModel):
    """Static stop metadata for a route direction."""

    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    stop_sequence: int
    shape_dist_traveled: Optional[float] = None


class TripSchedule(BaseModel):
    """One scheduled trip start."""

    trip_id: str
    start_time: str


class RouteDirectionResponse(BaseModel):
    """Full static route/direction details."""

    route_id: str
    direction_id: int
    trip_headsign: str
    shape: list[ShapePoint]
    stops: list[StopInfo]
    schedule: list[TripSchedule]


class ValidateRequest(BaseModel):
    """Retrospective validation request."""

    date: str

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%d-%m-%Y")
            return v
        except ValueError:
            raise ValueError("date must be in DD-MM-YYYY format")


class TripValidationSummary(BaseModel):
    """Per-trip validation summary."""

    trip_id: str
    route_id: str
    direction_id: int
    scheduled_start: str
    mse: float
    rmse: float
    n_measurements: int
    error: Optional[str] = None


class ValidateResponse(BaseModel):
    """Retrospective validation response."""

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
    """Live validation schedule request."""

    date: str

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%d-%m-%Y")
            return v
        except ValueError:
            raise ValueError("date must be in DD-MM-YYYY format")


class LiveValidateScheduleResponse(BaseModel):
    """Live validation schedule response."""

    session_id: str
    date: str
    total_scheduled: int
    total_predicted: int
    status: str
    started_at: str
    stops_at: Optional[str] = None


class LiveValidateStatusResponse(BaseModel):
    """Live validation status response."""

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


@dataclass
class APIState:
    """Runtime state owned by the API process."""

    model_loader: ModelLoader = field(default_factory=ModelLoader)
    available_models: list[ModelCandidate] = field(default_factory=list)
    predictor: Predictor | None = None
    observatory: object | None = None
    city: object | None = None
    runtime_context: object | None = None
    validation_controller: object | None = None
    bus_type_predictor: object | None = None


def load_api_config() -> dict:
    """Load only the API host/CORS config needed by the FastAPI boundary."""
    config = configparser.ConfigParser()
    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)
    return {
        "frontend_url": config.get(
            "api",
            "frontend_url",
            fallback="http://localhost:3000",
        ),
        "host": config.get("api", "host", fallback="0.0.0.0"),
        "port": config.getint("api", "port", fallback=8000),
    }


def create_app(
    time_model_name: str | None = None,
    crowd_model_name: str | None = None,
    lenient_pipeline: bool = False,
) -> FastAPI:
    """Create the FastAPI app and wire startup/shutdown lifecycle hooks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    api_config = load_api_config()
    state = APIState()

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

    async def generate_daily_predictions(date_yyyymmdd: str):
        """Pre-generate and cache predictions for every scheduled trip."""
        if state.predictor is None or state.observatory is None:
            return

        date_obj = datetime.strptime(date_yyyymmdd, "%Y%m%d").date()
        start_date_fmt = date_obj.strftime("%d-%m-%Y")
        topology = state.observatory.get_topology()
        weather_code = _get_current_weather_code(state.city)

        if state.bus_type_predictor is None:
            try:
                from application.services.bus_type_predictor import BusTypePredictor

                state.bus_type_predictor = BusTypePredictor()
            except Exception:
                pass

        already_cached = state.predictor.get_cached_trip_keys(start_date_fmt)
        seen: set[tuple[str, int, str]] = set()
        trips_to_run: list[tuple[str, int, str]] = []
        for trip in list(topology.trips.values()):
            if date_yyyymmdd not in (trip.dates or []):
                continue
            if not trip.stop_times:
                continue
            arrival = trip.stop_times[0].get("arrival_time", "")
            if not arrival:
                continue
            start_time = arrival[:5]
            key = (trip.route.id, int(trip.direction_id or 0), start_time)
            if key in seen or key in already_cached:
                continue
            seen.add(key)
            trips_to_run.append(key)

        logging.info(
            "[pre-gen] %s: %d trips to predict (%d already cached)",
            date_yyyymmdd,
            len(trips_to_run),
            len(already_cached),
        )

        generated = errors = 0
        for route_id, direction_id, start_time in trips_to_run:
            try:
                bus_type = 0
                if state.bus_type_predictor is not None:
                    try:
                        bus_type = int(
                            state.bus_type_predictor.predict(
                                route_id=route_id,
                                start_time=start_time,
                                trip_date=date_obj,
                            )
                            or 0
                        )
                    except Exception:
                        pass

                await asyncio.to_thread(
                    state.predictor.get_or_create_trip_forecast,
                    route_id,
                    direction_id,
                    start_date_fmt,
                    start_time,
                    weather_code,
                    bus_type,
                )
                generated += 1
            except Exception as e:
                errors += 1
                logging.debug(
                    "[pre-gen] skip %s/%s/%s: %s",
                    route_id,
                    direction_id,
                    start_time,
                    e,
                )

            if (generated + errors) % 20 == 0:
                await asyncio.sleep(0)

        logging.info("[pre-gen] done: %d generated, %d skipped", generated, errors)

    async def periodic_ledger_check():
        """Refresh static ledgers periodically and rewarm predictions on change."""
        while True:
            await asyncio.sleep(3600)
            if state.observatory and state.observatory.check_and_reload_ledger():
                print("Ledger updated with new GTFS data")
                today = datetime.now().strftime("%Y%m%d")
                asyncio.create_task(generate_daily_predictions(today))

    @app.on_event("startup")
    async def startup_event():
        print("\n" + "=" * 50)
        print("Initializing ATAC Backend...")
        print("=" * 50)

        print("\n[1/7] Initializing collection pipeline (Observatory, City, services)...")
        from bootstrapper import build_runtime_context

        state.runtime_context = build_runtime_context(
            lenient_pipeline=lenient_pipeline,
        )
        state.observatory = state.runtime_context.observatory
        state.city = state.runtime_context.city

        print("\n[2/7] Generating canonical route map (if needed)...")
        _generate_canonical_map()

        print("\n[3/7] Loading ML model...")
        loaded_model = _select_loaded_model(
            state.model_loader,
            time_model_name,
            crowd_model_name,
        )
        state.available_models = state.model_loader.discover_models()
        state.predictor = Predictor(
            loaded_model=loaded_model,
            observatory=state.observatory,
        )

        print("\n[4/7] Starting data collection services...")
        from bootstrapper import start_collection_services, wire_state_interface

        wire_state_interface(
            state.runtime_context,
            predictor=state.predictor,
            bus_type_predictor=state.bus_type_predictor,
            loaded_model_name=loaded_model.name,
        )
        start_collection_services(state.runtime_context)

        print("\n[6/7] Starting background tasks...")
        asyncio.create_task(periodic_ledger_check())
        today = datetime.now().strftime("%Y%m%d")
        asyncio.create_task(generate_daily_predictions(today))

        print("\n[7/7] Starting interactive console...")
        from interaction import console

        threading.Thread(
            target=console.run_console_loop,
            daemon=True,
        ).start()

        print(f"\nCORS enabled for: {api_config['frontend_url']}")
        print("Server ready with integrated data collection.")
        print("API documentation at: /docs")
        print("=" * 50 + "\n")

    @app.get("/models", response_model=list[ModelInfo])
    async def list_models():
        return [_model_info(candidate) for candidate in state.available_models]

    @app.get("/weather", response_model=WeatherResponse)
    async def get_weather(lat: float = None, lon: float = None, hex_id: str = None):
        try:
            if state.city is None:
                raise HTTPException(status_code=503, detail="City not initialized")

            hexagon = None
            if hex_id:
                hexagon = state.city.get_hexagon(hex_id)
            elif lat is not None and lon is not None:
                target_hex = state.city.get_hex_id(lat, lon)
                hexagon = state.city.get_hexagon(target_hex)

            weather = None
            if hexagon:
                weather = hexagon.get_weather()
            if weather is None:
                for h in state.city.hexagons.values():
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
        topology = _topology_or_503(state)
        routes = [{"route_id": rid} for rid in topology.routes.keys()]
        return RoutesResponse(routes=routes)

    @app.get("/routes/{route_id}/directions", response_model=DirectionsResponse)
    async def get_directions(route_id: str):
        topology = _topology_or_503(state)
        directions = {}
        for trip in list(topology.trips.values()):
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
        topology = _topology_or_503(state)
        canonical_trip, all_trips = _find_canonical_trip(
            topology,
            route_id,
            direction_id,
        )

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
        for st in canonical_trip.get_stop_times() or []:
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
        predictor = _predictor_or_503(state)
        topology = _topology_or_503(state)
        valid_route_directions = {
            (trip.route.id, trip.direction_id)
            for trip in list(topology.trips.values())
        }

        valid_trips_data = []
        failed = []
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
                    ],
                    record=True,
                )
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
                    trip_data = trip_data_by_key.get(
                        (forecast.route_id, forecast.direction_id, start_time)
                    )
                    if trip_data is None:
                        trip_data = next(
                            (
                                t
                                for t in valid_trips_data
                                if t["route_id"] == forecast.route_id
                                and t["direction_id"] == forecast.direction_id
                            ),
                            None,
                        )
                    if trip_data is None:
                        continue
                    successful.append(
                        _to_predicted_trip(
                            forecast,
                            topology,
                            weather_code=trip_data["weather_code"],
                            bus_type=trip_data["bus_type"],
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
        predictor = _predictor_or_503(state)
        topology = _topology_or_503(state)

        try:
            forecast = await asyncio.to_thread(
                predictor.get_or_create_trip_forecast,
                request.route_id,
                request.direction_id,
                request.start_date,
                request.start_time,
                request.weather_code,
                request.bus_type,
            )
            return _to_predicted_trip(
                forecast,
                topology,
                weather_code=request.weather_code,
                bus_type=request.bus_type,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    @app.post("/validate", response_model=ValidateResponse)
    async def validate(request: ValidateRequest):
        predictor = _predictor_or_503(state)
        if state.observatory is None:
            raise HTTPException(status_code=503, detail="Observatory not loaded")

        try:
            controller = _validation_controller(state)
            report = controller.run_historical(
                request.date,
                predictor=predictor,
                observatory=state.observatory,
            )
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
        predictor = _predictor_or_503(state)
        if state.observatory is None:
            raise HTTPException(status_code=503, detail="Observatory not loaded")

        controller = _validation_controller(state)
        current_session = controller.get_live_session()
        if current_session is not None and current_session.status in [
            "predicting",
            "monitoring",
        ]:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Session already running",
                    "session_id": current_session.session_id,
                    "date": current_session.target_date,
                    "status": current_session.status,
                },
            )

        from application.services.bus_type_predictor import BusTypePredictor

        if state.bus_type_predictor is None:
            try:
                state.bus_type_predictor = BusTypePredictor()
            except FileNotFoundError:
                pass

        session = await controller.start_live_session(
            date_str=request.date,
            predictor=predictor,
            observatory=state.observatory,
            bus_type_predictor=state.bus_type_predictor,
        )
        if session is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to start live validation session",
            )

        status = session.get_status()
        return LiveValidateScheduleResponse(
            session_id=session.session_id,
            date=request.date,
            total_scheduled=status.total_scheduled,
            total_predicted=status.total_predicted,
            status=status.status,
            started_at=status.started_at,
            stops_at=status.stops_at,
        )

    @app.post("/validate/live/stop")
    async def stop_live_validation():
        controller = _validation_controller(state)
        session = controller.get_live_session()
        if session is None:
            raise HTTPException(status_code=404, detail="No active session")

        await controller.stop_live_session()
        status = session.get_status()
        return {
            "session_id": session.session_id,
            "status": status.status,
            "total_validated": status.total_validated,
        }

    @app.get("/validate/live/status", response_model=LiveValidateStatusResponse)
    async def get_live_validation_status():
        controller = _validation_controller(state)
        session = controller.get_live_session()
        if session is None:
            return LiveValidateStatusResponse()

        status = session.get_status()
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
        controller = _validation_controller(state)
        session = controller.get_live_session(session_id)
        if session is None:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()
        await session.add_websocket(websocket)
        try:
            while session.status in ["predicting", "monitoring"]:
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass
        finally:
            session.remove_websocket(websocket)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_loaded": state.predictor is not None,
            "ledger_loaded": state.observatory is not None,
            "city_loaded": state.city is not None,
        }

    @app.on_event("shutdown")
    async def shutdown_event():
        print("\nShutting down data collection services...")
        from bootstrapper import shutdown_runtime

        shutdown_runtime(state.runtime_context, save_diaries=True)

    return app


def _model_info(candidate: ModelCandidate) -> ModelInfo:
    return ModelInfo(filename=candidate.name, path=str(candidate.time_path))


def _select_loaded_model(
    loader: ModelLoader,
    time_model_name: str | None,
    crowd_model_name: str | None,
):
    available = loader.discover_models()
    for time_filename, missing in loader.find_incomplete_models():
        print(f"[WARN] Skipping {time_filename}: missing {', '.join(missing)}")

    if not available:
        raise RuntimeError(
            "No trained model pairs found in application/model/. Expected "
            "bus_model_TIME_*.pth + bus_model_CROWD_*.pth + hyperparameters_DUAL_*.json"
        )

    cli_time = time_model_name or os.environ.get("TIME_MODEL_NAME")
    cli_crowd = crowd_model_name or os.environ.get("CROWD_MODEL_NAME")
    if cli_time and cli_crowd:
        return loader.load_pair(cli_time, cli_crowd)
    if cli_time:
        exp_id = cli_time.replace("bus_model_TIME_", "").replace(".pth", "")
        return loader.load_by_exp_id(exp_id)

    return _interactive_model_selection(loader, available)


def _interactive_model_selection(
    loader: ModelLoader,
    available_models: list[ModelCandidate],
):
    print("\n" + "=" * 50)
    print("ATAC Bus Delay Prediction - Model Selection")
    print("=" * 50)
    print("\nAvailable model pairs:")
    for i, model in enumerate(available_models):
        print(f"  [{i}] {model.name}")
        print(f"       TIME:  {model.time_filename}")
        print(f"       CROWD: {model.crowd_filename}")

    while True:
        try:
            choice_idx = int(input("\nSelect model number: ").strip())
            if 0 <= choice_idx < len(available_models):
                break
            print(f"Please enter a number between 0 and {len(available_models) - 1}")
        except (ValueError, EOFError):
            print("Invalid input. Please enter a number.")

    return loader.load_by_exp_id(available_models[choice_idx].name)


def _generate_canonical_map():
    if not (PARQUET_DIR / "stop_route_map.parquet").exists():
        print("Generating stop_route_map.parquet...")
        from prepare_dataset import build_canonical_shape_map

        build_canonical_shape_map()


def _get_current_weather_code(city) -> int:
    try:
        if city is None:
            return 0
        for hexagon in city.hexagons.values():
            weather = getattr(hexagon, "weather", None)
            if weather is not None:
                return int(weather.weather_code)
    except Exception:
        pass
    return 0


def _topology_or_503(state: APIState):
    if state.observatory is None:
        raise HTTPException(status_code=503, detail="Observatory not loaded")
    return state.observatory.get_topology()


def _predictor_or_503(state: APIState) -> Predictor:
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state.predictor


def _validation_controller(state: APIState):
    """Return the API/runtime validation controller."""
    runtime_context = state.runtime_context
    if runtime_context is not None:
        controller = getattr(runtime_context, "validation_controller", None)
        if controller is not None:
            state.validation_controller = controller
            return controller

    if state.validation_controller is None:
        from application.services.validator import ValidationController

        state.validation_controller = ValidationController()
        if runtime_context is not None:
            runtime_context.validation_controller = state.validation_controller

    return state.validation_controller


def _find_canonical_trip(topology, route_id: str, direction_id: int):
    shape_counts = {}
    canonical_trip = None
    all_trips = []

    for trip in list(topology.trips.values()):
        if trip.route.id == route_id and trip.direction_id == direction_id:
            all_trips.append(trip)
            if trip.shape:
                shape_id = trip.shape.id
                shape_counts[shape_id] = shape_counts.get(shape_id, 0) + 1
                current_count = shape_counts[shape_id]
                best_count = shape_counts.get(
                    canonical_trip.shape.id
                    if canonical_trip and canonical_trip.shape
                    else "",
                    0,
                )
                if canonical_trip is None or current_count > best_count:
                    canonical_trip = trip

    return canonical_trip, all_trips


def _to_predicted_trip(
    forecast: TripForecast,
    topology,
    weather_code: int,
    bus_type: int,
) -> PredictedTrip:
    stops_map = topology.build_stops_map(forecast.route_id, forecast.direction_id)
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

    return PredictedTrip(
        trip_id=f"{forecast.route_id}_{forecast.scheduled_start}",
        route_id=forecast.route_id,
        direction_id=forecast.direction_id,
        trip_date=forecast.trip_date,
        scheduled_start=forecast.scheduled_start,
        weather_code=weather_code,
        bus_type=bus_type,
        stop_sequence=stop_sequence,
    )
