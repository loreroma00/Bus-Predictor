"""FastAPI boundary for the backend process."""

from __future__ import annotations

import configparser
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"


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
    runtime = None

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

    def runtime_or_503():
        """Return the initialized API runtime."""
        if runtime is None:
            raise HTTPException(status_code=503, detail="API runtime not initialized")
        return runtime

    @app.on_event("startup")
    async def startup_event():
        nonlocal runtime
        print("\n" + "=" * 50)
        print("Initializing ATAC Backend...")
        print("=" * 50)

        from bootstrapper import build_serving_runtime

        runtime = build_serving_runtime(
            time_model_name=time_model_name,
            crowd_model_name=crowd_model_name,
            lenient_pipeline=lenient_pipeline,
        )

        print("\n[5/5] Starting background tasks and interactive console...")
        await runtime.start_background_tasks()

        print(f"\nCORS enabled for: {api_config['frontend_url']}")
        print("Server ready with integrated data collection.")
        print("API documentation at: /docs")
        print("=" * 50 + "\n")

    @app.get("/models", response_model=list[ModelInfo])
    async def list_models():
        return runtime_or_503().list_models()

    @app.get("/weather", response_model=WeatherResponse)
    async def get_weather(lat: float = None, lon: float = None, hex_id: str = None):
        return runtime_or_503().get_weather(lat=lat, lon=lon, hex_id=hex_id)

    @app.get("/routes", response_model=RoutesResponse)
    async def list_routes():
        return runtime_or_503().list_routes()

    @app.get("/routes/{route_id}/directions", response_model=DirectionsResponse)
    async def get_directions(route_id: str):
        return runtime_or_503().get_directions(route_id)

    @app.get(
        "/routes/{route_id}/directions/{direction_id}",
        response_model=RouteDirectionResponse,
    )
    async def get_route_info(route_id: str, direction_id: int):
        return runtime_or_503().get_route_info(route_id, direction_id)

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(request: BatchPredictRequest):
        return await runtime_or_503().predict_batch(
            [trip.model_dump() for trip in request.trips]
        )

    @app.post("/predict", response_model=PredictedTrip)
    async def predict(request: PredictRequest):
        return await runtime_or_503().predict(request.model_dump())

    @app.post("/validate", response_model=ValidateResponse)
    async def validate(request: ValidateRequest):
        return runtime_or_503().validate(request.date)

    @app.post("/validate/live/schedule", response_model=LiveValidateScheduleResponse)
    async def schedule_live_validation(request: LiveValidateScheduleRequest):
        return await runtime_or_503().schedule_live_validation(request.date)

    @app.post("/validate/live/stop")
    async def stop_live_validation():
        return await runtime_or_503().stop_live_validation()

    @app.get("/validate/live/status", response_model=LiveValidateStatusResponse)
    async def get_live_validation_status():
        return runtime_or_503().live_validation_status()

    @app.websocket("/validate/live/ws/{session_id}")
    async def live_validation_websocket(websocket: WebSocket, session_id: str):
        await runtime_or_503().live_validation_websocket(websocket, session_id)

    @app.get("/health")
    async def health():
        if runtime is None:
            return {
                "status": "starting",
                "model_loaded": False,
                "ledger_loaded": False,
                "city_loaded": False,
            }
        return runtime.health()

    @app.on_event("shutdown")
    async def shutdown_event():
        print("\nShutting down data collection services...")
        if runtime is not None:
            runtime.shutdown()

    return app
