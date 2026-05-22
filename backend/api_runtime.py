"""Runtime orchestration used by the FastAPI boundary."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastapi import HTTPException, WebSocket, WebSocketDisconnect


@dataclass
class APIRuntime:
    """Application services exposed through the HTTP boundary."""

    context: Any
    predictor: Any
    available_models: list[dict[str, str]] = field(default_factory=list)
    loaded_model_name: str | None = None
    bus_type_predictor: Any = None
    validation_controller: Any = None
    _tasks_started: bool = False

    @property
    def observatory(self):
        """Return the runtime Observatory."""
        return self.context.observatory

    @property
    def city(self):
        """Return the runtime City."""
        return self.context.city

    async def start_background_tasks(self):
        """Start API-owned warmup tasks and the interactive console thread."""
        if self._tasks_started:
            return
        self._tasks_started = True

        asyncio.create_task(self._periodic_ledger_check())
        today = datetime.now().strftime("%Y%m%d")
        asyncio.create_task(self.generate_daily_predictions(today))

        from interaction import console

        threading.Thread(
            target=console.run_console_loop,
            args=(self.context.shutdown_event,),
            daemon=True,
        ).start()

    def shutdown(self):
        """Stop collection services and persist completed measurements."""
        from bootstrapper import shutdown_runtime

        shutdown_runtime(self.context, save_measurements=True)

    def health(self) -> dict:
        """Return a compact runtime health snapshot."""
        return {
            "status": "healthy",
            "model_loaded": self.predictor is not None,
            "ledger_loaded": self.observatory is not None,
            "city_loaded": self.city is not None,
        }

    def list_models(self) -> list[dict[str, str]]:
        """Return discovered model cards."""
        return self.available_models

    def get_weather(
        self,
        lat: float | None = None,
        lon: float | None = None,
        hex_id: str | None = None,
    ) -> dict:
        """Return the current weather around a point/hex or the first known city weather."""
        if self.city is None:
            raise HTTPException(status_code=503, detail="City not initialized")

        try:
            hexagon = None
            if hex_id:
                hexagon = self.city.get_hexagon(hex_id)
            elif lat is not None and lon is not None:
                target_hex = self.city.get_hex_id(lat, lon)
                hexagon = self.city.get_hexagon(target_hex)

            weather = hexagon.get_weather() if hexagon else None
            if weather is None:
                for city_hexagon in self.city.hexagons.values():
                    weather = city_hexagon.get_weather()
                    if weather and weather.temperature is not None:
                        break

            if weather is None:
                raise HTTPException(
                    status_code=503,
                    detail="No weather data available yet",
                )

            return {
                "temperature": weather.temperature,
                "apparent_temperature": weather.apparent_temperature,
                "humidity": weather.humidity,
                "precip_intensity": weather.precip_intensity,
                "wind_speed": weather.wind_speed,
                "weather_code": weather.weather_code,
                "description": weather.description,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Weather error: {exc}")

    def list_routes(self) -> dict:
        """Return all static GTFS routes."""
        topology = self._topology_or_503()
        return {"routes": [{"route_id": route_id} for route_id in topology.routes.keys()]}

    def get_directions(self, route_id: str) -> dict:
        """Return known directions for one route."""
        topology = self._topology_or_503()
        directions = {}
        for trip in list(topology.trips.values()):
            if trip.route.id == route_id:
                direction_id = trip.direction_id
                if direction_id not in directions:
                    directions[direction_id] = (
                        trip.direction_name or f"Direction {direction_id}"
                    )

        if not directions:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        return {
            "route_id": route_id,
            "directions": [
                {"direction_id": direction_id, "trip_headsign": headsign}
                for direction_id, headsign in sorted(directions.items())
            ],
        }

    def get_route_info(self, route_id: str, direction_id: int) -> dict:
        """Return shape, stops and scheduled starts for one route direction."""
        topology = self._topology_or_503()
        canonical_trip, all_trips = self._find_canonical_trip(
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
                    {
                        "lat": float(canonical_trip.shape.coords[i][0]),
                        "lon": float(canonical_trip.shape.coords[i][1]),
                        "dist": float(canonical_trip.shape.distances[i]),
                    }
                )

        stops = []
        for stop_time in canonical_trip.get_stop_times() or []:
            stop_id = stop_time.get("stop_id")
            stop_info = topology.stops.get(stop_id, {})
            shape_dist = stop_time.get("shape_dist_traveled")
            stops.append(
                {
                    "stop_id": stop_id or "",
                    "stop_name": stop_info.get("stop_name", ""),
                    "stop_lat": float(stop_info.get("stop_lat", 0) or 0),
                    "stop_lon": float(stop_info.get("stop_lon", 0) or 0),
                    "stop_sequence": int(stop_time.get("stop_sequence", 0) or 0),
                    "shape_dist_traveled": float(shape_dist) if shape_dist else None,
                }
            )

        schedule = []
        for trip in all_trips:
            first_stop = (trip.get_stop_times() or [{}])[0]
            schedule.append(
                {
                    "trip_id": trip.id,
                    "start_time": first_stop.get("arrival_time", ""),
                }
            )

        return {
            "route_id": route_id,
            "direction_id": direction_id,
            "trip_headsign": canonical_trip.direction_name or "",
            "shape": shape_points,
            "stops": stops,
            "schedule": schedule,
        }

    async def predict(self, request: dict) -> dict:
        """Return one prediction, using the predictor cache/ledger when possible."""
        predictor = self._predictor_or_503()
        topology = self._topology_or_503()

        try:
            forecast = await asyncio.to_thread(
                predictor.get_or_create_trip_forecast,
                request["route_id"],
                request["direction_id"],
                request["start_date"],
                request["start_time"],
                request["weather_code"],
                request["bus_type"],
            )
            return self._to_predicted_trip(
                forecast,
                topology,
                weather_code=request["weather_code"],
                bus_type=request["bus_type"],
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    async def predict_batch(self, trips: list[dict]) -> dict:
        """Return predictions for a batch of trip requests."""
        predictor = self._predictor_or_503()
        topology = self._topology_or_503()
        valid_route_directions = {
            (trip.route.id, trip.direction_id)
            for trip in list(topology.trips.values())
        }

        valid_trips = []
        failed = []
        for idx, trip in enumerate(trips):
            key = (trip["route_id"], trip["direction_id"])
            if key in valid_route_directions:
                valid_trips.append({"orig_idx": idx, **trip})
            else:
                failed.append(
                    {
                        "index": idx,
                        "route_id": trip["route_id"],
                        "direction_id": trip["direction_id"],
                        "start_time": trip["start_time"],
                        "error": (
                            f"Route {trip['route_id']} direction "
                            f"{trip['direction_id']} not found"
                        ),
                    }
                )

        successful = []
        if valid_trips:
            try:
                forecasts = predictor.get_batch_forecast(
                    [
                        {key: value for key, value in trip.items() if key != "orig_idx"}
                        for trip in valid_trips
                    ],
                    record=True,
                )
                trip_data_by_key = {
                    (trip["route_id"], trip["direction_id"], trip["start_time"]): trip
                    for trip in valid_trips
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
                                trip
                                for trip in valid_trips
                                if trip["route_id"] == forecast.route_id
                                and trip["direction_id"] == forecast.direction_id
                            ),
                            None,
                        )
                    if trip_data is None:
                        continue
                    successful.append(
                        self._to_predicted_trip(
                            forecast,
                            topology,
                            weather_code=trip_data["weather_code"],
                            bus_type=trip_data["bus_type"],
                        )
                    )
            except Exception as exc:
                for trip in valid_trips:
                    failed.append(
                        {
                            "index": trip["orig_idx"],
                            "route_id": trip["route_id"],
                            "direction_id": trip["direction_id"],
                            "start_time": trip["start_time"],
                            "error": str(exc),
                        }
                    )

        return {
            "successful": successful,
            "failed": failed,
            "total_requested": len(trips),
            "total_successful": len(successful),
            "total_failed": len(failed),
        }

    def validate(self, date: str) -> dict:
        """Run retrospective validation."""
        predictor = self._predictor_or_503()
        if self.observatory is None:
            raise HTTPException(status_code=503, detail="Observatory not loaded")

        try:
            report = self._validation_controller().run_historical(
                date,
                predictor=predictor,
                observatory=self.observatory,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Validation error: {exc}")

        return {
            "date": report.date,
            "total_scheduled_trips": report.total_scheduled_trips,
            "total_trips_with_ground_truth": report.total_trips_with_ground_truth,
            "total_trips_predicted": report.total_trips_predicted,
            "total_trips_validated": report.total_trips_validated,
            "total_measurements": report.total_measurements,
            "median_mse": report.median_mse,
            "median_rmse": report.median_rmse,
            "min_mse": report.min_mse,
            "max_mse": report.max_mse,
            "min_rmse": report.min_rmse,
            "max_rmse": report.max_rmse,
            "occupancy_confusion_matrix": report.occupancy_confusion_matrix,
            "trips": [
                {
                    "trip_id": trip.trip_id,
                    "route_id": trip.route_id,
                    "direction_id": trip.direction_id,
                    "scheduled_start": trip.scheduled_start,
                    "mse": trip.mse,
                    "rmse": trip.rmse,
                    "n_measurements": trip.n_measurements,
                    "error": trip.error,
                }
                for trip in report.trips
            ],
            "log_file": report.log_file,
            "report_file": report.report_file,
        }

    async def schedule_live_validation(self, date: str) -> dict:
        """Start one live validation session."""
        predictor = self._predictor_or_503()
        if self.observatory is None:
            raise HTTPException(status_code=503, detail="Observatory not loaded")

        controller = self._validation_controller()
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

        self._ensure_bus_type_predictor(ignore_errors=True)
        session = await controller.start_live_session(
            date_str=date,
            predictor=predictor,
            observatory=self.observatory,
            bus_type_predictor=self.bus_type_predictor,
        )
        if session is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to start live validation session",
            )

        status = session.get_status()
        return {
            "session_id": session.session_id,
            "date": date,
            "total_scheduled": status.total_scheduled,
            "total_predicted": status.total_predicted,
            "status": status.status,
            "started_at": status.started_at,
            "stops_at": status.stops_at,
        }

    async def stop_live_validation(self) -> dict:
        """Stop the active live validation session."""
        controller = self._validation_controller()
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

    def live_validation_status(self) -> dict:
        """Return current live validation status."""
        session = self._validation_controller().get_live_session()
        if session is None:
            return {}

        status = session.get_status()
        return {
            "session_id": status.session_id,
            "date": status.date,
            "status": status.status,
            "total_scheduled": status.total_scheduled,
            "total_predicted": status.total_predicted,
            "total_validated": status.total_validated,
            "total_pending": status.total_pending,
            "total_discarded": status.total_discarded,
            "median_mse": status.median_mse,
            "median_rmse": status.median_rmse,
            "min_mse": status.min_mse,
            "max_mse": status.max_mse,
            "min_rmse": status.min_rmse,
            "max_rmse": status.max_rmse,
            "started_at": status.started_at,
            "stops_at": status.stops_at,
            "log_file": status.log_file,
            "report_file": status.report_file,
        }

    async def live_validation_websocket(self, websocket: WebSocket, session_id: str):
        """Attach a websocket to a live validation session."""
        session = self._validation_controller().get_live_session(session_id)
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

    async def generate_daily_predictions(self, date_yyyymmdd: str):
        """Pre-generate and cache predictions for every scheduled trip."""
        if self.predictor is None or self.observatory is None:
            return

        date_obj = datetime.strptime(date_yyyymmdd, "%Y%m%d").date()
        start_date_fmt = date_obj.strftime("%d-%m-%Y")
        topology = self.observatory.get_topology()
        weather_code = self._get_current_weather_code()
        self._ensure_bus_type_predictor(ignore_errors=True)

        already_cached = self.predictor.get_cached_trip_keys(start_date_fmt)
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
                if self.bus_type_predictor is not None:
                    try:
                        bus_type = int(
                            self.bus_type_predictor.predict(
                                route_id=route_id,
                                start_time=start_time,
                                trip_date=date_obj,
                            )
                            or 0
                        )
                    except Exception:
                        pass

                await asyncio.to_thread(
                    self.predictor.get_or_create_trip_forecast,
                    route_id,
                    direction_id,
                    start_date_fmt,
                    start_time,
                    weather_code,
                    bus_type,
                )
                generated += 1
            except Exception as exc:
                errors += 1
                logging.debug(
                    "[pre-gen] skip %s/%s/%s: %s",
                    route_id,
                    direction_id,
                    start_time,
                    exc,
                )

            if (generated + errors) % 20 == 0:
                await asyncio.sleep(0)

        logging.info("[pre-gen] done: %d generated, %d skipped", generated, errors)

    async def _periodic_ledger_check(self):
        """Refresh static ledgers periodically and rewarm predictions on change."""
        while True:
            await asyncio.sleep(3600)
            if self.observatory and self.observatory.check_and_reload_ledger():
                print("Ledger updated with new GTFS data")
                today = datetime.now().strftime("%Y%m%d")
                asyncio.create_task(self.generate_daily_predictions(today))

    def _topology_or_503(self):
        if self.observatory is None:
            raise HTTPException(status_code=503, detail="Observatory not loaded")
        return self.observatory.get_topology()

    def _predictor_or_503(self):
        if self.predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return self.predictor

    def _validation_controller(self):
        """Return the runtime validation controller."""
        controller = getattr(self.context, "validation_controller", None)
        if controller is not None:
            self.validation_controller = controller
            return controller

        if self.validation_controller is None:
            from application.services.validator import ValidationController

            self.validation_controller = ValidationController(
                persistence_gateway=getattr(self.context, "persistence_gateway", None),
            )

        self.context.validation_controller = self.validation_controller
        return self.validation_controller

    def _ensure_bus_type_predictor(self, ignore_errors: bool = False):
        """Lazy-load the bus type predictor used by warmup/live validation."""
        if self.bus_type_predictor is not None:
            return
        try:
            from application.services.bus_type_predictor import BusTypePredictor

            self.bus_type_predictor = BusTypePredictor()
            self.context.bus_type_predictor = self.bus_type_predictor
        except FileNotFoundError:
            if not ignore_errors:
                raise
        except Exception:
            if not ignore_errors:
                raise

    def _get_current_weather_code(self) -> int:
        try:
            if self.city is None:
                return 0
            for hexagon in self.city.hexagons.values():
                weather = getattr(hexagon, "weather", None)
                if weather is not None:
                    return int(weather.weather_code)
        except Exception:
            pass
        return 0

    @staticmethod
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

    @staticmethod
    def _to_predicted_trip(
        forecast,
        topology,
        weather_code: int,
        bus_type: int,
    ) -> dict:
        stops_map = topology.build_stops_map(forecast.route_id, forecast.direction_id)
        stop_sequence = {}
        for prediction in forecast.stops:
            stop_info = stops_map.get(prediction.stop_sequence, {})
            stop_sequence[prediction.stop_sequence] = {
                "stop_sequence": prediction.stop_sequence,
                "stop_id": stop_info.get("stop_id", ""),
                "stop_name": stop_info.get("stop_name", ""),
                "stop_lat": stop_info.get("stop_lat", 0),
                "stop_lon": stop_info.get("stop_lon", 0),
                "predicted_arrival": prediction.expected_arrival,
                "delay_seconds": prediction.cumulative_delay_sec,
                "confidence_rating": None,
            }

        return {
            "trip_id": f"{forecast.route_id}_{forecast.scheduled_start}",
            "route_id": forecast.route_id,
            "direction_id": forecast.direction_id,
            "trip_date": forecast.trip_date,
            "scheduled_start": forecast.scheduled_start,
            "weather_code": weather_code,
            "bus_type": bus_type,
            "stop_sequence": stop_sequence,
        }
