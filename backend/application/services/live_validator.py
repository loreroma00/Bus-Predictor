"""
Live Validation Session - Real-time validation of predictions against live data.

This module provides continuous monitoring of trip predictions, comparing them
against ground truth data as diaries are completed.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple

from fastapi import WebSocket

from application.domain.internal_events import domain_events, DIARY_FINISHED
from application.domain.observers import Diary


@dataclass
class TripValidationResult:
    trip_id: str
    route_id: str
    direction_id: int
    scheduled_start: str
    mse: float
    rmse: float
    n_measurements: int
    delay_errors: List[float] = field(default_factory=list)
    occupancy_matches: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class LiveValidationStatus:
    session_id: str
    date: str
    status: str
    total_scheduled: int
    total_predicted: int
    total_validated: int
    total_pending: int
    median_mse: float
    median_rmse: float
    min_mse: float
    max_mse: float
    min_rmse: float
    max_rmse: float
    started_at: str
    stops_at: str
    log_file: str
    report_file: str


class LiveValidationSession:
    """
    Manages a live validation session for a specific date.

    Lifecycle:
    1. Created with date and dependencies
    2. Loads scheduled trips and runs predictions upfront
    3. Subscribes to DIARY_FINISHED events
    4. Validates each completed diary against predictions
    5. Broadcasts updates via WebSocket
    6. Stops at 4AM next day or when all trips validated
    """

    def __init__(
        self,
        session_id: str,
        target_date: str,
        predictor,
        observatory,
        weather_service=None,
        bus_type_predictor=None,
    ):
        self.session_id = session_id
        self.target_date = target_date
        self.predictor = predictor
        self.observatory = observatory
        self.weather_service = weather_service
        self.bus_type_predictor = bus_type_predictor

        self.status = "created"
        self.predicted_trips: Dict[str, Any] = {}
        self.validated_trips: List[TripValidationResult] = []
        self.pending_trip_ids: Set[str] = set()
        self.failed_trip_ids: Set[str] = set()

        self.websockets: List[WebSocket] = []
        self._stop_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None

        self.started_at: Optional[datetime] = None
        self.stops_at: Optional[datetime] = None

        self.logger = logging.getLogger(f"live_validator.{session_id[:8]}")

    async def start(self) -> bool:
        """
        Start the validation session.

        Returns True if started successfully, False otherwise.
        """
        if self.status != "created":
            self.logger.warning(f"Cannot start session in status: {self.status}")
            return False

        self.started_at = datetime.now()

        target_dt = datetime.strptime(self.target_date, "%d-%m-%Y")
        self.stops_at = target_dt + timedelta(days=1, hours=4)

        self.status = "predicting"
        await self._broadcast_status()

        try:
            scheduled_trips = self._get_scheduled_trips()
            self.logger.info(
                "Session start: date=%s, scheduled_trips=%d",
                self.target_date,
                len(scheduled_trips),
            )

            if not scheduled_trips:
                self.status = "failed"
                await self._broadcast_error("No scheduled trips found for date")
                return False

            self.pending_trip_ids = {t["trip_id"] for t in scheduled_trips}
            self.logger.info(
                "Pending trip ids initialized: %d", len(self.pending_trip_ids)
            )

            weather_code = self._get_weather_code(target_dt)
            self.logger.info("Using weather_code=%s", weather_code)

            prediction_results = await self._run_predictions(
                scheduled_trips, weather_code
            )

            for trip in scheduled_trips:
                trip_id = trip["trip_id"]
                if trip_id in prediction_results:
                    self.predicted_trips[trip_id] = {
                        "forecast": prediction_results[trip_id],
                        "request": trip,
                    }
                else:
                    self.failed_trip_ids.add(trip_id)
                    self.pending_trip_ids.discard(trip_id)

            self.status = "monitoring"
            await self._broadcast_status()

            domain_events.subscribe(DIARY_FINISHED, self._on_diary_finished)

            self._monitor_task = asyncio.create_task(self._monitor_loop())

            self.logger.info(
                f"Session started: {len(self.predicted_trips)} trips predicted, "
                f"{len(self.failed_trip_ids)} failed, stops at {self.stops_at}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            self.status = "failed"
            await self._broadcast_error(str(e))
            return False

    async def stop(self):
        """Stop the validation session."""
        if self.status in ["completed", "stopped", "failed"]:
            return

        self.logger.info("Stopping session...")
        self._stop_event.set()

        domain_events.unsubscribe(DIARY_FINISHED, self._on_diary_finished)

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.status = "stopped"
        await self._write_output_files()
        await self._broadcast_status()

    async def add_websocket(self, websocket: WebSocket):
        """Add a WebSocket connection for updates."""
        self.websockets.append(websocket)

        await websocket.send_json(
            {
                "type": "connected",
                "session_id": self.session_id,
                "status": self.status,
            }
        )

        await self._send_current_state(websocket)

    def remove_websocket(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.websockets:
            self.websockets.remove(websocket)

    def get_status(self) -> LiveValidationStatus:
        """Get current session status."""
        valid_results = [r for r in self.validated_trips if r.n_measurements > 0]

        mse_values = [r.mse for r in valid_results if r.mse > 0]
        rmse_values = [r.rmse for r in valid_results if r.rmse > 0]

        if mse_values:
            import numpy as np

            median_mse = float(np.median(mse_values))
            min_mse = float(np.min(mse_values))
            max_mse = float(np.max(mse_values))
        else:
            median_mse = min_mse = max_mse = 0.0

        if rmse_values:
            import numpy as np

            median_rmse = float(np.median(rmse_values))
            min_rmse = float(np.min(rmse_values))
            max_rmse = float(np.max(rmse_values))
        else:
            median_rmse = min_rmse = max_rmse = 0.0

        date_yyyymmdd = datetime.strptime(self.target_date, "%d-%m-%Y").strftime(
            "%Y%m%d"
        )

        return LiveValidationStatus(
            session_id=self.session_id,
            date=self.target_date,
            status=self.status,
            total_scheduled=len(self.predicted_trips) + len(self.failed_trip_ids),
            total_predicted=len(self.predicted_trips),
            total_validated=len(self.validated_trips),
            total_pending=len(self.pending_trip_ids),
            median_mse=median_mse,
            median_rmse=median_rmse,
            min_mse=min_mse,
            max_mse=max_mse,
            min_rmse=min_rmse,
            max_rmse=max_rmse,
            started_at=self.started_at.isoformat() if self.started_at else "",
            stops_at=self.stops_at.isoformat() if self.stops_at else "",
            log_file=f"validation_live_{date_yyyymmdd}.log",
            report_file=f"validation_live_{date_yyyymmdd}_report.txt",
        )

    def _get_scheduled_trips(self) -> List[Dict]:
        """Extract all trips scheduled for the target date."""
        target_dt = datetime.strptime(self.target_date, "%d-%m-%Y")
        date_yyyymmdd = target_dt.strftime("%Y%m%d")

        ledger = self.observatory.get_ledger()
        scheduled = []

        for trip_id, trip in ledger["trips"].items():
            if date_yyyymmdd in trip.dates:
                stop_times = trip.get_stop_times() or []
                if stop_times:
                    first_stop = stop_times[0]
                    arrival_time = first_stop.get("arrival_time", "")
                    start_time = (
                        arrival_time[:5] if len(arrival_time) >= 5 else arrival_time
                    )

                    scheduled.append(
                        {
                            "trip_id": trip_id,
                            "route_id": trip.route.id,
                            "direction_id": trip.direction_id,
                            "start_time": start_time,
                            "start_date": self.target_date,
                        }
                    )

        self.logger.info(
            "Scheduled trips extracted for %s: %d",
            self.target_date,
            len(scheduled),
        )
        if scheduled:
            preview = scheduled[:3]
            self.logger.debug("Scheduled trips preview: %s", preview)
        return scheduled

    def _get_weather_code(self, target_dt: datetime) -> int:
        """Get weather code for the date."""
        if self.weather_service is None:
            return 0

        try:
            weather = self.weather_service.get_weather()
            return weather.weather_code
        except Exception as e:
            self.logger.warning(f"Could not fetch weather: {e}")
            return 0

    def _compute_bus_type(self, trip_data: Dict) -> int:
        """Predict bus type for a trip."""
        if self.bus_type_predictor is None:
            return 0

        try:
            trip_date = datetime.strptime(trip_data["start_date"], "%d-%m-%Y").date()
            return self.bus_type_predictor.predict(
                route_id=trip_data["route_id"],
                start_time=trip_data["start_time"],
                trip_date=trip_date,
            )
        except Exception as e:
            self.logger.warning(f"Bus type prediction failed: {e}")
            return 0

    async def _run_predictions(
        self, trips: List[Dict], weather_code: int
    ) -> Dict[str, Any]:
        """Run batch predictions for all trips."""
        if not trips:
            return {}

        results = {}

        try:
            start_ts = time.perf_counter()
            prediction_requests = []
            for trip in trips:
                prediction_requests.append(
                    {
                        "route_id": trip["route_id"],
                        "direction_id": trip["direction_id"],
                        "start_date": trip["start_date"],
                        "start_time": trip["start_time"],
                        "weather_code": weather_code,
                        "bus_type": self._compute_bus_type(trip),
                    }
                )

            self.logger.info(
                "Starting batch forecast for %d trips (weather_code=%s)",
                len(prediction_requests),
                weather_code,
            )
            if prediction_requests:
                self.logger.debug(
                    "Prediction requests preview: %s",
                    prediction_requests[:3],
                )

            forecasts = self.predictor.get_batch_forecast(prediction_requests)
            elapsed = time.perf_counter() - start_ts
            self.logger.info(
                "Batch forecast completed: %d forecasts in %.2fs",
                len(forecasts),
                elapsed,
            )

            for i, forecast in enumerate(forecasts):
                trip_id = trips[i]["trip_id"]
                results[trip_id] = forecast

            self.logger.info(f"Predicted {len(results)} trips successfully")

        except Exception:
            self.logger.exception("Batch prediction failed for %d trips", len(trips))
            if trips:
                self.logger.error("First trip in failed batch: %s", trips[0])

        return results

    def _on_diary_finished(self, event_data: Dict):
        """
        Handle DIARY_FINISHED event.

        This is called synchronously from the event bus, so we need to
        schedule the async validation work.
        """
        if self.status != "monitoring":
            return

        diary: Diary = event_data.get("diary")
        if not diary:
            return

        trip_id = diary.trip_id

        if trip_id not in self.predicted_trips:
            self.logger.debug(f"Ignoring diary for unknown trip: {trip_id}")
            return

        asyncio.create_task(self._validate_diary(trip_id, diary))

    async def _validate_diary(self, trip_id: str, diary: Diary):
        """Validate a completed diary against the prediction."""
        if trip_id not in self.pending_trip_ids:
            return

        prediction = self.predicted_trips.get(trip_id)
        if not prediction:
            return

        forecast = prediction["forecast"]

        try:
            result = self._compute_validation_result(trip_id, forecast, diary)

            self.validated_trips.append(result)
            self.pending_trip_ids.discard(trip_id)

            await self._broadcast_trip_validated(result)

            if not self.pending_trip_ids:
                await self._complete()

        except Exception as e:
            self.logger.error(f"Validation failed for {trip_id}: {e}")

    def _compute_validation_result(
        self, trip_id: str, forecast, diary: Diary
    ) -> TripValidationResult:
        """Compute validation metrics for a trip."""
        ledger = self.observatory.get_ledger()

        stops_map = self._build_stops_map(
            forecast.route_id, forecast.direction_id, ledger
        )

        delay_errors = []
        occupancy_matches = []

        for measurement in diary.measurements:
            matched_stop = self._match_measurement_to_stop(
                measurement, forecast.stops, stops_map
            )

            if matched_stop is None:
                continue

            predicted_delay = matched_stop.cumulative_delay_sec
            actual_delay = measurement.schedule_adherence

            if actual_delay is not None:
                delay_errors.append((predicted_delay - actual_delay) ** 2)

            predicted_crowd = matched_stop.crowd_level
            actual_occupancy = measurement.occupancy_status

            if actual_occupancy is not None and 0 <= int(actual_occupancy) <= 6:
                occupancy_matches.append((predicted_crowd, int(actual_occupancy)))

        if delay_errors:
            mse = sum(delay_errors) / len(delay_errors)
            rmse = mse**0.5
        else:
            mse = 0.0
            rmse = 0.0

        return TripValidationResult(
            trip_id=trip_id,
            route_id=forecast.route_id,
            direction_id=forecast.direction_id,
            scheduled_start=forecast.scheduled_start,
            mse=mse,
            rmse=rmse,
            n_measurements=len(delay_errors),
            delay_errors=[e**0.5 for e in delay_errors],
            occupancy_matches=occupancy_matches,
        )

    def _build_stops_map(self, route_id: str, direction_id: int, ledger: Dict) -> Dict:
        """Build a mapping of stop_sequence to stop info."""
        stops_map = {}

        for trip_id, trip in ledger["trips"].items():
            if trip.route.id == route_id and trip.direction_id == direction_id:
                for st in trip.get_stop_times() or []:
                    seq = int(st.get("stop_sequence", 0) or 0)
                    if seq not in stops_map:
                        stop_id = st.get("stop_id")
                        stop_info = ledger["stops"].get(stop_id, {})
                        shape_dist = st.get("shape_dist_travelled")
                        stops_map[seq] = {
                            "stop_id": stop_id or "",
                            "stop_name": stop_info.get("stop_name", ""),
                            "shape_dist_travelled": float(shape_dist)
                            if shape_dist
                            else None,
                        }
                break

        return stops_map

    def _match_measurement_to_stop(
        self, measurement, predicted_stops, stops_map
    ) -> Optional[Any]:
        """Match a measurement to the closest predicted stop."""
        gt_next_stop = measurement.next_stop

        for seq, stop_info in stops_map.items():
            if stop_info["stop_id"] == str(gt_next_stop):
                for pred_stop in predicted_stops:
                    if pred_stop.stop_sequence == seq:
                        return pred_stop

        gt_next_stop_dist = measurement.next_stop_distance
        if gt_next_stop_dist is not None:
            for seq, stop_info in stops_map.items():
                shape_dist = stop_info.get("shape_dist_travelled")
                if shape_dist is not None:
                    estimated_position = shape_dist - gt_next_stop_dist
                    closest_stop = min(
                        predicted_stops,
                        key=lambda s: abs(s.distance_m - estimated_position),
                    )
                    return closest_stop

        if predicted_stops:
            return predicted_stops[0]

        return None

    async def _monitor_loop(self):
        """Background loop to check for timeout."""
        while not self._stop_event.is_set():
            now = datetime.now()

            if self.stops_at and now >= self.stops_at:
                self.logger.info("Session timeout reached (4AM)")
                await self._complete()
                return

            await self._broadcast_progress()

            await asyncio.sleep(60)

    async def _complete(self):
        """Complete the session."""
        if self.status in ["completed", "stopped"]:
            return

        self.logger.info("Completing session...")

        domain_events.unsubscribe(DIARY_FINISHED, self._on_diary_finished)

        self._stop_event.set()

        self.status = "completed"
        await self._write_output_files()
        await self._broadcast_completed()

    async def _write_output_files(self):
        """Write log and report files."""
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        date_yyyymmdd = datetime.strptime(self.target_date, "%d-%m-%Y").strftime(
            "%Y%m%d"
        )
        log_path = results_dir / f"validation_live_{date_yyyymmdd}.log"
        report_path = results_dir / f"validation_live_{date_yyyymmdd}_report.txt"

        with open(log_path, "w") as f:
            f.write(f"=== Live Validation for {self.target_date} ===\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(
                f"Started: {self.started_at.isoformat() if self.started_at else 'N/A'}\n"
            )
            f.write(f"Stopped: {datetime.now().isoformat()}\n\n")

            f.write(
                f"Total scheduled: {len(self.predicted_trips) + len(self.failed_trip_ids)}\n"
            )
            f.write(f"Total predicted: {len(self.predicted_trips)}\n")
            f.write(f"Total validated: {len(self.validated_trips)}\n")
            f.write(f"Total pending: {len(self.pending_trip_ids)}\n\n")

            for result in self.validated_trips:
                f.write(f"--- Trip {result.trip_id} ---\n")
                f.write(f"Route: {result.route_id}, Direction: {result.direction_id}\n")
                f.write(f"Scheduled Start: {result.scheduled_start}\n")
                f.write(f"Measurements: {result.n_measurements}\n")
                f.write(f"MSE: {result.mse:.2f} sec^2, RMSE: {result.rmse:.2f} sec\n\n")

            for trip_id in self.failed_trip_ids:
                f.write(f"--- Failed Trip {trip_id} ---\n")
                f.write("Prediction failed\n\n")

        with open(report_path, "w") as f:
            status = self.get_status()

            f.write("=" * 60 + "\n")
            f.write(f"LIVE VALIDATION REPORT: {self.target_date}\n")
            f.write("=" * 60 + "\n\n")

            f.write("SESSION INFO\n")
            f.write("-" * 40 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(
                f"Started: {self.started_at.isoformat() if self.started_at else 'N/A'}\n"
            )
            f.write(f"Status: {self.status}\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Scheduled trips:     {status.total_scheduled}\n")
            f.write(f"Predicted trips:     {status.total_predicted}\n")
            f.write(f"Validated trips:     {status.total_validated}\n")
            f.write(f"Pending trips:       {status.total_pending}\n\n")

            f.write("DELAY PREDICTION (seconds)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Median MSE:     {status.median_mse:.2f}\n")
            f.write(f"Median RMSE:    {status.median_rmse:.2f}\n")
            f.write(f"Min RMSE:       {status.min_rmse:.2f}\n")
            f.write(f"Max RMSE:       {status.max_rmse:.2f}\n\n")

            confusion = self._build_confusion_matrix()

            f.write("OCCUPANCY CONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")

            class_labels = [
                "EMPTY",
                "MANY_SEATS",
                "FEW_SEATS",
                "STANDING",
                "CRUSHED",
                "FULL",
                "NOT_ACCEPT",
            ]

            f.write("         " + "  ".join(f"{i:>5}" for i in range(7)) + "\n")
            f.write("Actual:\n")

            for i, row in enumerate(confusion):
                label = class_labels[i][:7] if i < len(class_labels) else str(i)
                f.write(f"  {label:7} " + "  ".join(f"{v:>5}" for v in row) + "\n")

            f.write("\n" + "=" * 60 + "\n")

        self.logger.info(f"Output files written: {log_path}, {report_path}")

    def _build_confusion_matrix(self) -> List[List[int]]:
        """Build confusion matrix for occupancy predictions."""
        num_classes = 7
        cm = [[0] * num_classes for _ in range(num_classes)]

        for result in self.validated_trips:
            for pred, actual in result.occupancy_matches:
                if 0 <= pred < num_classes and 0 <= actual < num_classes:
                    cm[actual][pred] += 1

        return cm

    async def _broadcast(self, message: Dict):
        """Broadcast a message to all connected WebSockets."""
        disconnected = []

        for ws in self.websockets:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.remove_websocket(ws)

    async def _broadcast_status(self):
        """Broadcast current status."""
        status = self.get_status()
        await self._broadcast(
            {
                "type": "status",
                "status": status.status,
                "total_scheduled": status.total_scheduled,
                "total_predicted": status.total_predicted,
                "total_validated": status.total_validated,
                "total_pending": status.total_pending,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def _broadcast_progress(self):
        """Broadcast progress update."""
        await self._broadcast(
            {
                "type": "progress",
                "total_scheduled": len(self.predicted_trips)
                + len(self.failed_trip_ids),
                "total_validated": len(self.validated_trips),
                "total_pending": len(self.pending_trip_ids),
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def _broadcast_trip_validated(self, result: TripValidationResult):
        """Broadcast a trip validation result."""
        await self._broadcast(
            {
                "type": "trip_validated",
                "trip_id": result.trip_id,
                "route_id": result.route_id,
                "direction_id": result.direction_id,
                "scheduled_start": result.scheduled_start,
                "mse": result.mse,
                "rmse": result.rmse,
                "n_measurements": result.n_measurements,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def _broadcast_error(self, error: str):
        """Broadcast an error."""
        await self._broadcast(
            {
                "type": "error",
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def _broadcast_completed(self):
        """Broadcast completion."""
        status = self.get_status()
        await self._broadcast(
            {
                "type": "completed",
                "median_mse": status.median_mse,
                "median_rmse": status.median_rmse,
                "total_validated": status.total_validated,
                "log_file": status.log_file,
                "report_file": status.report_file,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def _send_current_state(self, websocket: WebSocket):
        """Send current state to a newly connected WebSocket."""
        status = self.get_status()

        await websocket.send_json(
            {
                "type": "status",
                "status": status.status,
                "total_scheduled": status.total_scheduled,
                "total_predicted": status.total_predicted,
                "total_validated": status.total_validated,
                "total_pending": status.total_pending,
                "median_mse": status.median_mse,
                "median_rmse": status.median_rmse,
                "started_at": status.started_at,
                "stops_at": status.stops_at,
                "timestamp": datetime.now().isoformat(),
            }
        )

        for result in self.validated_trips:
            await websocket.send_json(
                {
                    "type": "trip_validated",
                    "trip_id": result.trip_id,
                    "route_id": result.route_id,
                    "direction_id": result.direction_id,
                    "scheduled_start": result.scheduled_start,
                    "mse": result.mse,
                    "rmse": result.rmse,
                    "n_measurements": result.n_measurements,
                    "timestamp": datetime.now().isoformat(),
                }
            )
