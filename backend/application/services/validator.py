"""
Validation services for historical and live model checks.

The shared flow is:
1. Use the already-loaded Predictor/LoadedModel.
2. Build the trips to predict.
3. Run model inference for each trip.
4. Compare forecasts with historical observations.
5. Write validation outputs.
"""

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from application.domain.internal_events import LIVE_TRIP_FINISHED, domain_events
from application.domain.live_data import LiveTrip
from application.services.persistence_gateway import get_persistence_gateway

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DIARIES_PATH = PROJECT_ROOT / "diaries" / "diaries.parquet"
MIN_MEASUREMENTS = 7


@dataclass
class TripValidationResult:
    """Per-trip validation outcome."""

    trip_id: str
    route_id: str
    direction_id: int
    scheduled_start: str
    mse: float
    rmse: float
    n_measurements: int
    delay_errors: List[float] = field(default_factory=list)
    occupancy_matches: List[Tuple[int, int]] = field(default_factory=list)
    error: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregated historical validation results for a target date."""

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
    occupancy_confusion_matrix: List[List[int]]
    trips: List[TripValidationResult]
    log_file: str
    report_file: str


@dataclass
class LiveValidationStatus:
    """Snapshot of a live validation session."""

    session_id: str
    date: str
    status: str
    total_scheduled: int
    total_predicted: int
    total_validated: int
    total_pending: int
    total_discarded: int
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


@dataclass
class ValidationObservation:
    """Ground-truth observation normalized from dataframe, DB rows, or LiveTrip."""

    next_stop: Optional[str] = None
    next_stop_distance: Optional[float] = None
    stop_sequence: Optional[int] = None
    schedule_adherence: Optional[float] = None
    occupancy_status: Optional[int] = None


@dataclass
class TripValidationChartResult:
    """Output metadata for a prediction-vs-actual chart."""

    trip_id: str
    route_id: str
    output_path: str
    has_predicted: bool
    has_actual: bool
    rmse: Optional[float] = None


@dataclass
class LossPlotResult:
    """Output metadata for a rendered training-loss chart."""

    log_file: str
    output_path: str
    best_epoch: int
    best_validation_loss: float


class Validator(ABC):
    """Common validation workflow shared by historical and live validators."""

    def __init__(
        self,
        predictor,
        observatory,
        bus_type_predictor=None,
        persistence_gateway=None,
        logger_name: str = "validator",
    ):
        """Bind shared dependencies."""
        self.predictor = predictor
        self.observatory = observatory
        self._bus_type_predictor = bus_type_predictor
        self.persistence = persistence_gateway or get_persistence_gateway()
        self.logger = logging.getLogger(logger_name)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def output_prefix(self) -> str:
        """Prefix used for validation output files."""
        raise NotImplementedError

    @property
    def bus_type_predictor(self):
        """Lazy-load the bus type predictor on first use."""
        if self._bus_type_predictor is None:
            from application.services.bus_type_predictor import BusTypePredictor

            try:
                self._bus_type_predictor = BusTypePredictor()
            except FileNotFoundError as e:
                self.logger.warning("Bus type predictor not available: %s", e)
        return self._bus_type_predictor

    def _date_yyyymmdd(self, date_str: str) -> str:
        """Return YYYYMMDD from DD-MM-YYYY."""
        return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y%m%d")

    def _yyyymmdd_to_ddmmyyyy(self, date_str: str) -> str:
        """Convert YYYYMMDD to DD-MM-YYYY."""
        return datetime.strptime(date_str, "%Y%m%d").strftime("%d-%m-%Y")

    def _parse_start_minutes(self, start_hhmm: str) -> Optional[int]:
        """Return minutes since midnight for an HH:MM string."""
        if not start_hhmm or ":" not in str(start_hhmm):
            return None
        try:
            hh_str, mm_str = str(start_hhmm).split(":", 1)
            hh = int(hh_str)
            mm = int(mm_str)
        except (TypeError, ValueError):
            return None
        if mm < 0 or mm > 59 or hh < 0:
            return None
        return hh * 60 + mm

    def _get_scheduled_trips(self, date_str: str) -> List[Dict[str, Any]]:
        """Extract all trips scheduled for a target DD-MM-YYYY date."""
        topology = self.observatory.get_topology()
        date_yyyymmdd = self._date_yyyymmdd(date_str)
        scheduled = []

        for trip_id, trip in list(topology.trips.items()):
            if date_yyyymmdd not in (trip.dates or []):
                continue
            stop_times = trip.get_stop_times() or []
            if not stop_times:
                continue

            arrival_time = stop_times[0].get("arrival_time", "")
            start_time = arrival_time[:5] if len(arrival_time) >= 5 else arrival_time
            start_minutes = self._parse_start_minutes(start_time)
            if start_minutes is None:
                self.logger.warning(
                    "Skipping trip %s: invalid start_time=%s",
                    trip_id,
                    start_time,
                )
                continue

            try:
                direction_id = int(trip.direction_id)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Skipping trip %s: invalid direction_id=%s",
                    trip_id,
                    trip.direction_id,
                )
                continue

            scheduled.append(
                {
                    "trip_id": trip_id,
                    "route_id": trip.route.id,
                    "direction_id": direction_id,
                    "start_time": start_time,
                    "start_minutes": start_minutes,
                    "start_date": date_str,
                }
            )

        self.logger.info("Scheduled trips extracted for %s: %d", date_str, len(scheduled))
        return scheduled

    def _get_any_hexagon_weather(self):
        """Return the first available city hexagon with weather."""
        city = self.observatory.get_city("Rome")
        if city is None:
            return None
        for hexagon in city.hexagons.values():
            if hexagon.weather is not None:
                return hexagon
        return None

    def _get_weather_code_for_datetime(self, target_dt: datetime) -> int:
        """Get weather code from hexagon forecast for the given datetime."""
        try:
            hexagon = self._get_any_hexagon_weather()
            if hexagon is None:
                return 0
            weather = hexagon.get_weather_for_hour_bucket(target_dt.hour)
            if weather is None:
                weather = hexagon.get_weather()
            return int(getattr(weather, "weather_code", 0) or 0)
        except Exception as e:
            self.logger.warning("Could not fetch weather from hexagon: %s", e)
            return 0

    def _get_weather_code_for_trip(self, trip_data: Dict[str, Any]) -> int:
        """Get forecast weather code for a specific trip start."""
        try:
            trip_date = datetime.strptime(trip_data["start_date"], "%d-%m-%Y")
            hhmm = str(trip_data.get("start_time", "00:00"))
            hh_str, mm_str = hhmm.split(":", 1)
            hour = int(hh_str)
            minute = int(mm_str)
            day_offset = hour // 24
            trip_dt = trip_date + timedelta(
                days=day_offset,
                hours=hour % 24,
                minutes=minute,
            )
            return self._get_weather_code_for_datetime(trip_dt)
        except Exception as e:
            self.logger.warning(
                "Could not fetch forecast weather for trip %s: %s",
                trip_data.get("trip_id", "?"),
                e,
            )
            return 0

    def _compute_bus_type(self, trip_data: Dict[str, Any]) -> int:
        """Predict bus type for a trip."""
        if self.bus_type_predictor is None:
            return 0
        try:
            trip_date = datetime.strptime(trip_data["start_date"], "%d-%m-%Y").date()
            return int(
                self.bus_type_predictor.predict(
                    route_id=trip_data["route_id"],
                    start_time=trip_data["start_time"],
                    trip_date=trip_date,
                )
                or 0
            )
        except Exception as e:
            self.logger.warning("Bus type prediction failed: %s", e)
            return 0

    def _supports_trip_template(self, trip: Dict[str, Any]) -> bool:
        """Return whether the predictor has a canonical template for this route/direction."""
        has_template = getattr(self.predictor, "has_trip_template", None)
        if has_template is None:
            return True
        return bool(has_template(trip["route_id"], trip["direction_id"]))

    def _run_predictions(self, trips: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Run batch predictions for trips with shared request construction."""
        results = {"successful": [], "failed": []}
        if not trips:
            return results

        start_ts = time.perf_counter()
        valid_trips = []
        prediction_requests = []

        for trip in trips:
            if not self._supports_trip_template(trip):
                results["failed"].append(
                    {
                        "trip_id": trip["trip_id"],
                        "error": "No canonical template for route/direction",
                    }
                )
                continue

            valid_trips.append(trip)
            prediction_requests.append(
                {
                    "route_id": trip["route_id"],
                    "direction_id": trip["direction_id"],
                    "start_date": trip["start_date"],
                    "start_time": trip["start_time"],
                    "weather_code": self._get_weather_code_for_trip(trip),
                    "bus_type": self._compute_bus_type(trip),
                }
            )

        self.logger.info(
            "Prepared prediction requests: total=%d valid=%d skipped=%d",
            len(trips),
            len(prediction_requests),
            len(results["failed"]),
        )

        chunk_size = 512
        for chunk_start in range(0, len(prediction_requests), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(prediction_requests))
            chunk_requests = prediction_requests[chunk_start:chunk_end]
            chunk_trips = valid_trips[chunk_start:chunk_end]
            try:
                forecasts = self.predictor.get_batch_forecast(chunk_requests)
            except Exception as e:
                self.logger.exception(
                    "Prediction chunk failed for range %d-%d",
                    chunk_start + 1,
                    chunk_end,
                )
                for trip in chunk_trips:
                    results["failed"].append(
                        {"trip_id": trip["trip_id"], "error": str(e)}
                    )
                continue

            forecasts = list(forecasts or [])
            for trip, request, forecast in zip(chunk_trips, chunk_requests, forecasts):
                results["successful"].append(
                    {
                        "trip_id": trip["trip_id"],
                        "forecast": forecast,
                        "request": request,
                        "trip": trip,
                    }
                )
            if len(forecasts) < len(chunk_trips):
                for trip in chunk_trips[len(forecasts) :]:
                    results["failed"].append(
                        {
                            "trip_id": trip["trip_id"],
                            "error": "Predictor returned no forecast for trip",
                        }
                    )

        elapsed = time.perf_counter() - start_ts
        self.logger.info(
            "Batch forecast completed: %d successful, %d failed in %.2fs",
            len(results["successful"]),
            len(results["failed"]),
            elapsed,
        )
        return results

    async def _run_predictions_async(
        self,
        trips: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict]]:
        """Async wrapper around shared batch prediction."""
        return await asyncio.to_thread(self._run_predictions, trips)

    def _load_historical_rows_for_trip(self, trip_id: str) -> List[Dict[str, Any]]:
        """Read one trip from Observatory's historical ledger when available."""
        historical = getattr(self.observatory, "historical", None)
        if historical is None:
            return []

        try:
            rows = historical.get_trip_measurements(trip_id)
            if rows:
                return rows
        except Exception:
            pass

        query = getattr(historical, "query", None)
        if query is None:
            return []
        try:
            df = query(trip_id=trip_id)
            if df is not None and not df.empty:
                return df.to_dict("records")
        except Exception as e:
            self.logger.debug("Historical query failed for trip %s: %s", trip_id, e)
        return []

    def _observations_from_dataframe(self, df: pd.DataFrame) -> List[ValidationObservation]:
        """Normalize dataframe rows into validation observations."""
        if df is None or df.empty:
            return []
        return [self._observation_from_row(row) for _, row in df.iterrows()]

    def _observations_from_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[ValidationObservation]:
        """Normalize DB/ledger rows into validation observations."""
        return [self._observation_from_row(row) for row in rows or []]

    def _observations_from_live_trip(
        self,
        live_trip: LiveTrip,
    ) -> List[ValidationObservation]:
        """Normalize a completed LiveTrip into validation observations."""
        observations = []
        for measurement in live_trip.measurements:
            gpsdata = getattr(measurement, "gpsdata", None)
            observations.append(
                ValidationObservation(
                    next_stop=str(measurement.next_stop)
                    if measurement.next_stop is not None
                    else None,
                    next_stop_distance=measurement.next_stop_distance,
                    stop_sequence=getattr(gpsdata, "current_stop_sequence", None),
                    schedule_adherence=measurement.schedule_adherence,
                    occupancy_status=measurement.occupancy_status,
                )
            )
        return observations

    def _observation_from_row(self, row) -> ValidationObservation:
        """Normalize one pandas/DB row."""
        next_stop_distance = self._row_get(row, "distance_to_next_stop")
        if next_stop_distance is None:
            next_stop_distance = self._row_get(row, "next_stop_distance")

        occupancy = self._row_get(row, "occupancy_status")
        if occupancy is None:
            occupancy = self._row_get(row, "occupancy")

        next_stop = self._row_get(row, "next_stop")

        return ValidationObservation(
            next_stop=str(next_stop) if next_stop is not None else None,
            next_stop_distance=next_stop_distance,
            stop_sequence=self._row_get(row, "stop_sequence"),
            schedule_adherence=self._row_get(row, "schedule_adherence"),
            occupancy_status=occupancy,
        )

    def _row_get(self, row, key: str, default=None):
        """Read a field from dicts, pandas rows, or asyncpg records."""
        if hasattr(row, "get"):
            return row.get(key, default)
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            return default

    def _compute_validation_result(
        self,
        trip_id: str,
        forecast,
        observations: List[ValidationObservation],
    ) -> TripValidationResult:
        """Compute validation metrics for one forecast and normalized observations."""
        topology = self.observatory.get_topology()
        stops_map = topology.build_stops_map(forecast.route_id, forecast.direction_id)
        pred_by_stop_seq = {int(stop.stop_sequence): stop for stop in forecast.stops}

        delay_squared_errors = []
        occupancy_matches = []
        telemetry = {
            "trip_id": trip_id,
            "route_id": forecast.route_id,
            "predicted_curve": [
                float(stop.cumulative_delay_sec) for stop in forecast.stops
            ],
            "actual_measurements": [],
        }

        for observation in observations:
            matched_stop = self._match_observation_to_stop(
                observation,
                forecast.stops,
                stops_map,
                pred_by_stop_seq,
            )
            if matched_stop is None:
                continue

            actual_delay = observation.schedule_adherence
            if self._is_valid_actual_delay(actual_delay):
                err = float(matched_stop.cumulative_delay_sec) - float(actual_delay)
                delay_squared_errors.append(err * err)
                telemetry["actual_measurements"].append(
                    {
                        "segment_idx": int(matched_stop.stop_sequence),
                        "actual_delay": float(actual_delay),
                    }
                )

            actual_occupancy = observation.occupancy_status
            if actual_occupancy is not None and pd.notna(actual_occupancy):
                actual_occupancy = int(actual_occupancy)
                if 0 <= actual_occupancy <= 6:
                    occupancy_matches.append(
                        (int(matched_stop.crowd_level), actual_occupancy)
                    )

        if delay_squared_errors:
            mse = sum(delay_squared_errors) / len(delay_squared_errors)
            rmse = mse**0.5
        else:
            mse = 0.0
            rmse = 0.0

        telemetry["rmse"] = rmse
        telemetry["num_measurements"] = len(delay_squared_errors)

        return TripValidationResult(
            trip_id=trip_id,
            route_id=forecast.route_id,
            direction_id=forecast.direction_id,
            scheduled_start=forecast.scheduled_start,
            mse=mse,
            rmse=rmse,
            n_measurements=len(delay_squared_errors),
            delay_errors=[e**0.5 for e in delay_squared_errors],
            occupancy_matches=occupancy_matches,
            telemetry=telemetry,
        )

    def _is_valid_actual_delay(self, actual_delay) -> bool:
        """Return whether an actual delay value is usable for metrics."""
        if actual_delay is None or pd.isna(actual_delay):
            return False
        actual_delay = float(actual_delay)
        return actual_delay != -1000.0 and abs(actual_delay) <= 7200

    def _match_observation_to_stop(
        self,
        observation: ValidationObservation,
        predicted_stops: List[Any],
        stops_map: Dict[int, Dict],
        pred_by_stop_seq: Dict[int, Any],
    ) -> Optional[Any]:
        """Match an observation to the closest predicted stop."""
        if observation.stop_sequence is not None and pd.notna(observation.stop_sequence):
            predicted_stop = pred_by_stop_seq.get(int(observation.stop_sequence))
            if predicted_stop is not None:
                return predicted_stop

        if observation.next_stop is not None:
            for seq, stop_info in stops_map.items():
                if stop_info["stop_id"] == str(observation.next_stop):
                    predicted_stop = pred_by_stop_seq.get(int(seq))
                    if predicted_stop is not None:
                        return predicted_stop

        if (
            observation.next_stop_distance is not None
            and pd.notna(observation.next_stop_distance)
            and predicted_stops
        ):
            for _, stop_info in stops_map.items():
                shape_dist = stop_info.get("shape_dist_travelled")
                if shape_dist is not None:
                    estimated_position = float(shape_dist) - float(
                        observation.next_stop_distance
                    )
                    return min(
                        predicted_stops,
                        key=lambda s: abs(s.distance_m - estimated_position),
                    )

        return predicted_stops[0] if predicted_stops else None

    def _metric_summary(
        self,
        results: List[TripValidationResult],
    ) -> Dict[str, float]:
        """Compute aggregate MSE/RMSE metrics."""
        valid_results = [r for r in results if r.error is None and r.n_measurements > 0]
        mse_values = [r.mse for r in valid_results if r.mse > 0]
        rmse_values = [r.rmse for r in valid_results if r.rmse > 0]

        if mse_values:
            median_mse = float(np.median(mse_values))
            min_mse = float(np.min(mse_values))
            max_mse = float(np.max(mse_values))
        else:
            median_mse = min_mse = max_mse = 0.0

        if rmse_values:
            median_rmse = float(np.median(rmse_values))
            min_rmse = float(np.min(rmse_values))
            max_rmse = float(np.max(rmse_values))
        else:
            median_rmse = min_rmse = max_rmse = 0.0

        return {
            "median_mse": median_mse,
            "median_rmse": median_rmse,
            "min_mse": min_mse,
            "max_mse": max_mse,
            "min_rmse": min_rmse,
            "max_rmse": max_rmse,
        }

    def _build_confusion_matrix(
        self,
        results: List[TripValidationResult],
    ) -> List[List[int]]:
        """Build confusion matrix for occupancy predictions."""
        num_classes = 7
        cm = [[0] * num_classes for _ in range(num_classes)]
        for result in results:
            for pred, actual in result.occupancy_matches:
                if 0 <= pred < num_classes and 0 <= actual < num_classes:
                    cm[actual][pred] += 1
        return cm


class HistoricalValidator(Validator):
    """Historical validator that compares forecasts with stored observations."""

    @property
    def output_prefix(self) -> str:
        """Prefix used for historical validation output files."""
        return "validation"

    def __init__(
        self,
        predictor,
        observatory,
        bus_type_predictor=None,
        persistence_gateway=None,
    ):
        super().__init__(
            predictor=predictor,
            observatory=observatory,
            bus_type_predictor=bus_type_predictor,
            persistence_gateway=persistence_gateway,
            logger_name="validator.historical",
        )

    def validate_date(self, date_str: str) -> ValidationReport:
        """Validate all predictions for a DD-MM-YYYY date."""
        self.logger.info("Starting historical validation for %s", date_str)
        date_yyyymmdd = self._date_yyyymmdd(date_str)

        scheduled_trips = self._get_scheduled_trips(date_str)
        ground_truth = self._load_ground_truth(date_yyyymmdd)
        self.logger.info("Loaded %d historical measurements", len(ground_truth))

        trips_with_gt = self._match_scheduled_to_ground_truth(
            scheduled_trips,
            ground_truth,
        )
        self.logger.info("Matched %d trips with ground truth", len(trips_with_gt))

        prediction_results = self._run_predictions(trips_with_gt)
        validation_results = self._validate_prediction_results(
            prediction_results,
            ground_truth,
        )

        report = self._build_report(
            date_str=date_str,
            scheduled_trips=scheduled_trips,
            trips_with_gt=trips_with_gt,
            validation_results=validation_results,
            ground_truth=ground_truth,
        )
        self._write_log(report, validation_results)
        self._write_report(report)
        return report

    def _load_ground_truth(self, date_yyyymmdd: str) -> pd.DataFrame:
        """Load historical measurements for the given date."""
        target_date = datetime.strptime(date_yyyymmdd, "%Y%m%d")
        start_ts = target_date.timestamp()
        end_ts = (target_date + timedelta(days=1)).timestamp()

        historical = getattr(self.observatory, "historical", None)
        if historical is not None and hasattr(historical, "query"):
            try:
                df = historical.query(date_start=start_ts, date_end=end_ts)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                self.logger.warning("Historical ledger query failed: %s", e)

        if not DIARIES_PATH.exists():
            self.logger.warning("Diaries file not found: %s", DIARIES_PATH)
            return pd.DataFrame()

        df = pd.read_parquet(DIARIES_PATH)
        return df[
            (df["measurement_time"] >= start_ts) & (df["measurement_time"] < end_ts)
        ]

    def _match_scheduled_to_ground_truth(
        self,
        scheduled_trips: List[Dict[str, Any]],
        ground_truth: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Find scheduled trips that have ground truth data."""
        if ground_truth.empty or "trip_id" not in ground_truth.columns:
            return []
        gt_trip_ids = set(ground_truth["trip_id"].astype(str).unique())
        return [trip for trip in scheduled_trips if str(trip["trip_id"]) in gt_trip_ids]

    def _validate_prediction_results(
        self,
        prediction_results: Dict[str, List[Dict]],
        ground_truth: pd.DataFrame,
    ) -> List[TripValidationResult]:
        """Validate prediction batch output against historical rows."""
        results = []
        for pred in prediction_results["successful"]:
            trip_id = pred["trip_id"]
            forecast = pred["forecast"]
            trip_gt = ground_truth[ground_truth["trip_id"].astype(str) == str(trip_id)]
            observations = self._observations_from_dataframe(trip_gt)
            if not observations:
                results.append(
                    TripValidationResult(
                        trip_id=trip_id,
                        route_id=forecast.route_id,
                        direction_id=forecast.direction_id,
                        scheduled_start=forecast.scheduled_start,
                        mse=0.0,
                        rmse=0.0,
                        n_measurements=0,
                        error="No ground truth measurements",
                    )
                )
                continue
            results.append(
                self._compute_validation_result(trip_id, forecast, observations)
            )

        for failed in prediction_results["failed"]:
            results.append(
                TripValidationResult(
                    trip_id=failed["trip_id"],
                    route_id="",
                    direction_id=0,
                    scheduled_start="",
                    mse=0.0,
                    rmse=0.0,
                    n_measurements=0,
                    error=failed["error"],
                )
            )
        return results

    def _build_report(
        self,
        date_str: str,
        scheduled_trips: List[Dict[str, Any]],
        trips_with_gt: List[Dict[str, Any]],
        validation_results: List[TripValidationResult],
        ground_truth: pd.DataFrame,
    ) -> ValidationReport:
        """Build the final validation report."""
        valid_results = [
            r for r in validation_results if r.error is None and r.n_measurements > 0
        ]
        metrics = self._metric_summary(validation_results)
        date_yyyymmdd = self._date_yyyymmdd(date_str)

        return ValidationReport(
            date=date_str,
            total_scheduled_trips=len(scheduled_trips),
            total_trips_with_ground_truth=len(trips_with_gt),
            total_trips_predicted=len(
                [r for r in validation_results if r.error is None]
            ),
            total_trips_validated=len(valid_results),
            total_measurements=sum(r.n_measurements for r in valid_results),
            median_mse=metrics["median_mse"],
            median_rmse=metrics["median_rmse"],
            min_mse=metrics["min_mse"],
            max_mse=metrics["max_mse"],
            min_rmse=metrics["min_rmse"],
            max_rmse=metrics["max_rmse"],
            occupancy_confusion_matrix=self._build_confusion_matrix(valid_results),
            trips=validation_results,
            log_file=f"{self.output_prefix}_{date_yyyymmdd}.log",
            report_file=f"{self.output_prefix}_{date_yyyymmdd}_report.txt",
        )

    def _write_log(
        self,
        report: ValidationReport,
        results: List[TripValidationResult],
    ):
        """Write detailed per-trip log file."""
        log_path = RESULTS_DIR / report.log_file
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Validation for {report.date} ===\n\n")
            f.write(f"Total scheduled trips: {report.total_scheduled_trips}\n")
            f.write(f"Trips with ground truth: {report.total_trips_with_ground_truth}\n")
            f.write(f"Trips predicted successfully: {report.total_trips_predicted}\n")
            f.write(f"Trips validated: {report.total_trips_validated}\n\n")

            for result in results:
                f.write(f"--- Trip {result.trip_id} ---\n")
                f.write(
                    f"Route: {result.route_id}, Direction: {result.direction_id}, "
                    f"Start: {result.scheduled_start}\n"
                )
                if result.error:
                    f.write(f"ERROR: {result.error}\n\n")
                    continue
                f.write(f"Measurements: {result.n_measurements}\n")
                f.write(
                    f"Delay MSE: {result.mse:.2f} sec^2, "
                    f"RMSE: {result.rmse:.2f} sec\n"
                )
                if result.occupancy_matches:
                    correct = sum(1 for p, a in result.occupancy_matches if p == a)
                    accuracy = correct / len(result.occupancy_matches)
                    f.write(f"Occupancy accuracy: {accuracy:.2%}\n")
                f.write("\n")
        self.logger.info("Log written to %s", log_path)

    def _write_report(self, report: ValidationReport):
        """Write summary report file."""
        report_path = RESULTS_DIR / report.report_file
        _write_metric_report(
            report_path=report_path,
            title=f"VALIDATION REPORT: {report.date}",
            summary=[
                ("Scheduled trips", report.total_scheduled_trips),
                ("Trips with ground truth", report.total_trips_with_ground_truth),
                ("Validated trips", report.total_trips_validated),
                ("Total measurements", report.total_measurements),
            ],
            metrics={
                "median_mse": report.median_mse,
                "median_rmse": report.median_rmse,
                "min_rmse": report.min_rmse,
                "max_rmse": report.max_rmse,
            },
            confusion_matrix=report.occupancy_confusion_matrix,
        )
        self.logger.info("Report written to %s", report_path)


class LiveValidator(Validator):
    """Live validation session for one target date."""

    @property
    def output_prefix(self) -> str:
        """Prefix used for live validation output files."""
        return "validation_live"

    def __init__(
        self,
        session_id: str,
        target_date: str,
        predictor,
        observatory,
        bus_type_predictor=None,
        persistence_gateway=None,
    ):
        super().__init__(
            predictor=predictor,
            observatory=observatory,
            bus_type_predictor=bus_type_predictor,
            persistence_gateway=persistence_gateway,
            logger_name=f"validator.live.{session_id[:8]}",
        )
        self.session_id = session_id
        self.target_date = target_date
        self.status = "created"
        self.predicted_trips: Dict[str, Any] = {}
        self.validated_trips: List[TripValidationResult] = []
        self.pending_trip_ids: Set[str] = set()
        self._past_trip_ids: Set[str] = set()
        self.failed_trip_ids: Set[str] = set()
        self.discarded_trip_ids: Set[str] = set()
        self.websockets: List[Any] = []
        self._stop_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        self.started_at: Optional[datetime] = None
        self.stops_at: Optional[datetime] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> bool:
        """Start the validation session."""
        if self.status != "created":
            self.logger.warning("Cannot start session in status: %s", self.status)
            return False

        self.started_at = datetime.now()
        self.status = "predicting"
        await self._broadcast_status()

        try:
            scheduled_trips = self._get_scheduled_trips(self.target_date)
            if not scheduled_trips:
                self.status = "failed"
                await self._broadcast_error("No scheduled trips found for date")
                return False

            self.pending_trip_ids = {t["trip_id"] for t in scheduled_trips}
            prediction_results = await self._run_predictions_async(scheduled_trips)

            now_minutes = datetime.now().hour * 60 + datetime.now().minute
            for pred in prediction_results["successful"]:
                trip = pred["trip"]
                trip_id = pred["trip_id"]
                self.predicted_trips[trip_id] = {
                    "forecast": pred["forecast"],
                    "request": pred["request"],
                    "trip": trip,
                }
                if trip["start_minutes"] < now_minutes:
                    self._past_trip_ids.add(trip_id)

            for failed in prediction_results["failed"]:
                trip_id = failed["trip_id"]
                self.failed_trip_ids.add(trip_id)
                self.pending_trip_ids.discard(trip_id)

            self.status = "monitoring"
            await self._broadcast_status()

            await self._resolve_past_trips()

            self._loop = asyncio.get_running_loop()
            domain_events.subscribe(LIVE_TRIP_FINISHED, self._on_live_trip_finished)
            self._monitor_task = asyncio.create_task(self._monitor_loop())

            self.logger.info(
                "Live validation started: predicted=%d failed=%d",
                len(self.predicted_trips),
                len(self.failed_trip_ids),
            )
            return True
        except Exception as e:
            self.logger.exception("Failed to start live validation")
            self.status = "failed"
            await self._broadcast_error(str(e))
            return False

    async def stop(self):
        """Stop the validation session."""
        if self.status in ["completed", "stopped", "failed"]:
            return

        self._stop_event.set()
        domain_events.unsubscribe(LIVE_TRIP_FINISHED, self._on_live_trip_finished)
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.status = "stopped"
        await self._write_output_files()
        await self._broadcast_status()

    async def add_websocket(self, websocket):
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

    def remove_websocket(self, websocket):
        """Remove a WebSocket connection."""
        if websocket in self.websockets:
            self.websockets.remove(websocket)

    def get_status(self) -> LiveValidationStatus:
        """Get current session status."""
        metrics = self._metric_summary(self.validated_trips)
        date_yyyymmdd = self._date_yyyymmdd(self.target_date)
        return LiveValidationStatus(
            session_id=self.session_id,
            date=self.target_date,
            status=self.status,
            total_scheduled=len(self.predicted_trips) + len(self.failed_trip_ids),
            total_predicted=len(self.predicted_trips),
            total_validated=len(self.validated_trips),
            total_pending=len(self.pending_trip_ids),
            total_discarded=len(self.discarded_trip_ids),
            median_mse=metrics["median_mse"],
            median_rmse=metrics["median_rmse"],
            min_mse=metrics["min_mse"],
            max_mse=metrics["max_mse"],
            min_rmse=metrics["min_rmse"],
            max_rmse=metrics["max_rmse"],
            started_at=self.started_at.isoformat() if self.started_at else "",
            stops_at=self.stops_at.isoformat() if self.stops_at else "",
            log_file=f"{self.output_prefix}_{date_yyyymmdd}.log",
            report_file=f"{self.output_prefix}_{date_yyyymmdd}_report.txt",
        )

    async def _resolve_past_trips(self):
        """Resolve already-started trips through historical rows, memory, then DB."""
        if not self._past_trip_ids:
            return

        still_missing = set()
        resolved = 0
        for trip_id in list(self._past_trip_ids):
            if await self._validate_from_historical_rows(trip_id):
                resolved += 1
                continue

            live_trip = self.observatory.search_completed_live_trip(trip_id)
            if live_trip and live_trip.measurements:
                await self._append_live_result(trip_id, live_trip, broadcast=False)
                resolved += 1
            else:
                still_missing.add(trip_id)

        if still_missing:
            resolved_db = await self._resolve_from_db(still_missing)
            for trip_id in list(still_missing):
                if trip_id in self.pending_trip_ids:
                    self.pending_trip_ids.discard(trip_id)
                    self.discarded_trip_ids.add(trip_id)
            self.logger.info(
                "Past trips resolved: ledger/memory=%d db=%d discarded=%d",
                resolved,
                resolved_db,
                len(still_missing),
            )
        self._past_trip_ids.clear()

    async def _validate_from_historical_rows(self, trip_id: str) -> bool:
        """Validate a pending trip from Observatory's historical ledger."""
        rows = self._load_historical_rows_for_trip(trip_id)
        if len(rows) < MIN_MEASUREMENTS:
            return False

        prediction = self.predicted_trips.get(trip_id)
        if not prediction:
            return False

        observations = self._observations_from_rows(rows)
        result = self._compute_validation_result(
            trip_id,
            prediction["forecast"],
            observations,
        )
        self.validated_trips.append(result)
        self.pending_trip_ids.discard(trip_id)
        return True

    async def _resolve_from_db(self, trip_ids: Set[str]) -> int:
        """One-shot DB lookup for trips not found in the historical ledger."""
        resolved = 0
        rows_by_trip = await self.persistence.fetch_validation_rows_by_trip_ids(
            trip_ids,
        )
        for trip_id, rows in rows_by_trip.items():
            if len(rows) < MIN_MEASUREMENTS:
                continue
            prediction = self.predicted_trips.get(trip_id)
            if not prediction:
                continue
            observations = self._observations_from_rows(rows)
            result = self._compute_validation_result(
                trip_id,
                prediction["forecast"],
                observations,
            )
            self.validated_trips.append(result)
            self.pending_trip_ids.discard(trip_id)
            trip_ids.discard(trip_id)
            resolved += 1
        return resolved

    def _on_live_trip_finished(self, event_data: Dict[str, Any]):
        """Schedule validation when a live trip finishes."""
        if self.status != "monitoring":
            return
        live_trip: LiveTrip = event_data.get("live_trip")
        if not live_trip:
            return
        trip_id = live_trip.trip_id
        if trip_id not in self.predicted_trips:
            return
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._validate_live_trip(trip_id, live_trip),
                self._loop,
            )

    async def _validate_live_trip(self, trip_id: str, live_trip: LiveTrip):
        """Validate a completed live trip against the prediction."""
        if trip_id not in self.pending_trip_ids:
            return
        try:
            await asyncio.sleep(0)
            if not await self._validate_from_historical_rows(trip_id):
                await self._append_live_result(trip_id, live_trip, broadcast=False)

            result = self.validated_trips[-1] if self.validated_trips else None
            if result and result.trip_id == trip_id:
                await self._broadcast_trip_validated(result)
            if not self.pending_trip_ids:
                await self._complete()
        except Exception as e:
            self.logger.error("Validation failed for %s: %s", trip_id, e)

    async def _append_live_result(
        self,
        trip_id: str,
        live_trip: LiveTrip,
        broadcast: bool,
    ):
        """Fallback validation from the LiveTrip object itself."""
        prediction = self.predicted_trips.get(trip_id)
        if not prediction:
            return
        observations = self._observations_from_live_trip(live_trip)
        result = self._compute_validation_result(
            trip_id,
            prediction["forecast"],
            observations,
        )
        self.validated_trips.append(result)
        self.pending_trip_ids.discard(trip_id)
        if broadcast:
            await self._broadcast_trip_validated(result)

    async def _monitor_loop(self):
        """Background loop to broadcast progress."""
        while not self._stop_event.is_set():
            await self._broadcast_progress()
            await asyncio.sleep(60)

    async def _complete(self):
        """Complete the live session."""
        if self.status in ["completed", "stopped"]:
            return
        domain_events.unsubscribe(LIVE_TRIP_FINISHED, self._on_live_trip_finished)
        self._stop_event.set()
        self.status = "completed"
        await self._write_output_files()
        await self._broadcast_completed()

    async def _write_output_files(self):
        """Write live log, report, and diagnostics files."""
        status = self.get_status()
        log_path = RESULTS_DIR / status.log_file
        report_path = RESULTS_DIR / status.report_file
        date_yyyymmdd = self._date_yyyymmdd(self.target_date)
        diagnostics_path = RESULTS_DIR / f"{self.output_prefix}_{date_yyyymmdd}_diagnostics.json"

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Live Validation for {self.target_date} ===\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(
                f"Started: {self.started_at.isoformat() if self.started_at else 'N/A'}\n"
            )
            f.write(f"Stopped: {datetime.now().isoformat()}\n\n")
            f.write(f"Total scheduled: {status.total_scheduled}\n")
            f.write(f"Total predicted: {status.total_predicted}\n")
            f.write(f"Total validated: {status.total_validated}\n")
            f.write(f"Total pending: {status.total_pending}\n")
            f.write(f"Total discarded: {status.total_discarded}\n\n")

            for result in self.validated_trips:
                f.write(f"--- Trip {result.trip_id} ---\n")
                f.write(f"Route: {result.route_id}, Direction: {result.direction_id}\n")
                f.write(f"Scheduled Start: {result.scheduled_start}\n")
                f.write(f"Measurements: {result.n_measurements}\n")
                f.write(f"MSE: {result.mse:.2f} sec^2, RMSE: {result.rmse:.2f} sec\n\n")

            for trip_id in self.failed_trip_ids:
                f.write(f"--- Failed Trip {trip_id} ---\n")
                f.write("Prediction failed\n\n")

        _write_metric_report(
            report_path=report_path,
            title=f"LIVE VALIDATION REPORT: {self.target_date}",
            summary=[
                ("Scheduled trips", status.total_scheduled),
                ("Predicted trips", status.total_predicted),
                ("Validated trips", status.total_validated),
                ("Pending trips", status.total_pending),
                ("Discarded trips", status.total_discarded),
            ],
            metrics={
                "median_mse": status.median_mse,
                "median_rmse": status.median_rmse,
                "min_rmse": status.min_rmse,
                "max_rmse": status.max_rmse,
            },
            confusion_matrix=self._build_confusion_matrix(self.validated_trips),
            session_id=self.session_id,
            started_at=self.started_at.isoformat() if self.started_at else "N/A",
            status=self.status,
        )

        diagnostics_data = [
            result.telemetry
            for result in self.validated_trips
            if result.telemetry and result.n_measurements > 0
        ]
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics_data, f, indent=2)

        self.logger.info(
            "Output files written: %s, %s, %s",
            log_path,
            report_path,
            diagnostics_path,
        )

    async def _broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSockets."""
        disconnected = []
        for ws in self.websockets:
            try:
                await ws.send_json(message)
            except Exception as e:
                disconnected.append(ws)
                self.logger.exception(e)
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

    async def _send_current_state(self, websocket):
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


class ValidationController:
    """Lifecycle facade for historical and live validation sessions."""

    def __init__(
        self,
        logger_name: str = "validator.controller",
        persistence_gateway=None,
    ):
        """Initialize controller-owned validation state."""
        self.logger = logging.getLogger(logger_name)
        self.persistence = persistence_gateway or get_persistence_gateway()
        self.historical_thread: Optional[threading.Thread] = None
        self.live_thread: Optional[threading.Thread] = None
        self.live_session: Optional[LiveValidator] = None
        self.last_historical_report: Optional[ValidationReport] = None
        self._live_loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.RLock()

    def run_historical(
        self,
        date_str: str,
        predictor,
        observatory,
        bus_type_predictor=None,
    ) -> ValidationReport:
        """Run historical validation synchronously and return the report."""
        validator = HistoricalValidator(
            predictor=predictor,
            observatory=observatory,
            bus_type_predictor=bus_type_predictor,
            persistence_gateway=self.persistence,
        )
        report = validator.validate_date(date_str)
        with self._lock:
            self.last_historical_report = report
        return report

    def start_historical(
        self,
        date_str: str,
        predictor,
        observatory,
        bus_type_predictor=None,
    ) -> bool:
        """Start historical validation in a background thread."""
        with self._lock:
            if self._thread_running(self.historical_thread):
                self.logger.warning("Historical validation already running.")
                return False

            self.historical_thread = threading.Thread(
                target=self._run_historical_thread,
                args=(date_str, predictor, observatory, bus_type_predictor),
                daemon=True,
            )
            self.historical_thread.start()

        self.logger.info("Historical validation started for %s", date_str)
        return True

    async def start_live_session(
        self,
        date_str: str,
        predictor,
        observatory,
        bus_type_predictor=None,
        session_id: str = None,
    ) -> Optional[LiveValidator]:
        """Start a live validation session on the current asyncio loop."""
        with self._lock:
            if self._live_running_locked():
                self.logger.warning("Live validation already running.")
                return None

            session = self._build_live_session(
                date_str=date_str,
                predictor=predictor,
                observatory=observatory,
                bus_type_predictor=bus_type_predictor,
                session_id=session_id,
            )
            self.live_session = session

        success = await session.start()
        if not success:
            return None
        return session

    def start_live(
        self,
        date_str: str,
        predictor,
        observatory,
        bus_type_predictor=None,
        session_id: str = None,
    ) -> Optional[str]:
        """Start a live validation session in a background thread."""
        with self._lock:
            if self._live_running_locked():
                self.logger.warning("Live validation already running.")
                return None

            session = self._build_live_session(
                date_str=date_str,
                predictor=predictor,
                observatory=observatory,
                bus_type_predictor=bus_type_predictor,
                session_id=session_id,
            )
            self.live_session = session
            self.live_thread = threading.Thread(
                target=self._run_live_thread,
                args=(session,),
                daemon=True,
            )
            self.live_thread.start()

        self.logger.info(
            "Live validation started for %s (session %s)",
            date_str,
            session.session_id[:8],
        )
        return session.session_id

    async def stop_live_session(self) -> bool:
        """Stop the current live session on the current asyncio loop."""
        session = self.get_live_session()
        if session is None:
            self.logger.info("No live validation session running.")
            return False

        await session.stop()
        return True

    def stop_live(self, timeout: float = 10.0) -> bool:
        """Stop the current live validation session from another thread."""
        session = self.get_live_session()
        if session is None:
            self.logger.info("No live validation session running.")
            return False
        if session.status in ("completed", "stopped", "failed"):
            self.logger.info("No live validation session running.")
            return False

        loop = self._live_loop or getattr(session, "_loop", None)
        deadline = time.perf_counter() + timeout
        while (
            loop is None
            and self._thread_running(self.live_thread)
            and time.perf_counter() < deadline
        ):
            time.sleep(0.05)
            loop = self._live_loop or getattr(session, "_loop", None)

        if loop is not None and loop.is_running() and not loop.is_closed():
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is loop:
                current_loop.create_task(session.stop())
                self.logger.info("Live validation stop scheduled.")
                return True

            future = asyncio.run_coroutine_threadsafe(session.stop(), loop)
            try:
                future.result(timeout=max(0.1, deadline - time.perf_counter()))
            except Exception as e:
                self.logger.error("Error stopping live validation: %s", e)
                return False
        else:
            self.logger.warning("Live validation loop is not available.")
            return False

        self.logger.info("Live validation stopped.")
        return True

    def get_live_session(self, session_id: str = None) -> Optional[LiveValidator]:
        """Return the current live session, optionally matching an id."""
        with self._lock:
            session = self.live_session
        if session is None:
            return None
        if session_id is not None and session.session_id != session_id:
            return None
        return session

    def get_status(self) -> Dict[str, Any]:
        """Return status info for historical and live validation."""
        with self._lock:
            historical_thread = self.historical_thread
            live_session = self.live_session
            live_thread = self.live_thread

        status = {
            "batch": "running" if self._thread_running(historical_thread) else "idle",
            "live": "idle",
        }

        if self.last_historical_report is not None:
            status["batch_last_report"] = self.last_historical_report.report_file

        if live_session is None:
            return status

        live_status = live_session.get_status()
        status["live"] = live_status.status
        status["live_thread"] = (
            "running" if self._thread_running(live_thread) else "idle"
        )
        status["live_session_id"] = live_status.session_id
        status["live_validated"] = live_status.total_validated
        status["live_pending"] = live_status.total_pending
        status["live_predicted"] = live_status.total_predicted
        status["live_discarded"] = live_status.total_discarded
        status["live_median_rmse"] = live_status.median_rmse
        return status

    def generate_trip_validation_chart(
        self,
        trip_id: str,
        observatory,
        predictor=None,
        output_path: str | Path = None,
    ) -> Optional[TripValidationChartResult]:
        """Generate a predicted-vs-actual delay chart for one historical trip."""
        historical = getattr(observatory, "historical", None)
        if historical is None:
            self.logger.warning("Historical ledger not available.")
            return None

        actual_df = historical.query(trip_id=trip_id)
        if actual_df is None or actual_df.empty:
            self.logger.warning("No historical measurements found for trip %s", trip_id)
            return None

        row = actual_df.iloc[0]
        route_id = str(row["route_id"])
        direction_id = int(row["direction_id"])
        scheduled_start = str(row["scheduled_start_time"])[:5]
        trip_date_dt = datetime.fromtimestamp(float(row["measurement_time"]))
        trip_date = trip_date_dt.strftime("%d-%m-%Y")

        predicted_df = pd.DataFrame()
        prediction_ledger = getattr(predictor, "predicted", None)
        if prediction_ledger is not None:
            predicted_query = prediction_ledger.query_trip(
                route_id=route_id,
                direction_id=direction_id,
                trip_date=trip_date,
                scheduled_start=scheduled_start,
            )
            if predicted_query is not None:
                predicted_df = predicted_query

        topology = observatory.get_topology()
        stops_map = topology.build_stops_map(route_id, direction_id) if topology else {}
        if output_path is None:
            safe_id = trip_id.replace("/", "_").replace("\\", "_")[:50]
            output_path = RESULTS_DIR / f"trip_validation_chart_{safe_id}.png"

        result = _render_trip_validation_chart(
            predicted_df=predicted_df if not predicted_df.empty else None,
            actual_df=actual_df,
            stops_map=stops_map,
            route_id=route_id,
            trip_id=trip_id,
            output_path=Path(output_path),
        )
        if result:
            self.logger.info("Trip validation chart written to %s", result.output_path)
        return result

    def render_training_loss(
        self,
        log_path: str | Path,
        output_path: str | Path = None,
    ) -> Optional[LossPlotResult]:
        """Render train/validation loss from one training log."""
        return EvaluationValidator().render_training_loss(log_path, output_path)

    def _run_historical_thread(
        self,
        date_str: str,
        predictor,
        observatory,
        bus_type_predictor,
    ):
        """Thread body for historical validation."""
        try:
            report = self.run_historical(
                date_str=date_str,
                predictor=predictor,
                observatory=observatory,
                bus_type_predictor=bus_type_predictor,
            )
            self.logger.info(
                "Historical validation complete: %d trips, median RMSE=%.2fs",
                report.total_trips_validated,
                report.median_rmse,
            )
        except Exception as e:
            self.logger.error("Historical validation failed: %s", e)

    def _run_live_thread(self, session: LiveValidator):
        """Thread body for live validation."""
        loop = asyncio.new_event_loop()
        with self._lock:
            self._live_loop = loop

        asyncio.set_event_loop(loop)
        try:
            started = loop.run_until_complete(session.start())
            if started:
                loop.run_until_complete(self._wait_for_live_session(session))
            else:
                self.logger.warning(
                    "Live validation session failed to start for %s",
                    session.target_date,
                )
        except Exception as e:
            self.logger.error("Live validation error: %s", e)
        finally:
            with self._lock:
                if self._live_loop is loop:
                    self._live_loop = None
            loop.close()

    async def _wait_for_live_session(self, session: LiveValidator):
        """Wait until a live session reaches a terminal state."""
        while session.status in ("predicting", "monitoring"):
            await asyncio.sleep(1)

    def _build_live_session(
        self,
        date_str: str,
        predictor,
        observatory,
        bus_type_predictor=None,
        session_id: str = None,
    ) -> LiveValidator:
        """Create a live validator session."""
        return LiveValidator(
            session_id=session_id or str(uuid.uuid4()),
            target_date=date_str,
            predictor=predictor,
            observatory=observatory,
            bus_type_predictor=bus_type_predictor,
            persistence_gateway=self.persistence,
        )

    def _live_running_locked(self) -> bool:
        """Return whether the current live session is active."""
        if self._thread_running(self.live_thread):
            return True
        return bool(
            self.live_session
            and self.live_session.status in ("predicting", "monitoring")
        )

    def _thread_running(self, thread: Optional[threading.Thread]) -> bool:
        """Return whether a thread exists and is alive."""
        return thread is not None and thread.is_alive()


class EvaluationValidator(Validator):
    """Validator for training/evaluation artifacts such as loss logs."""

    @property
    def output_prefix(self) -> str:
        """Prefix used for evaluation output files."""
        return "evaluation"

    def __init__(self):
        super().__init__(
            predictor=None,
            observatory=None,
            logger_name="validator.evaluation",
        )

    def render_training_loss(
        self,
        log_path: str | Path,
        output_path: str | Path = None,
    ) -> Optional[LossPlotResult]:
        """Parse a training log and render train/validation loss curves."""
        log_path = Path(log_path)
        if not log_path.exists():
            self.logger.warning("Training log not found: %s", log_path)
            return None

        epochs, train_loss, validation_loss = self._parse_training_loss(log_path)
        if not epochs:
            self.logger.warning("No compatible loss rows found in %s", log_path)
            return None

        if output_path is None:
            output_path = RESULTS_DIR / f"loss_{log_path.stem}.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt = _load_matplotlib()
        if plt is None:
            self.logger.warning("matplotlib not available; loss plot skipped.")
            return None

        best_idx = validation_loss.index(min(validation_loss))
        best_epoch = epochs[best_idx]
        best_validation_loss = validation_loss[best_idx]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(
            epochs,
            train_loss,
            label="Training Loss",
            color="#2c3e50",
            linewidth=2.5,
            alpha=0.85,
        )
        ax.plot(
            epochs,
            validation_loss,
            label="Validation Loss",
            color="#e74c3c",
            linewidth=2.5,
            alpha=0.9,
        )
        ax.axvline(
            x=best_epoch,
            color="#7f8c8d",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.scatter(best_epoch, best_validation_loss, color="#c0392b", s=80, zorder=5)

        offset_x = 2 if best_epoch < (max(epochs) - 10) else -15
        loss_span = max(validation_loss) - min(validation_loss)
        ax.annotate(
            f"Early Stopping\nEpoch {best_epoch}",
            xy=(best_epoch, best_validation_loss),
            xytext=(best_epoch + offset_x, best_validation_loss + loss_span * 0.05),
            arrowprops=dict(facecolor="#34495e", arrowstyle="->", alpha=0.8),
            fontsize=11,
            weight="bold",
            color="#2c3e50",
        )

        title = f"Learning Dynamics - {log_path.stem.replace('_', ' ').upper()}"
        ax.set_title(title, fontsize=14, weight="bold", pad=20)
        ax.set_xlabel("Training Epochs", fontsize=12, weight="bold", labelpad=10)
        ax.set_ylabel("Combined Loss (MSE / CE)", fontsize=12, weight="bold", labelpad=10)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper right", frameon=True, fontsize=11, shadow=True)
        plt.tight_layout()
        plt.savefig(output_path, format="png", bbox_inches="tight")
        plt.close(fig)

        result = LossPlotResult(
            log_file=str(log_path),
            output_path=str(output_path),
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
        )
        self.logger.info("Loss plot written to %s", output_path)
        return result

    def _parse_training_loss(self, log_path: Path) -> Tuple[List[int], List[float], List[float]]:
        """Extract epoch, train loss and validation loss from a training log."""
        epochs, train_loss, validation_loss = [], [], []
        pattern = re.compile(
            r"Epoch \[(\d+)/\d+\].*?Tot Train: ([\d.]+) \(Val: ([\d.]+)\)"
        )

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if not match:
                    continue
                epochs.append(int(match.group(1)))
                train_loss.append(float(match.group(2)))
                validation_loss.append(float(match.group(3)))
        return epochs, train_loss, validation_loss


def _render_trip_validation_chart(
    predicted_df: Optional[pd.DataFrame],
    actual_df: pd.DataFrame,
    stops_map: Dict[int, Dict],
    route_id: str,
    trip_id: str,
    output_path: Path,
) -> Optional[TripValidationChartResult]:
    """Render a predicted-vs-actual delay chart."""
    has_predicted = predicted_df is not None and not predicted_df.empty
    has_actual = actual_df is not None and not actual_df.empty
    if not has_predicted and not has_actual:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    all_sequences = set()
    if has_predicted:
        all_sequences.update(predicted_df["stop_sequence"].values)
    if has_actual:
        all_sequences.update(actual_df["stop_sequence"].values)
    sequence_range = sorted(all_sequences)

    ax.axhline(
        y=0,
        color="#7f8c8d",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Scheduled (on-time)",
    )

    if has_predicted:
        predicted_sorted = predicted_df.sort_values("stop_sequence")
        ax.plot(
            predicted_sorted["stop_sequence"],
            predicted_sorted["predicted_delay_sec"],
            label="Predicted delay",
            color="#2c3e50",
            linewidth=2.5,
            alpha=0.85,
        )

    rmse = None
    if has_actual:
        actual_grouped = (
            actual_df.groupby("stop_sequence")["schedule_adherence"]
            .median()
            .reset_index()
            .sort_values("stop_sequence")
        )
        ax.scatter(
            actual_grouped["stop_sequence"],
            actual_grouped["schedule_adherence"],
            label="Actual delay",
            color="#e74c3c",
            s=40,
            zorder=5,
            alpha=0.9,
        )

        if has_predicted:
            merged = actual_grouped.merge(
                predicted_df[["stop_sequence", "predicted_delay_sec"]],
                on="stop_sequence",
                how="inner",
            )
            if not merged.empty:
                errors = (
                    merged["predicted_delay_sec"] - merged["schedule_adherence"]
                ) ** 2
                rmse = float(np.sqrt(errors.mean()))

    if stops_map and sequence_range:
        tick_sequences = sequence_range[:: max(1, len(sequence_range) // 15)]
        tick_labels = [
            stops_map.get(seq, {}).get("stop_name", str(seq))[:18]
            for seq in tick_sequences
        ]
        ax.set_xticks(tick_sequences)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(_format_delay_tick))
    short_trip = trip_id[:30] + "..." if len(trip_id) > 30 else trip_id
    rmse_label = f" | RMSE: {rmse:.1f}s" if rmse is not None else ""
    ax.set_title(
        f"Trip Validation - Route {route_id} | {short_trip}{rmse_label}",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    ax.set_xlabel("Stop Sequence", fontsize=11, weight="bold", labelpad=8)
    ax.set_ylabel("Delay (mm:ss)", fontsize=11, weight="bold", labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best", frameon=True, fontsize=10, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)

    return TripValidationChartResult(
        trip_id=trip_id,
        route_id=route_id,
        output_path=str(output_path),
        has_predicted=has_predicted,
        has_actual=has_actual,
        rmse=rmse,
    )


def _format_delay_tick(value, _):
    """Format seconds as a signed mm:ss delay value."""
    sign = "-" if value < 0 else "+"
    minutes, seconds = divmod(abs(int(value)), 60)
    return f"{sign}{minutes}:{seconds:02d}"


def _load_matplotlib():
    """Load matplotlib in headless mode when available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _write_metric_report(
    report_path: Path,
    title: str,
    summary: List[Tuple[str, Any]],
    metrics: Dict[str, float],
    confusion_matrix: List[List[int]],
    session_id: str = "",
    started_at: str = "",
    status: str = "",
):
    """Write a text report shared by historical and live validators."""
    class_labels = [
        "EMPTY",
        "MANY_SEATS",
        "FEW_SEATS",
        "STANDING",
        "CRUSHED",
        "FULL",
        "NOT_ACCEPT",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n\n")

        if session_id or started_at or status:
            f.write("SESSION INFO\n")
            f.write("-" * 40 + "\n")
            if session_id:
                f.write(f"Session ID: {session_id}\n")
            if started_at:
                f.write(f"Started: {started_at}\n")
            if status:
                f.write(f"Status: {status}\n")
            f.write("\n")

        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        for label, value in summary:
            f.write(f"{label:24}: {value}\n")
        f.write("\n")

        f.write("DELAY PREDICTION (seconds)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Median MSE:     {metrics['median_mse']:.2f}\n")
        f.write(f"Median RMSE:    {metrics['median_rmse']:.2f}\n")
        f.write(f"Min RMSE:       {metrics['min_rmse']:.2f}\n")
        f.write(f"Max RMSE:       {metrics['max_rmse']:.2f}\n\n")

        f.write("OCCUPANCY CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write("         " + "  ".join(f"{i:>5}" for i in range(7)) + "\n")
        f.write("         " + "  ".join(f"{l[:5]:>5}" for l in class_labels) + "\n")
        f.write("Actual:\n")
        for i, row in enumerate(confusion_matrix):
            label = class_labels[i][:7] if i < len(class_labels) else str(i)
            f.write(f"  {label:7} " + "  ".join(f"{v:>5}" for v in row) + "\n")

        f.write("\nPer-class metrics:\n")
        num_classes = 7
        for i in range(num_classes):
            tp = confusion_matrix[i][i]
            fp = sum(confusion_matrix[j][i] for j in range(num_classes)) - tp
            fn = sum(confusion_matrix[i][j] for j in range(num_classes)) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            label = class_labels[i] if i < len(class_labels) else str(i)
            f.write(f"  {label:12}: P={precision:.2f} R={recall:.2f} F1={f1:.2f}\n")

        f.write("\n" + "=" * 60 + "\n")


__all__ = [
    "EvaluationValidator",
    "HistoricalValidator",
    "LiveValidationStatus",
    "LiveValidator",
    "LossPlotResult",
    "TripValidationResult",
    "TripValidationChartResult",
    "ValidationController",
    "ValidationReport",
    "Validator",
]
