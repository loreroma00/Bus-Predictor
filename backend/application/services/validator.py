"""
Validator Service - Compare batch predictions against ground truth data.

This module validates ML model predictions by:
1. Loading scheduled trips from GTFS ledger for a specific date
2. Loading ground truth measurements from diaries.parquet
3. Running batch predictions for trips with ground truth
4. Matching predictions to ground truth by shape_dist_travelled
5. Computing MSE, RMSE, and confusion matrices
6. Writing detailed logs and summary reports
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DIARIES_PATH = PROJECT_ROOT / "diaries" / "diaries.parquet"


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
    error: Optional[str] = None


@dataclass
class ValidationReport:
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


class Validator:
    def __init__(self, predictor, observatory):
        self.predictor = predictor
        self.observatory = observatory
        self.logger = logging.getLogger("validator")

        self._bus_type_predictor = None

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def bus_type_predictor(self):
        """Lazy-load the bus type predictor on first use."""
        if self._bus_type_predictor is None:
            from application.services.bus_type_predictor import BusTypePredictor

            try:
                self._bus_type_predictor = BusTypePredictor()
            except FileNotFoundError as e:
                self.logger.warning(f"Bus type predictor not available: {e}")
        return self._bus_type_predictor

    def validate_date(self, date_str: str) -> ValidationReport:
        """
        Main entry point: validate all predictions for a given date.

        Args:
            date_str: Date in DD-MM-YYYY format

        Returns:
            ValidationReport with all metrics and results
        """
        self.logger.info(f"Starting validation for {date_str}")

        target_date = datetime.strptime(date_str, "%d-%m-%Y")
        date_yyyymmdd = target_date.strftime("%Y%m%d")

        topology = self.observatory.get_topology()

        scheduled_trips = self._get_scheduled_trips(topology, date_yyyymmdd)
        self.logger.info(
            f"Found {len(scheduled_trips)} scheduled trips for {date_yyyymmdd}"
        )

        ground_truth = self._load_ground_truth(date_yyyymmdd)
        self.logger.info(f"Loaded {len(ground_truth)} measurements from diaries")

        trips_with_gt = self._match_scheduled_to_ground_truth(
            scheduled_trips, ground_truth
        )
        self.logger.info(f"Matched {len(trips_with_gt)} trips with ground truth")

        weather_code = self._get_weather_code(target_date)

        prediction_results = self._run_predictions(trips_with_gt, weather_code)

        validation_results = self._validate_predictions(
            prediction_results, ground_truth, topology
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

    def _get_scheduled_trips(self, topology, date_yyyymmdd: str) -> List[Dict]:
        """Extract all trips scheduled for the given date."""
        scheduled = []

        for trip_id, trip in list(topology.trips.items()):
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
                            "start_date": self._yyyymmdd_to_ddmmyyyy(date_yyyymmdd),
                        }
                    )

        return scheduled

    def _yyyymmdd_to_ddmmyyyy(self, date_str: str) -> str:
        """Convert YYYYMMDD to DD-MM-YYYY."""
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.strftime("%d-%m-%Y")

    def _load_ground_truth(self, date_yyyymmdd: str) -> pd.DataFrame:
        """Load diary measurements for the given date."""
        if not DIARIES_PATH.exists():
            self.logger.warning(f"Diaries file not found: {DIARIES_PATH}")
            return pd.DataFrame()

        df = pd.read_parquet(DIARIES_PATH)

        target_date = datetime.strptime(date_yyyymmdd, "%Y%m%d")
        start_ts = target_date.timestamp()
        end_ts = (target_date + timedelta(days=1)).timestamp()

        filtered = df[
            (df["measurement_time"] >= start_ts) & (df["measurement_time"] < end_ts)
        ]

        return filtered

    def _match_scheduled_to_ground_truth(
        self, scheduled_trips: List[Dict], ground_truth: pd.DataFrame
    ) -> List[Dict]:
        """Find scheduled trips that have ground truth data."""
        if ground_truth.empty:
            return []

        gt_trip_ids = set(ground_truth["trip_id"].unique())

        matched = []
        for trip in scheduled_trips:
            if trip["trip_id"] in gt_trip_ids:
                matched.append(trip)

        return matched

    def _get_weather_code(self, target_date: datetime) -> int:
        """Get weather code from hexagon forecast. Defaults to 0 if unavailable."""
        try:
            city = self.observatory.get_city("Rome")
            if city is None:
                return 0
            for hexagon in city.hexagons.values():
                if hexagon.weather is not None:
                    hour_bucket = target_date.hour
                    weather = hexagon.get_weather_for_hour_bucket(hour_bucket)
                    if weather is None:
                        weather = hexagon.get_weather()
                    return int(getattr(weather, "weather_code", 0) or 0)
            return 0
        except Exception as e:
            self.logger.warning(f"Could not fetch weather from hexagon: {e}")
            return 0

    def _compute_bus_type(self, trip_data: Dict) -> int:
        """Predict bus type using the ML model."""
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

    def _run_predictions(self, trips: List[Dict], weather_code: int) -> Dict[str, Any]:
        """Run batch predictions for all trips."""
        if not trips:
            return {"successful": [], "failed": []}

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

        try:
            forecasts = self.predictor.get_batch_forecast(prediction_requests)

            results = {"successful": [], "failed": []}

            for i, forecast in enumerate(forecasts):
                results["successful"].append(
                    {
                        "trip_id": trips[i]["trip_id"],
                        "forecast": forecast,
                        "request": prediction_requests[i],
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return {
                "successful": [],
                "failed": [{"trip_id": t["trip_id"], "error": str(e)} for t in trips],
            }

    def _validate_predictions(
        self,
        prediction_results: Dict[str, Any],
        ground_truth: pd.DataFrame,
        topology,
    ) -> List[TripValidationResult]:
        """Validate predictions against ground truth."""
        results = []

        for pred in prediction_results["successful"]:
            trip_id = pred["trip_id"]
            forecast = pred["forecast"]

            trip_gt = ground_truth[ground_truth["trip_id"] == trip_id]

            if trip_gt.empty:
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

            stops_map = topology.build_stops_map(
                forecast.route_id, forecast.direction_id
            )

            delay_errors = []
            occupancy_matches = []

            for _, gt_row in trip_gt.iterrows():
                matched_stop = self._match_measurement_to_stop(
                    gt_row, forecast.stops, stops_map
                )

                if matched_stop is None:
                    continue

                predicted_delay = matched_stop.cumulative_delay_sec
                actual_delay = gt_row.get("schedule_adherence", 0.0)

                if pd.notna(actual_delay):
                    delay_errors.append((predicted_delay - actual_delay) ** 2)

                predicted_crowd = matched_stop.crowd_level
                actual_occupancy = gt_row.get("occupancy", -1)

                if pd.notna(actual_occupancy) and 0 <= int(actual_occupancy) <= 6:
                    occupancy_matches.append((predicted_crowd, int(actual_occupancy)))

            if delay_errors:
                mse = sum(delay_errors) / len(delay_errors)
                rmse = mse**0.5
            else:
                mse = 0.0
                rmse = 0.0

            results.append(
                TripValidationResult(
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

    # _build_stops_map removed — now uses TopologyLedger.build_stops_map()

    def _match_measurement_to_stop(
        self, gt_row: pd.Series, predicted_stops: List, stops_map: Dict[int, Dict]
    ) -> Optional[Any]:
        """
        Match a ground truth measurement to the closest predicted stop.

        Strategy:
        1. Try to match by next_stop (stop_id) to stop_sequence
        2. Fallback to matching by cumulative distance
        """
        gt_next_stop = gt_row.get("next_stop")
        gt_next_stop_dist = gt_row.get("next_stop_distance", 0)

        for seq, stop_info in stops_map.items():
            if stop_info["stop_id"] == str(gt_next_stop):
                for pred_stop in predicted_stops:
                    if pred_stop.stop_sequence == seq:
                        return pred_stop

        if gt_next_stop_dist is not None and pd.notna(gt_next_stop_dist):
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

    def _build_report(
        self,
        date_str: str,
        scheduled_trips: List[Dict],
        trips_with_gt: List[Dict],
        validation_results: List[TripValidationResult],
        ground_truth: pd.DataFrame,
    ) -> ValidationReport:
        """Build the final validation report."""
        valid_results = [
            r for r in validation_results if r.error is None and r.n_measurements > 0
        ]

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

        confusion_matrix = self._build_confusion_matrix(valid_results)

        date_yyyymmdd = datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y%m%d")

        return ValidationReport(
            date=date_str,
            total_scheduled_trips=len(scheduled_trips),
            total_trips_with_ground_truth=len(trips_with_gt),
            total_trips_predicted=len(
                [r for r in validation_results if r.error is None]
            ),
            total_trips_validated=len(valid_results),
            total_measurements=sum(r.n_measurements for r in valid_results),
            median_mse=median_mse,
            median_rmse=median_rmse,
            min_mse=min_mse,
            max_mse=max_mse,
            min_rmse=min_rmse,
            max_rmse=max_rmse,
            occupancy_confusion_matrix=confusion_matrix,
            trips=validation_results,
            log_file=f"validation_{date_yyyymmdd}.log",
            report_file=f"validation_{date_yyyymmdd}_report.txt",
        )

    def _build_confusion_matrix(
        self, results: List[TripValidationResult]
    ) -> List[List[int]]:
        """Build confusion matrix for occupancy predictions."""
        num_classes = 7
        cm = [[0] * num_classes for _ in range(num_classes)]

        for result in results:
            for pred, actual in result.occupancy_matches:
                if 0 <= pred < num_classes and 0 <= actual < num_classes:
                    cm[actual][pred] += 1

        return cm

    def _write_log(self, report: ValidationReport, results: List[TripValidationResult]):
        """Write detailed per-trip log file."""
        log_path = RESULTS_DIR / report.log_file

        with open(log_path, "w") as f:
            f.write(f"=== Validation for {report.date} ===\n\n")
            f.write(f"Total scheduled trips: {report.total_scheduled_trips}\n")
            f.write(
                f"Trips with ground truth: {report.total_trips_with_ground_truth}\n"
            )
            f.write(f"Trips predicted successfully: {report.total_trips_predicted}\n")
            f.write(f"Trips validated: {report.total_trips_validated}\n\n")

            for result in results:
                f.write(f"--- Trip {result.trip_id} ---\n")
                f.write(
                    f"Route: {result.route_id}, Direction: {result.direction_id}, Start: {result.scheduled_start}\n"
                )

                if result.error:
                    f.write(f"ERROR: {result.error}\n\n")
                else:
                    f.write(f"Measurements: {result.n_measurements}\n")
                    f.write(
                        f"Delay MSE: {result.mse:.2f} sec^2, RMSE: {result.rmse:.2f} sec\n"
                    )

                    if result.occupancy_matches:
                        correct = sum(1 for p, a in result.occupancy_matches if p == a)
                        accuracy = correct / len(result.occupancy_matches)
                        f.write(f"Occupancy accuracy: {accuracy:.2%}\n")

                    f.write("\n")

        self.logger.info(f"Log written to {log_path}")

    def _write_report(self, report: ValidationReport):
        """Write summary report file."""
        report_path = RESULTS_DIR / report.report_file

        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"VALIDATION REPORT: {report.date}\n")
            f.write("=" * 60 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Scheduled trips:        {report.total_scheduled_trips}\n")
            f.write(
                f"Trips with ground truth: {report.total_trips_with_ground_truth}\n"
            )
            f.write(f"Validated trips:        {report.total_trips_validated}\n")
            f.write(f"Total measurements:     {report.total_measurements}\n\n")

            f.write("DELAY PREDICTION (seconds)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Median MSE:     {report.median_mse:.2f}\n")
            f.write(f"Median RMSE:    {report.median_rmse:.2f}\n")
            f.write(f"Min RMSE:       {report.min_rmse:.2f}\n")
            f.write(f"Max RMSE:       {report.max_rmse:.2f}\n\n")

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

            cm = report.occupancy_confusion_matrix
            f.write("         " + "  ".join(f"{i:>5}" for i in range(7)) + "\n")
            f.write("         " + "  ".join(f"{l[:5]:>5}" for l in class_labels) + "\n")
            f.write("Actual:\n")

            for i, row in enumerate(cm):
                label = class_labels[i][:7] if i < len(class_labels) else str(i)
                f.write(f"  {label:7} " + "  ".join(f"{v:>5}" for v in row) + "\n")

            f.write("\nPer-class metrics:\n")
            num_classes = 7
            for i in range(num_classes):
                tp = cm[i][i]
                fp = sum(cm[j][i] for j in range(num_classes)) - tp
                fn = sum(cm[i][j] for j in range(num_classes)) - tp

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

        self.logger.info(f"Report written to {report_path}")
