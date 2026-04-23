"""
Bus Type Predictor Service - Predicts bus type from route and schedule info.

Uses a LightGBM model trained on historical data to predict bus_type
based on: route_id, time_sin, time_cos, day_type.
"""

import math
import joblib
from datetime import date
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "bus_type_predictor.pkl"


class BusTypePredictor:
    """Wraps the trained LightGBM bus-type classifier for inference at predict time."""

    def __init__(self, model_path: Optional[str] = None):
        """Load the LightGBM model from ``model_path`` (or the default ``bus_type_predictor.pkl``)."""
        path = Path(model_path) if model_path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Bus type model not found: {path}")
        self.model = joblib.load(path)

    def _get_day_type(self, d: date) -> int:
        """0=weekday, 1=saturday, 2=sunday/holiday"""
        weekday = d.weekday()
        if weekday == 5:
            return 1
        elif weekday == 6:
            return 2
        return 0

    def _compute_time_encoding(self, start_time: str) -> tuple:
        """
        Compute sin/cos encoding from start_time string.

        Args:
            start_time: Time in "HH:MM" or "HH:MM:SS" format

        Returns:
            (time_sin, time_cos) tuple
        """
        parts = start_time.split(":")
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        seconds = int(parts[2]) if len(parts) > 2 else 0

        time_seconds = hours * 3600 + minutes * 60 + seconds
        time_sin = math.sin(2 * math.pi * time_seconds / 86400)
        time_cos = math.cos(2 * math.pi * time_seconds / 86400)

        return time_sin, time_cos

    def predict(self, route_id: str, start_time: str, trip_date: date) -> int:
        """
        Predict bus type for a trip.

        Args:
            route_id: Route identifier (e.g., "64", "780")
            start_time: Scheduled start time in "HH:MM" format
            trip_date: Date of the trip

        Returns:
            Predicted bus_type (integer)
        """
        time_sin, time_cos = self._compute_time_encoding(start_time)
        day_type = self._get_day_type(trip_date)

        X = pd.DataFrame(
            [
                {
                    "route_id": route_id,
                    "time_sin": time_sin,
                    "time_cos": time_cos,
                    "day_type": day_type,
                }
            ]
        )

        X["route_id"] = X["route_id"].astype("category")
        X["day_type"] = X["day_type"].astype("category")

        prediction = self.model.predict(X)[0]
        return int(prediction)

    def predict_batch(self, trips: List[Dict]) -> List[int]:
        """
        Predict bus types for multiple trips at once.

        Args:
            trips: List of dicts with keys: route_id, start_time, trip_date

        Returns:
            List of predicted bus_types
        """
        if not trips:
            return []

        rows = []
        for trip in trips:
            time_sin, time_cos = self._compute_time_encoding(trip["start_time"])
            day_type = self._get_day_type(trip["trip_date"])

            rows.append(
                {
                    "route_id": trip["route_id"],
                    "time_sin": time_sin,
                    "time_cos": time_cos,
                    "day_type": day_type,
                }
            )

        X = pd.DataFrame(rows)
        X["route_id"] = X["route_id"].astype("category")
        X["day_type"] = X["day_type"].astype("category")

        predictions = self.model.predict(X)
        return [int(p) for p in predictions]
