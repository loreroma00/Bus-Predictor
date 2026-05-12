"""Inference wrapper: loads the twin BusLSTM/OccupancyLSTM checkpoints and maps raw trip inputs to per-stop forecasts."""

import json
import logging
import math
from pathlib import Path
from datetime import date
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import joblib

try:
    from .model_loader import LoadedModel, ModelLoader
except ImportError:  # pragma: no cover - keeps script-style imports working
    from model_loader import LoadedModel, ModelLoader
try:
    from application.domain.ledgers import PredictedLedger, StopPredictionRecord
except ImportError:  # pragma: no cover
    from ..domain.ledgers import PredictedLedger, StopPredictionRecord

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

PARQUET_DIR = PROJECT_ROOT / "parquets"
ROUTE_ENCODING_PATH = PARQUET_DIR / "route_encoding.json"
ROUTE_ENCODER_PKL_PATH = PARQUET_DIR / "route_encoder.pkl"
H3_ENCODING_PATH = PARQUET_DIR / "h3_encoding.json"
STATIC_MAP_PATH = PARQUET_DIR / "stop_route_map.parquet"
logger = logging.getLogger(__name__)


@dataclass
class StopPrediction:
    """Per-stop forecast: expected arrival, cumulative delay, and crowd class."""
    stop_sequence: int
    distance_m: float
    cumulative_delay_sec: float
    delay_formatted: str
    expected_arrival: str
    crowd_level: int

@dataclass
class TripForecast:
    """One full trip forecast: route/direction identity plus a list of StopPredictions."""
    route_id: str
    direction_id: int
    trip_date: str
    scheduled_start: str
    stops: list[StopPrediction]


class Predictor:
    """Application-level prediction service backed by a loaded model and ledgers."""

    def __init__(
        self,
        loaded_model: LoadedModel | None = None,
        config_path: str | None = None,
        time_weights_path: str | None = None,
        crowd_weights_path: str | None = None,
        observatory: Any = None,
        prediction_ledger: Any = None,
        route_encoding_path: Optional[str] = None,
        h3_encoding_path: Optional[str] = None,
        static_map_path: Optional[str] = None,
    ):
        """Load encoders/static stop map and bind a low-level LoadedModel."""
        if loaded_model is None:
            if not (config_path and time_weights_path and crowd_weights_path):
                raise ValueError(
                    "Predictor requires a LoadedModel or config/time/crowd paths"
                )
            loaded_model = ModelLoader(Path(time_weights_path).parent).load_from_paths(
                config_path,
                time_weights_path,
                crowd_weights_path,
            )

        self.loaded_model = loaded_model
        self.config = loaded_model.config
        self.max_stops = loaded_model.max_stops
        self.observatory = observatory
        self.predicted = prediction_ledger or PredictedLedger()

        route_path = Path(route_encoding_path) if route_encoding_path else ROUTE_ENCODING_PATH
        route_encoder_pkl_path = ROUTE_ENCODER_PKL_PATH
        h3_path = Path(h3_encoding_path) if h3_encoding_path else H3_ENCODING_PATH
        static_path = Path(static_map_path) if static_map_path else STATIC_MAP_PATH

        # Load route encoder
        self.route_encoder = {}
        if route_encoder_pkl_path.exists():
            route_obj = joblib.load(route_encoder_pkl_path)
            route_le = route_obj.get("route") if isinstance(route_obj, dict) else route_obj
            if route_le is not None and hasattr(route_le, "classes_"):
                self.route_encoder = {
                    str(label): int(idx) for idx, label in enumerate(route_le.classes_)
                }
                print("Loaded route encoder from route_encoder.pkl")

        if not self.route_encoder:
            with open(route_path, "r") as f:
                self.route_encoder = {str(k): int(v) for k, v in json.load(f).items()}
            print("Loaded route encoder from route_encoding.json")

        # Load H3 encoder
        with open(h3_path, "r") as f:
            self.h3_encoder = json.load(f)

        # Load static stop map
        self.static_map = pd.read_parquet(static_path)
        self.static_map["route_id_norm"] = self.static_map["route_id"].astype(str)
        self.static_map["direction_id_norm"] = self.static_map["direction_id"].astype(str)

        print(f"MAX_STOPS = {self.max_stops}")

    def has_trip_template(self, route_id: str, direction_id: Any) -> bool:
        """Return True if the static stop map has a trip template for this route/direction."""
        trip_static = self.static_map[
            (self.static_map["route_id_norm"] == str(route_id)) &
            (self.static_map["direction_id_norm"] == str(direction_id))
        ]
        return not trip_static.empty

    def _get_day_type(self, d: date) -> int:
        """Encode calendar day as 0 (weekday), 1 (Saturday), or 2 (Sunday)."""
        weekday = d.weekday()
        if weekday == 5:
            return 1
        elif weekday == 6:
            return 2
        return 0

    def _format_delay(self, delay_seconds: float) -> str:
        """Render a signed delay (seconds) as a human ``±Xm Ys`` string."""
        total_seconds = int(round(delay_seconds))
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{sign}{minutes}m {seconds}s"

    def _format_time(self, time_seconds: float) -> str:
        """Render a seconds-of-day value as ``HH:MM:SS`` (modulo 24h)."""
        total_seconds = int(round(time_seconds)) % 86400
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _build_trip_tensors(self, trip_stops: pd.DataFrame, route_id: str, direction_id: int,
                            time_seconds: int, day_type: int, weather_code: int, bus_type: int):
        """Build padded and scaled input tensors for a single trip from its stop map data.

        Returns x1_cat, x1_dense, x2_cat, x2_dense, t_grid, lengths, num_stops.
        All x2 arrays are padded to max_stops and pre-scaled.
        """
        ms = self.max_stops
        num_stops = len(trip_stops)
        trip_stops = trip_stops.sort_values("stop_idx")

        # ==============================
        # x1: trip-level categorical
        # ==============================
        route_enc = self.route_encoder.get(str(route_id), 0)
        x1_cat = np.array([
            route_enc,
            int(direction_id),
            int(day_type),
            min(int(int(weather_code) / 33), 2),
            int(float(bus_type) / 9.0),
        ], dtype=np.int64)

        # ==============================
        # x1: trip-level dense
        # ==============================
        time_sin = math.sin(2 * math.pi * time_seconds / 86400)
        time_cos = math.cos(2 * math.pi * time_seconds / 86400)
        # 13 deposit flags (default 0) + door_number (scaled) + time_sin + time_cos
        x1_dense = np.zeros(16, dtype=np.float32)
        x1_dense[13] = 1.0  # door_number: clip(3, 1, 3) / 3.0 = 1.0
        x1_dense[14] = time_sin
        x1_dense[15] = time_cos

        # ==============================
        # x2: stop-level categorical (h3_index_encoded, stop_sequence)
        # ==============================
        h3_raw = trip_stops["h3_index"].values
        h3_encoded = np.array(
            [self.h3_encoder.get(str(h), 0) if h else 0 for h in h3_raw],
            dtype=np.int64
        )
        stop_seq = trip_stops["stop_sequence"].fillna(0).astype(int).values

        x2_cat = np.zeros((ms, 2), dtype=np.int64)
        x2_cat[:num_stops, 0] = h3_encoded
        x2_cat[:num_stops, 1] = stop_seq

        # ==============================
        # x2: stop-level dense (pre-scaled)
        # Columns: shape_dist, dist_to_next, stop_idx, is_genuine, speed_ratio, traffic_speed
        # ==============================
        shape_dist = trip_stops["shape_dist_at_stop"].values.astype(np.float64)
        dist_to_next = trip_stops["distance_to_next_stop"].values.astype(np.float64)
        stop_idx_vals = trip_stops["stop_idx"].values.astype(np.float64)

        x2_dense = np.zeros((ms, 6), dtype=np.float32)
        x2_dense[:num_stops, 0] = np.clip(shape_dist, 0, 15000) / 15000.0
        x2_dense[:num_stops, 1] = np.clip(dist_to_next, 0, 1000) / 1000.0
        x2_dense[:num_stops, 2] = stop_idx_vals / max(ms - 1, 1)
        x2_dense[:num_stops, 3] = 1.0                                        # is_genuine
        x2_dense[:num_stops, 4] = 1.0                                        # speed_ratio (default)
        x2_dense[:num_stops, 5] = np.clip(25.0, 0, 65.0) / 65.0             # traffic_speed (default, scaled)

        # ==============================
        # t_grid: normalized cumulative stop distances for ODE integration
        # ==============================
        route_len = float(trip_stops["route_len_m"].iloc[0]) if "route_len_m" in trip_stops.columns else float(shape_dist[-1])
        t_grid = np.zeros(ms, dtype=np.float32)
        if route_len > 0:
            t_grid[:num_stops] = (shape_dist / route_len).astype(np.float32)
        else:
            t_grid[:num_stops] = np.linspace(0, 1, num_stops, dtype=np.float32)

        lengths = np.array([num_stops], dtype=np.int64)

        return x1_cat, x1_dense, x2_cat, x2_dense, t_grid, lengths, num_stops

    def get_trip_forecast_raw(self, route_id: str, direction_id: int,
                              time_seconds: int, day_type: int,
                              weather_code: int, bus_type: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Run both models on a single trip and return raw (delays, crowd, trip_stops_df) without formatting."""
        trip_stops = self.static_map[
            (self.static_map["route_id_norm"] == str(route_id)) &
            (self.static_map["direction_id_norm"] == str(direction_id))
        ]
        if trip_stops.empty:
            raise ValueError(f"Route {route_id} direction {direction_id} not found in stop map")

        x1_cat, x1_dense, x2_cat, x2_dense, t_grid, lengths, num_stops = \
            self._build_trip_tensors(trip_stops, route_id, direction_id,
                                    time_seconds, day_type, weather_code, bus_type)

        delays, crowd = self.loaded_model.infer_single(
            x1_cat,
            x1_dense,
            x2_cat,
            x2_dense,
            t_grid,
            lengths,
        )

        # Only return predictions for real stops (not padding)
        delays = delays[:num_stops]
        crowd = crowd[:num_stops]

        trip_stops_sorted = trip_stops.sort_values("stop_idx")
        return delays, crowd, trip_stops_sorted

    def get_trip_forecast(self, route_id: str, direction_id: int,
                          start_date: str, start_time: str,
                          weather_code: int, bus_type: int) -> TripForecast:
        """Produce a fully-formatted TripForecast for one scheduled trip."""
        from datetime import datetime
        trip_date = datetime.strptime(start_date, "%d-%m-%Y").date()
        time_parts = start_time.split(":")
        time_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60
        day_type = self._get_day_type(trip_date)

        delays, crowd, trip_stops = self.get_trip_forecast_raw(
            route_id, direction_id, time_seconds, day_type, weather_code, bus_type
        )

        # Predictions are per-stop — direct mapping (no segment groupby needed)
        stop_predictions = []
        for i, (_, row) in enumerate(trip_stops.iterrows()):
            delay_at_stop = float(delays[i])
            crowd_at_stop = int(crowd[i])
            distance_m = float(row["shape_dist_at_stop"])
            expected_arrival_seconds = time_seconds + delay_at_stop

            stop_predictions.append(
                StopPrediction(
                    stop_sequence=int(row["stop_sequence"]),
                    distance_m=distance_m,
                    cumulative_delay_sec=delay_at_stop,
                    delay_formatted=self._format_delay(delay_at_stop),
                    expected_arrival=self._format_time(expected_arrival_seconds),
                    crowd_level=crowd_at_stop,
                )
            )

        return TripForecast(
            route_id=route_id, direction_id=direction_id, trip_date=start_date,
            scheduled_start=start_time + ":00", stops=stop_predictions,
        )

    def attach_observatory(self, observatory: Any):
        """Bind or replace the observatory used for prediction ledger access."""
        self.observatory = observatory

    def get_or_create_trip_forecast(
        self,
        route_id: str,
        direction_id: int,
        start_date: str,
        start_time: str,
        weather_code: int,
        bus_type: int,
    ) -> TripForecast:
        """Return a cached prediction if present, otherwise infer and record it."""
        cached = self.get_cached_trip_forecast(
            route_id=route_id,
            direction_id=direction_id,
            start_date=start_date,
            start_time=start_time,
        )
        if cached is not None:
            return cached

        forecast = self.get_trip_forecast(
            route_id,
            direction_id,
            start_date,
            start_time,
            weather_code,
            bus_type,
        )
        self.record_forecast(forecast)
        return forecast

    def get_cached_trip_forecast(
        self,
        route_id: str,
        direction_id: int,
        start_date: str,
        start_time: str,
    ) -> TripForecast | None:
        """Load a previously-recorded forecast from the prediction ledger."""
        if self.observatory is None:
            return None

        try:
            cached = self.predicted.query_trip(
                route_id=route_id,
                direction_id=direction_id,
                trip_date=start_date,
                scheduled_start=start_time,
            )
        except Exception as exc:
            logger.debug("Prediction ledger cache lookup failed: %s", exc)
            return None

        if cached.empty:
            return None

        topology = self.observatory.get_topology()
        stops_map = topology.build_stops_map(route_id, direction_id)
        stop_predictions: list[StopPrediction] = []
        for _, row in cached.iterrows():
            seq = int(row["stop_sequence"])
            stop_info = stops_map.get(seq, {})
            delay = float(row["predicted_delay_sec"])
            stop_predictions.append(
                StopPrediction(
                    stop_sequence=seq,
                    distance_m=float(stop_info.get("shape_dist_travelled") or 0.0),
                    cumulative_delay_sec=delay,
                    delay_formatted=self._format_delay(delay),
                    expected_arrival=str(row["predicted_arrival"]),
                    crowd_level=int(row.get("predicted_crowd_level") or 0),
                )
            )

        first = cached.iloc[0]
        trip_date = first["trip_date"]
        if hasattr(trip_date, "strftime"):
            trip_date = trip_date.strftime("%d-%m-%Y")
        else:
            trip_date = str(trip_date)

        return TripForecast(
            route_id=str(first["route_id"]),
            direction_id=int(first["direction_id"]),
            trip_date=trip_date,
            scheduled_start=str(first["scheduled_start"]),
            stops=stop_predictions,
        )

    def get_cached_trip_keys(self, trip_date: str) -> set[tuple[str, int, str]]:
        """Return cached `(route_id, direction_id, HH:MM)` keys for a date."""
        if self.observatory is None:
            return set()
        try:
            cached = self.predicted.query(trip_date=trip_date)
        except Exception:
            return set()
        if cached.empty:
            return set()
        return {
            (
                str(row["route_id"]),
                int(row["direction_id"]),
                str(row["scheduled_start"])[:5],
            )
            for _, row in cached.iterrows()
        }

    def record_forecasts(self, forecasts: list[TripForecast]):
        """Record multiple forecasts into the prediction ledger."""
        for forecast in forecasts:
            self.record_forecast(forecast)

    def record_forecast(self, forecast: TripForecast):
        """Record a forecast into the PredictedLedger."""
        if self.observatory is None:
            return

        try:
            import time as _time
            topology = self.observatory.get_topology()
            stops_map = topology.build_stops_map(
                forecast.route_id,
                forecast.direction_id,
            )
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
            self.predicted.record_predictions(records)
        except Exception as exc:
            logger.warning("Failed to record predictions: %s", exc)
            return

    def get_batch_forecast(
        self,
        trips: List[Dict[str, Any]],
        record: bool = False,
    ) -> List[TripForecast]:
        """Batch prediction for multiple trips.

        Accepts a list of trip dicts with keys: route_id, direction_id,
        start_date (DD-MM-YYYY), start_time (HH:MM), weather_code, bus_type.
        Returns one TripForecast per trip in the same order.
        Raises ValueError for any trip whose route/direction is absent from
        the static stop map (caller is responsible for catching per-chunk).
        """
        from datetime import datetime

        all_x1_cat, all_x1_dense = [], []
        all_x2_cat, all_x2_dense = [], []
        all_t_grid, all_lengths = [], []
        num_stops_list: List[int] = []
        trip_stops_list: List[pd.DataFrame] = []
        meta_list = []  # (route_id, direction_id, start_date, start_time, time_seconds)

        for trip in trips:
            route_id     = trip["route_id"]
            direction_id = int(trip["direction_id"])
            start_date   = trip["start_date"]
            start_time   = trip["start_time"]
            weather_code = int(trip["weather_code"])
            bus_type     = int(trip["bus_type"])

            trip_stops = self.static_map[
                (self.static_map["route_id_norm"] == str(route_id)) &
                (self.static_map["direction_id_norm"] == str(direction_id))
            ]
            if trip_stops.empty:
                raise ValueError(f"Route {route_id}/{direction_id} not in stop map")

            trip_date    = datetime.strptime(start_date, "%d-%m-%Y").date()
            time_parts   = start_time.split(":")
            time_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60
            day_type     = self._get_day_type(trip_date)

            x1_cat, x1_dense, x2_cat, x2_dense, t_grid, lengths, num_stops = \
                self._build_trip_tensors(trip_stops, route_id, direction_id,
                                         time_seconds, day_type, weather_code, bus_type)

            all_x1_cat.append(x1_cat)
            all_x1_dense.append(x1_dense)
            all_x2_cat.append(x2_cat)
            all_x2_dense.append(x2_dense)
            all_t_grid.append(t_grid)
            all_lengths.append(lengths[0])
            num_stops_list.append(num_stops)
            trip_stops_list.append(trip_stops.sort_values("stop_idx"))
            meta_list.append((route_id, direction_id, start_date, start_time, time_seconds))

        # Single batched forward pass through both models
        delays_batch, crowd_batch = self.loaded_model.infer_batch(
            np.stack(all_x1_cat),
            np.stack(all_x1_dense),
            np.stack(all_x2_cat),
            np.stack(all_x2_dense),
            np.stack(all_t_grid),
            np.array(all_lengths, dtype=np.int64),
        )

        # Build one TripForecast per trip, trimming padding to real stops only
        forecasts: List[TripForecast] = []
        for i, (route_id, direction_id, start_date, start_time, time_seconds) in enumerate(meta_list):
            n      = num_stops_list[i]
            delays = delays_batch[i, :n]
            crowd  = crowd_batch[i,  :n]

            stop_predictions = []
            for j, (_, row) in enumerate(trip_stops_list[i].iterrows()):
                delay_at_stop = float(delays[j])
                stop_predictions.append(StopPrediction(
                    stop_sequence=int(row["stop_sequence"]),
                    distance_m=float(row["shape_dist_at_stop"]),
                    cumulative_delay_sec=delay_at_stop,
                    delay_formatted=self._format_delay(delay_at_stop),
                    expected_arrival=self._format_time(time_seconds + delay_at_stop),
                    crowd_level=int(crowd[j]),
                ))

            forecasts.append(TripForecast(
                route_id=route_id,
                direction_id=direction_id,
                trip_date=start_date,
                scheduled_start=start_time + ":00",
                stops=stop_predictions,
            ))

        if record:
            self.record_forecasts(forecasts)

        return forecasts
