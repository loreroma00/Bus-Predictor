import os
import json
import math
from pathlib import Path
from datetime import date
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import numpy as np
import pandas as pd

from model import BusLSTM, BusODELSTM

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

PARQUET_DIR = PROJECT_ROOT / "parquets"
ROUTE_ENCODING_PATH = PARQUET_DIR / "route_encoding.json"
H3_ENCODING_PATH = PARQUET_DIR / "h3_encoding.json"
STATIC_MAP_PATH = PARQUET_DIR / "canonical_route_map.parquet"

BATCH_SIZE = 100  # Number of segments per trip


@dataclass
class StopPrediction:
    stop_sequence: int
    distance_m: float
    cumulative_delay_sec: float
    delay_formatted: str
    expected_arrival: str
    crowd_level: int


@dataclass
class TripForecast:
    route_id: str
    direction_id: int
    trip_date: str
    scheduled_start: str
    stops: list[StopPrediction]


class Predictor:
    def __init__(
        self,
        weights_path: str,
        config_path: str,
        route_encoding_path: Optional[str] = None,
        h3_encoding_path: Optional[str] = None,
        static_map_path: Optional[str] = None,
    ):
        print(f"Loading model from: {os.path.basename(weights_path)}")

        route_path = (
            Path(route_encoding_path) if route_encoding_path else ROUTE_ENCODING_PATH
        )
        h3_path = Path(h3_encoding_path) if h3_encoding_path else H3_ENCODING_PATH
        static_path = Path(static_map_path) if static_map_path else STATIC_MAP_PATH

        with open(route_path, "r") as f:
            self.route_encoder = json.load(f)

        with open(h3_path, "r") as f:
            self.h3_encoder = json.load(f)

        self.static_map = pd.read_parquet(static_path)
        self.static_map["route_id_norm"] = self.static_map["route_id"].astype(str)
        self.static_map["direction_id_norm"] = self.static_map["direction_id"].astype(
            str
        )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        arch = self.config.get("architecture", "lstm")
        model_class = BusODELSTM if arch == "ode_lstm" else BusLSTM

        model_kwargs = {
            "n_x1_dense_features": self.config["x1_dense_features"],
            "n_x2_dense_features": self.config["x2_dense_features"],
            "x1_cat_cardinalities": self.config["x1_cat_cards"],
            "x2_cat_cardinalities": self.config["x2_cat_cards"],
            "encoder_hidden_size": self.config["encoder_hidden_size"],
            "lstm_hidden_size": self.config["decoder_hidden_size"],
        }
        if arch == "lstm":
            model_kwargs["num_lstm_layers"] = self.config.get("num_lstm_layers", 2)

        self.model = model_class(**model_kwargs)

        state_dict = torch.load(
            weights_path, map_location=torch.device("cpu"), weights_only=True
        )
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Model loaded. Encoder hidden: {self.config['encoder_hidden_size']}")

    def _sanitize_categorical_inputs(
        self,
        x1_cat_batch: np.ndarray,
        x2_cat_batch: np.ndarray,
        context: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        x1_cards = self.config["x1_cat_cards"]
        x2_cards = self.config["x2_cat_cards"]

        for col_idx, card in enumerate(x1_cards):
            if card <= 0:
                continue
            invalid_mask = (x1_cat_batch[:, col_idx] < 0) | (
                x1_cat_batch[:, col_idx] >= card
            )
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                print(
                    f"[WARN] {context}: x1_cat col={col_idx} had {invalid_count} out-of-range values; clipping to [0, {card - 1}]"
                )
                x1_cat_batch[:, col_idx] = np.clip(
                    x1_cat_batch[:, col_idx], 0, card - 1
                )

        for col_idx, card in enumerate(x2_cards):
            if card <= 0:
                continue
            invalid_mask = (x2_cat_batch[:, :, col_idx] < 0) | (
                x2_cat_batch[:, :, col_idx] >= card
            )
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                print(
                    f"[WARN] {context}: x2_cat col={col_idx} had {invalid_count} out-of-range values; clipping to [0, {card - 1}]"
                )
                x2_cat_batch[:, :, col_idx] = np.clip(
                    x2_cat_batch[:, :, col_idx], 0, card - 1
                )

        return x1_cat_batch, x2_cat_batch

    def has_trip_template(self, route_id: str, direction_id: Any) -> bool:
        trip_static = self.static_map[
            (self.static_map["route_id_norm"] == str(route_id))
            & (self.static_map["direction_id_norm"] == str(direction_id))
        ]
        return not trip_static.empty and len(trip_static) == BATCH_SIZE

    def _get_day_type(self, d: date) -> int:
        weekday = d.weekday()
        if weekday == 5:
            return 1
        elif weekday == 6:
            return 2
        return 0

    def _format_delay(self, delay_seconds: float) -> str:
        total_seconds = int(round(delay_seconds))
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{sign}{minutes}m {seconds}s"

    def _format_time(self, time_seconds: float) -> str:
        total_seconds = int(round(time_seconds)) % 86400
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _predict_tensors(
        self,
        x1_cat_raw: list,
        x1_dense_raw: list,
        x2_cat_raw: np.ndarray,
        x2_dense_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        route_id_str = x1_cat_raw[0]
        x1_cat_raw[0] = self.route_encoder.get(route_id_str, 0)

        x1_dense_raw[13] = np.clip(x1_dense_raw[13], 1, 3) / 3.0

        x2_dense_raw[:, 0] = np.clip(x2_dense_raw[:, 0], 0, 15000) / 15000.0
        x2_dense_raw[:, 1] = np.clip(x2_dense_raw[:, 1], 0, 1000) / 1000.0
        x2_dense_raw[:, 2] = x2_dense_raw[:, 2] / 99.0
        x2_dense_raw[:, 4] = np.clip(x2_dense_raw[:, 4], 0, 65.0) / 65.0
        x2_dense_raw[:, 5] = np.clip(x2_dense_raw[:, 5], 0.0, 2.0)

        x1_cat_np = np.asarray(x1_cat_raw, dtype=np.int64).reshape(1, -1)
        x2_cat_np = np.asarray(x2_cat_raw, dtype=np.int64).reshape(1, BATCH_SIZE, -1)
        x1_cat_np, x2_cat_np = self._sanitize_categorical_inputs(
            x1_cat_np, x2_cat_np, context="single_trip"
        )

        x1_cat_tensor = torch.tensor(x1_cat_np[0], dtype=torch.int64).unsqueeze(0)
        x1_dense_tensor = torch.tensor(x1_dense_raw, dtype=torch.float32).unsqueeze(0)
        x2_cat_tensor = torch.tensor(x2_cat_np[0], dtype=torch.int64).unsqueeze(0)
        x2_dense_tensor = torch.tensor(x2_dense_raw, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_time_scaled, pred_crowd_logits = self.model(
                x1_cat_tensor, x1_dense_tensor, x2_cat_tensor, x2_dense_tensor
            )

        delays = pred_time_scaled.squeeze().numpy() * 3600.0
        crowd = pred_crowd_logits.argmax(dim=-1).squeeze().numpy()

        return delays, crowd

    def get_trip_forecast_raw(
        self,
        route_id: str,
        direction_id: int,
        time_seconds: int,
        day_type: int,
        weather_code: int,
        bus_type: int,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        trip_static = self.static_map[
            (self.static_map["route_id_norm"] == str(route_id))
            & (self.static_map["direction_id_norm"] == str(direction_id))
        ]

        if trip_static.empty or len(trip_static) != 100:
            raise ValueError(
                f"Route {route_id} (direction {direction_id}) not found in canonical map"
            )

        trip_static = trip_static.sort_values("segment_idx")

        x1_cat = [route_id, direction_id, day_type, weather_code, bus_type]
        time_sin = math.sin(2 * math.pi * time_seconds / 86400)
        time_cos = math.cos(2 * math.pi * time_seconds / 86400)
        x1_dense = [0.0] * 13 + [3.0, time_sin, time_cos]

        h3_encoded = [
            self.h3_encoder.get(h, 0) if h else 0
            for h in trip_static["h3_index"].values
        ]
        stop_seq = trip_static["stop_sequence"].fillna(-1).astype(int).values
        x2_cat = np.column_stack((h3_encoded, stop_seq))

        shape_dist = trip_static["can_shape_dist_travelled"].values
        dist_to_next = trip_static["can_distance_to_next_stop"].values
        seg_idx = trip_static["segment_idx"].values
        is_gen = np.ones(100)

        base_speed = np.full(100, 25.0)
        speed_ratio = np.full(100, 1.0)

        x2_dense = np.column_stack(
            (shape_dist, dist_to_next, seg_idx, is_gen, base_speed, speed_ratio)
        )

        delays, crowd = self._predict_tensors(x1_cat, x1_dense, x2_cat, x2_dense)

        return delays, crowd, trip_static

    def get_trip_forecast(
        self,
        route_id: str,
        direction_id: int,
        start_date: str,
        start_time: str,
        weather_code: int,
        bus_type: int,
    ) -> TripForecast:
        from datetime import datetime

        trip_date = datetime.strptime(start_date, "%d-%m-%Y").date()
        time_parts = start_time.split(":")
        time_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60

        day_type = self._get_day_type(trip_date)

        delays, crowd, trip_static = self.get_trip_forecast_raw(
            route_id=route_id,
            direction_id=direction_id,
            time_seconds=time_seconds,
            day_type=day_type,
            weather_code=weather_code,
            bus_type=bus_type,
        )

        stop_groups = trip_static.groupby("stop_sequence", sort=True)

        stop_predictions = []
        for stop_seq, group in stop_groups:
            last_segment_idx = group["segment_idx"].max()
            last_segment_mask = trip_static["segment_idx"] == last_segment_idx

            delay_at_stop = delays[last_segment_idx]
            crowd_at_stop = int(crowd[last_segment_idx])
            distance_m = trip_static.loc[
                last_segment_mask, "can_shape_dist_travelled"
            ].values[0]

            expected_arrival_seconds = time_seconds + delay_at_stop

            stop_predictions.append(
                StopPrediction(
                    stop_sequence=int(stop_seq),
                    distance_m=float(distance_m),
                    cumulative_delay_sec=float(delay_at_stop),
                    delay_formatted=self._format_delay(delay_at_stop),
                    expected_arrival=self._format_time(expected_arrival_seconds),
                    crowd_level=crowd_at_stop,
                )
            )

        return TripForecast(
            route_id=route_id,
            direction_id=direction_id,
            trip_date=start_date,
            scheduled_start=start_time + ":00",
            stops=stop_predictions,
        )

    def _predict_batch_tensors(
        self,
        x1_cat_batch: np.ndarray,
        x1_dense_batch: np.ndarray,
        x2_cat_batch: np.ndarray,
        x2_dense_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batched inference for multiple trips.

        Args:
            x1_cat_batch: (N, 5) - route_id, direction_id, day_type, weather_code, bus_type
            x1_dense_batch: (N, 16) - static features
            x2_cat_batch: (N, 100, 2) - h3_encoded, stop_sequence per segment
            x2_dense_batch: (N, 100, 6) - dynamic features per segment

        Returns:
            delays: (N, 100) - delay in seconds for each segment
            crowd: (N, 100) - crowd level for each segment
        """
        x1_cat_batch, x2_cat_batch = self._sanitize_categorical_inputs(
            x1_cat_batch, x2_cat_batch, context="batch"
        )

        # Normalize x1_dense
        x1_dense_batch[:, 13] = np.clip(x1_dense_batch[:, 13], 1, 3) / 3.0

        # Normalize x2_dense
        x2_dense_batch[:, :, 0] = np.clip(x2_dense_batch[:, :, 0], 0, 15000) / 15000.0
        x2_dense_batch[:, :, 1] = np.clip(x2_dense_batch[:, :, 1], 0, 1000) / 1000.0
        x2_dense_batch[:, :, 2] = x2_dense_batch[:, :, 2] / 99.0
        x2_dense_batch[:, :, 4] = np.clip(x2_dense_batch[:, :, 4], 0, 65.0) / 65.0
        x2_dense_batch[:, :, 5] = np.clip(x2_dense_batch[:, :, 5], 0.0, 2.0)

        # Convert to tensors
        x1_cat_tensor = torch.tensor(x1_cat_batch, dtype=torch.int64)
        x1_dense_tensor = torch.tensor(x1_dense_batch, dtype=torch.float32)
        x2_cat_tensor = torch.tensor(x2_cat_batch, dtype=torch.int64)
        x2_dense_tensor = torch.tensor(x2_dense_batch, dtype=torch.float32)

        # Batched inference
        with torch.no_grad():
            pred_time_scaled, pred_crowd_logits = self.model(
                x1_cat_tensor, x1_dense_tensor, x2_cat_tensor, x2_dense_tensor
            )

        delays = pred_time_scaled.numpy() * 3600.0
        crowd = pred_crowd_logits.argmax(dim=-1).numpy()

        return delays, crowd

    def get_batch_forecast(
        self,
        trips: List[Dict[str, Any]],
    ) -> List[TripForecast]:
        """
        Predict multiple trips in a single batched inference pass.

        Args:
            trips: List of dicts, each with:
                - route_id: str
                - direction_id: int
                - start_date: str (DD-MM-YYYY)
                - start_time: str (HH:MM)
                - weather_code: int
                - bus_type: int

        Returns:
            List[TripForecast] - one per input trip, in same order

        Raises:
            ValueError: If any trip has invalid route_id/direction_id
        """
        from datetime import datetime

        if not trips:
            return []

        # 1. Validate all trips and collect metadata
        trip_metadata = []
        invalid_trips = []

        for idx, trip in enumerate(trips):
            route_id = str(trip["route_id"])
            direction_id = trip["direction_id"]
            try:
                direction_id = int(direction_id)
            except (TypeError, ValueError):
                invalid_trips.append(
                    f"Trip {idx}: route_id={route_id}, direction_id={direction_id} (invalid direction_id)"
                )
                continue

            trip_static = self.static_map[
                (self.static_map["route_id_norm"] == route_id)
                & (self.static_map["direction_id_norm"] == str(direction_id))
            ]

            if trip_static.empty or len(trip_static) != BATCH_SIZE:
                invalid_trips.append(
                    f"Trip {idx}: route_id={route_id}, direction_id={direction_id}"
                )
                continue

            trip_static = trip_static.sort_values("segment_idx")

            trip_date = datetime.strptime(trip["start_date"], "%d-%m-%Y").date()
            time_parts = trip["start_time"].split(":")
            time_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60
            day_type = self._get_day_type(trip_date)

            trip_metadata.append(
                {
                    "idx": idx,
                    "route_id": route_id,
                    "direction_id": direction_id,
                    "route_id_encoded": int(self.route_encoder.get(route_id, 0)),
                    "trip_date": trip["start_date"],
                    "start_time": trip["start_time"],
                    "time_seconds": time_seconds,
                    "day_type": day_type,
                    "weather_code": trip["weather_code"],
                    "bus_type": trip["bus_type"],
                    "trip_static": trip_static,
                }
            )

        if invalid_trips:
            raise ValueError(f"Invalid trips found:\n" + "\n".join(invalid_trips))

        n_trips = len(trip_metadata)

        # 2. Build batch tensors
        x1_cat_batch = np.zeros((n_trips, 5), dtype=np.int64)
        x1_dense_batch = np.zeros((n_trips, 16), dtype=np.float32)
        x2_cat_batch = np.zeros((n_trips, BATCH_SIZE, 2), dtype=np.int64)
        x2_dense_batch = np.zeros((n_trips, BATCH_SIZE, 6), dtype=np.float32)

        for i, meta in enumerate(trip_metadata):
            trip_static = meta["trip_static"]

            # x1_cat: [route_id, direction_id, day_type, weather_code, bus_type]
            x1_cat_batch[i] = [
                int(meta["route_id_encoded"]),
                int(meta["direction_id"]),
                int(meta["day_type"]),
                int(meta["weather_code"]),
                int(meta["bus_type"]),
            ]

            # x1_dense: 13 zeros + [3.0, time_sin, time_cos]
            time_sin = math.sin(2 * math.pi * meta["time_seconds"] / 86400)
            time_cos = math.cos(2 * math.pi * meta["time_seconds"] / 86400)
            x1_dense_batch[i] = [0.0] * 13 + [3.0, time_sin, time_cos]

            # x2_cat: [h3_encoded, stop_sequence] for each segment
            h3_encoded = [
                self.h3_encoder.get(h, 0) if h else 0
                for h in trip_static["h3_index"].values
            ]
            stop_seq = trip_static["stop_sequence"].fillna(-1).astype(int).values
            x2_cat_batch[i, :, 0] = h3_encoded
            x2_cat_batch[i, :, 1] = stop_seq

            # x2_dense: [shape_dist, dist_to_next, seg_idx, is_gen, base_speed, speed_ratio]
            x2_dense_batch[i, :, 0] = trip_static["can_shape_dist_travelled"].values
            x2_dense_batch[i, :, 1] = trip_static["can_distance_to_next_stop"].values
            x2_dense_batch[i, :, 2] = trip_static["segment_idx"].values
            x2_dense_batch[i, :, 3] = 1.0  # is_genuine
            x2_dense_batch[i, :, 4] = 25.0  # base_speed
            x2_dense_batch[i, :, 5] = 1.0  # speed_ratio

        # 3. Batched inference
        delays_batch, crowd_batch = self._predict_batch_tensors(
            x1_cat_batch, x1_dense_batch, x2_cat_batch, x2_dense_batch
        )

        # 4. Build TripForecast objects
        forecasts = []

        for i, meta in enumerate(trip_metadata):
            trip_static = meta["trip_static"]
            delays = delays_batch[i]
            crowd = crowd_batch[i]
            time_seconds = meta["time_seconds"]

            # Build stop predictions
            stop_groups = trip_static.groupby("stop_sequence", sort=True)
            stop_predictions = []

            for stop_seq, group in stop_groups:
                last_segment_idx = group["segment_idx"].max()
                last_segment_mask = trip_static["segment_idx"] == last_segment_idx

                delay_at_stop = delays[last_segment_idx]
                crowd_at_stop = int(crowd[last_segment_idx])
                distance_m = trip_static.loc[
                    last_segment_mask, "can_shape_dist_travelled"
                ].values[0]

                expected_arrival_seconds = time_seconds + delay_at_stop

                stop_predictions.append(
                    StopPrediction(
                        stop_sequence=int(stop_seq),
                        distance_m=float(distance_m),
                        cumulative_delay_sec=float(delay_at_stop),
                        delay_formatted=self._format_delay(delay_at_stop),
                        expected_arrival=self._format_time(expected_arrival_seconds),
                        crowd_level=crowd_at_stop,
                    )
                )

            forecasts.append(
                TripForecast(
                    route_id=meta["route_id"],
                    direction_id=meta["direction_id"],
                    trip_date=meta["trip_date"],
                    scheduled_start=meta["start_time"] + ":00",
                    stops=stop_predictions,
                )
            )

        return forecasts
