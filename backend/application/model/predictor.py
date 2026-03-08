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
import joblib

from model import BusLSTM, OccupancyLSTM

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

PARQUET_DIR = PROJECT_ROOT / "parquets"
ROUTE_ENCODING_PATH = PARQUET_DIR / "route_encoding.json"
ROUTE_ENCODER_PKL_PATH = PARQUET_DIR / "route_encoder.pkl"
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
        config_path: str,
        time_weights_path: str,
        crowd_weights_path: str,
        route_encoding_path: Optional[str] = None,
        h3_encoding_path: Optional[str] = None,
        static_map_path: Optional[str] = None,
    ):
        print(f"Loading DUAL models...")

        route_path = Path(route_encoding_path) if route_encoding_path else ROUTE_ENCODING_PATH
        route_encoder_pkl_path = ROUTE_ENCODER_PKL_PATH
        h3_path = Path(h3_encoding_path) if h3_encoding_path else H3_ENCODING_PATH
        static_path = Path(static_map_path) if static_map_path else STATIC_MAP_PATH

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

        with open(h3_path, "r") as f:
            self.h3_encoder = json.load(f)

        self.static_map = pd.read_parquet(static_path)
        self.static_map["route_id_norm"] = self.static_map["route_id"].astype(str)
        self.static_map["direction_id_norm"] = self.static_map["direction_id"].astype(str)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        model_kwargs = {
            "n_x1_dense_features": self.config["x1_dense_features"],
            "n_x2_dense_features": self.config["x2_dense_features"],
            "x1_cat_cardinalities": self.config["x1_cat_cards"],
            "x2_cat_cardinalities": self.config["x2_cat_cards"],
            "encoder_hidden_size": self.config["encoder_hidden_size"],
            "lstm_hidden_size": self.config["decoder_hidden_size"],
        }
        
        crowd_kwargs = model_kwargs.copy()
        crowd_kwargs["num_lstm_layers"] = self.config.get("num_lstm_layers", 2)

        # Inizializziamo i DUE modelli
        self.model_time = BusLSTM(**model_kwargs)
        self.model_crowd = OccupancyLSTM(**crowd_kwargs)

        # Carichiamo i pesi
        state_dict_time = torch.load(time_weights_path, map_location=torch.device("cpu"), weights_only=True)
        state_dict_crowd = torch.load(crowd_weights_path, map_location=torch.device("cpu"), weights_only=True)
        
        # Pulizia chiavi se necessario
        if any(k.startswith("_orig_mod.") for k in state_dict_time.keys()):
            state_dict_time = {k.replace("_orig_mod.", ""): v for k, v in state_dict_time.items()}
        if any(k.startswith("_orig_mod.") for k in state_dict_crowd.keys()):
            state_dict_crowd = {k.replace("_orig_mod.", ""): v for k, v in state_dict_crowd.items()}

        self.model_time.load_state_dict(state_dict_time)
        self.model_crowd.load_state_dict(state_dict_crowd)
        
        self.model_time.eval()
        self.model_crowd.eval()
        print(f"Dual models loaded successfully.")

    def _sanitize_categorical_inputs(self, x1_cat_batch: np.ndarray, x2_cat_batch: np.ndarray, context: str) -> tuple[np.ndarray, np.ndarray]:
        x1_cards = self.config["x1_cat_cards"]
        x2_cards = self.config["x2_cat_cards"]

        for col_idx, card in enumerate(x1_cards):
            if card <= 0: continue
            invalid_mask = (x1_cat_batch[:, col_idx] < 0) | (x1_cat_batch[:, col_idx] >= card)
            if int(invalid_mask.sum()) > 0:
                x1_cat_batch[invalid_mask, col_idx] = 0

        for col_idx, card in enumerate(x2_cards):
            if card <= 0: continue
            invalid_mask = (x2_cat_batch[:, :, col_idx] < 0) | (x2_cat_batch[:, :, col_idx] >= card)
            if int(invalid_mask.sum()) > 0:
                x2_cat_batch[:, :, col_idx][invalid_mask] = 0

        return x1_cat_batch, x2_cat_batch

    def has_trip_template(self, route_id: str, direction_id: Any) -> bool:
        trip_static = self.static_map[
            (self.static_map["route_id_norm"] == str(route_id)) & (self.static_map["direction_id_norm"] == str(direction_id))
        ]
        return not trip_static.empty and len(trip_static) == BATCH_SIZE

    def _get_day_type(self, d: date) -> int:
        weekday = d.weekday()
        if weekday == 5: return 1
        elif weekday == 6: return 2
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

    def _predict_tensors(self, x1_cat_raw: list, x1_dense_raw: list, x2_cat_raw: np.ndarray, x2_dense_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        route_id_str = str(x1_cat_raw[0])
        x1_cat_raw[0] = self.route_encoder.get(route_id_str, 0)

        x1_dense_raw[13] = np.clip(x1_dense_raw[13], 1, 3) / 3.0
        x2_dense_raw[:, 0] = np.clip(x2_dense_raw[:, 0], 0, 15000) / 15000.0
        x2_dense_raw[:, 1] = np.clip(x2_dense_raw[:, 1], 0, 1000) / 1000.0
        x2_dense_raw[:, 2] = x2_dense_raw[:, 2] / 99.0
        x2_dense_raw[:, 4] = np.clip(x2_dense_raw[:, 4], 0, 65.0) / 65.0
        x2_dense_raw[:, 5] = np.clip(x2_dense_raw[:, 5], 0.0, 2.0)

        x1_cat_np = np.asarray(x1_cat_raw, dtype=np.int64).reshape(1, -1)
        x2_cat_np = np.asarray(x2_cat_raw, dtype=np.int64).reshape(1, BATCH_SIZE, -1)
        x1_cat_np, x2_cat_np = self._sanitize_categorical_inputs(x1_cat_np, x2_cat_np, context="single_trip")

        x1_cat_tensor = torch.tensor(x1_cat_np[0], dtype=torch.int64).unsqueeze(0)
        x1_dense_tensor = torch.tensor(x1_dense_raw, dtype=torch.float32).unsqueeze(0)
        x2_cat_tensor = torch.tensor(x2_cat_np[0], dtype=torch.int64).unsqueeze(0)
        x2_dense_tensor = torch.tensor(x2_dense_raw, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # ISTRUZIONE AI DUE EMISFERI
            pred_time_scaled = self.model_time(x1_cat_tensor, x1_dense_tensor, x2_cat_tensor, x2_dense_tensor)
            pred_crowd_logits = self.model_crowd(x1_cat_tensor, x1_dense_tensor, x2_cat_tensor, x2_dense_tensor)

        # FATTORE DI SCALING MODIFICATO A 600.0 (10 MINUTI)
        delays = pred_time_scaled.squeeze().numpy() * 600.0
        crowd = pred_crowd_logits.argmax(dim=-1).squeeze().numpy()

        return delays, crowd

    def get_trip_forecast_raw(self, route_id: str, direction_id: int, time_seconds: int, day_type: int, weather_code: int, bus_type: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        trip_static = self.static_map[
            (self.static_map["route_id_norm"] == str(route_id)) & (self.static_map["direction_id_norm"] == str(direction_id))
        ]
        if trip_static.empty or len(trip_static) != 100:
            raise ValueError(f"Route {route_id} not found")

        trip_static = trip_static.sort_values("segment_idx")

        x1_cat = [route_id, int(direction_id), int(day_type), min(int(int(weather_code) / 33), 2), int(float(bus_type) / 9.0)]
        time_sin = math.sin(2 * math.pi * time_seconds / 86400)
        time_cos = math.cos(2 * math.pi * time_seconds / 86400)
        x1_dense = [0.0] * 13 + [3.0, time_sin, time_cos]

        h3_encoded = [self.h3_encoder.get(h, 0) if h else 0 for h in trip_static["h3_index"].values]
        stop_seq = trip_static["stop_sequence"].fillna(-1).astype(int).values
        x2_cat = np.column_stack((h3_encoded, stop_seq))

        shape_dist = trip_static["can_shape_dist_travelled"].values
        dist_to_next = trip_static["can_distance_to_next_stop"].values
        seg_idx = trip_static["segment_idx"].values
        is_gen = np.ones(100)
        base_speed = np.full(100, 25.0)
        speed_ratio = np.full(100, 1.0)

        x2_dense = np.column_stack((shape_dist, dist_to_next, seg_idx, is_gen, base_speed, speed_ratio))

        delays, crowd = self._predict_tensors(x1_cat, x1_dense, x2_cat, x2_dense)
        return delays, crowd, trip_static

    def get_trip_forecast(self, route_id: str, direction_id: int, start_date: str, start_time: str, weather_code: int, bus_type: int) -> TripForecast:
        from datetime import datetime
        trip_date = datetime.strptime(start_date, "%d-%m-%Y").date()
        time_parts = start_time.split(":")
        time_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60
        day_type = self._get_day_type(trip_date)

        delays, crowd, trip_static = self.get_trip_forecast_raw(route_id, direction_id, time_seconds, day_type, weather_code, bus_type)

        stop_groups = trip_static.groupby("stop_sequence", sort=True)
        stop_predictions = []
        for stop_seq, group in stop_groups:
            last_segment_idx = group["segment_idx"].max()
            last_segment_mask = trip_static["segment_idx"] == last_segment_idx

            delay_at_stop = delays[last_segment_idx]
            crowd_at_stop = int(crowd[last_segment_idx])
            distance_m = trip_static.loc[last_segment_mask, "can_shape_dist_travelled"].values[0]
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
            route_id=route_id, direction_id=direction_id, trip_date=start_date,
            scheduled_start=start_time + ":00", stops=stop_predictions,
        )

    def _predict_batch_tensors(self, x1_cat_batch: np.ndarray, x1_dense_batch: np.ndarray, x2_cat_batch: np.ndarray, x2_dense_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x1_cat_batch, x2_cat_batch = self._sanitize_categorical_inputs(x1_cat_batch, x2_cat_batch, context="batch")

        x1_dense_batch[:, 13] = np.clip(x1_dense_batch[:, 13], 1, 3) / 3.0
        x2_dense_batch[:, :, 0] = np.clip(x2_dense_batch[:, :, 0], 0, 15000) / 15000.0
        x2_dense_batch[:, :, 1] = np.clip(x2_dense_batch[:, :, 1], 0, 1000) / 1000.0
        x2_dense_batch[:, :, 2] = x2_dense_batch[:, :, 2] / 99.0
        x2_dense_batch[:, :, 4] = np.clip(x2_dense_batch[:, :, 4], 0, 65.0) / 65.0
        x2_dense_batch[:, :, 5] = np.clip(x2_dense_batch[:, :, 5], 0.0, 2.0)

        x1_cat_tensor = torch.tensor(x1_cat_batch, dtype=torch.int64)
        x1_dense_tensor = torch.tensor(x1_dense_batch, dtype=torch.float32)
        x2_cat_tensor = torch.tensor(x2_cat_batch, dtype=torch.int64)
        x2_dense_tensor = torch.tensor(x2_dense_batch, dtype=torch.float32)

        with torch.no_grad():
            pred_time_scaled = self.model_time(x1_cat_tensor, x1_dense_tensor, x2_cat_tensor, x2_dense_tensor)
            pred_crowd_logits = self.model_crowd(x1_cat_tensor, x1_dense_tensor, x2_cat_tensor, x2_dense_tensor)

        # FATTORE DI SCALING MODIFICATO A 600.0 (10 MINUTI)
        delays = pred_time_scaled.squeeze(-1).numpy() * 600.0
        crowd = pred_crowd_logits.argmax(dim=-1).numpy()

        return delays, crowd

    def get_batch_forecast(self, trips: List[Dict[str, Any]]) -> List[TripForecast]:
        # ... Rimane invariata eccetto il richiamo nativo a _predict_batch_tensors ...
        pass
