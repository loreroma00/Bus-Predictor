import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
STOP_ROUTE_CONFIG = os.path.join(PROJECT_ROOT, "parquets", "stop_route_config.json")


class BusDataset(Dataset):
    def __init__(self, file_path):
        # 1. READING DATA
        print(f"Reading dataset from {file_path}...")
        trips_df = pd.read_parquet(file_path)

        # Load MAX_STOPS from config
        with open(STOP_ROUTE_CONFIG, "r") as f:
            config = json.load(f)
        self.max_stops = config["max_stops"]
        print(f"MAX_STOPS = {self.max_stops}")

        # 2. DEFINING CATEGORICAL COLUMNS
        self.x1_cat_cols = [
            "route_id",
            "direction_id",
            "day_type",
            "weather_code",
            "bus_type",
        ]
        self.x2_cat_cols = ["h3_index_encoded", "stop_sequence"]
        self.x1_cat = trips_df[self.x1_cat_cols].values.astype(np.int64)
        self.x2_cat = trips_df[self.x2_cat_cols].values.astype(np.int64)

        # 3. DEFINING DENSE COLUMNS
        self.x1_dense_cols = [
            "deposit_grottarossa",
            "deposit_magliana",
            "deposit_tor_sapienza",
            "deposit_portonaccio",
            "deposit_monte_sacro",
            "deposit_tor_pagnotta",
            "deposit_tor_cervara",
            "deposit_maglianella",
            "deposit_costi",
            "deposit_trastevere",
            "deposit_acilia",
            "deposit_tor_vergata",
            "deposit_porta_maggiore",
            "door_number",
            "scheduled_start_time_sin",
            "scheduled_start_time_cos",
        ]
        self.x2_dense_cols = [
            "can_shape_dist_travelled",
            "can_distance_to_next_stop",
            "stop_idx",
            "is_genuine",
            "can_avg_speed_ratio",
            "can_avg_traffic_speed",
        ]
        self.x1_dense = trips_df[self.x1_dense_cols].values.astype(np.float32)
        self.x2_dense = trips_df[self.x2_dense_cols].values.astype(np.float32)

        # 4. DEFINING LABELS
        self.y_time = trips_df["schedule_adherence"].values.astype(np.float32)
        self.y_crowd = trips_df["occupancy_status"].values.astype(np.int64)

        # 5. LENGTHS & T_GRID (per-trip metadata for variable-length sequences)
        # num_stops is constant within a trip (replicated across all rows)
        # t_grid is the normalized cumulative stop distance for ODE integration
        self.lengths_raw = trips_df["num_stops"].values.astype(np.int64)
        self.t_grid_raw = trips_df["t_grid"].values.astype(np.float32)

        # 6. SLICING & RESHAPING
        ms = self.max_stops
        self.num_trips = len(trips_df) // ms

        print(f"Reshaping data for {self.num_trips} trips (max_stops={ms})...")

        # Trip-level features: take first row of each trip (every ms rows)
        self.x1_cat = torch.tensor(
            trips_df[self.x1_cat_cols].iloc[::ms].values.astype(np.int64)
        )
        self.x1_dense = torch.tensor(
            trips_df[self.x1_dense_cols].iloc[::ms].values.astype(np.float32)
        )

        # Sequence-level features: reshape to [num_trips, max_stops, features]
        # Clamp x2_cat to >= 0: padded rows have -1 which is invalid for nn.Embedding on CUDA.
        # Padded positions are masked during loss, so the embedding output there is irrelevant.
        self.x2_cat = np.clip(self.x2_cat, 0, None)
        self.x2_cat = torch.tensor(self.x2_cat.reshape(self.num_trips, ms, -1))
        self.x2_dense = torch.tensor(self.x2_dense.reshape(self.num_trips, ms, -1))
        self.y_time = torch.tensor(self.y_time.reshape(self.num_trips, ms, 1))
        self.y_crowd = torch.tensor(self.y_crowd.reshape(self.num_trips, ms))

        # Per-trip lengths: take first row of each trip
        self.lengths = torch.tensor(
            self.lengths_raw[::ms], dtype=torch.int64
        )
        self.t_grid = torch.tensor(
            self.t_grid_raw.reshape(self.num_trips, ms), dtype=torch.float32
        )

        # 7. METADATA FOR MODEL BUILDING
        # Only count cardinalities from non-padded rows (stop_idx >= 0, h3 >= 0)
        real_mask = trips_df["num_stops"].values > 0  # all rows have num_stops > 0
        self.x1_cat_cardinalities = [
            int(trips_df[c].max() + 1) for c in self.x1_cat_cols
        ]
        self.x2_cat_cardinalities = [
            int(trips_df.loc[trips_df["stop_sequence"] >= 0, c].max() + 1)
            if c == "stop_sequence"
            else int(trips_df.loc[trips_df["h3_index_encoded"] >= 0, c].max() + 1)
            for c in self.x2_cat_cols
        ]

        self.n_x1_dense_features = len(self.x1_dense_cols)
        self.n_x2_dense_features = len(self.x2_dense_cols)

        print(f"Lengths distribution: min={self.lengths.min().item()}, "
              f"max={self.lengths.max().item()}, "
              f"mean={self.lengths.float().mean().item():.1f}")
        print("Dataset initialized successfully.")

    def __len__(self):
        return self.num_trips

    def __getitem__(self, idx):
        return (
            self.x1_cat[idx],
            self.x1_dense[idx],
            self.x2_cat[idx],
            self.x2_dense[idx],
            self.y_time[idx],
            self.y_crowd[idx],
            self.lengths[idx],
            self.t_grid[idx],
        )


def load_dataset(path: str, train_size: float):
    full_dataset = BusDataset(path)

    train_size = int(train_size * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    print(f"Training on {len(train_ds)} trips. Validation on {len(val_ds)} trips.")

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True
    )

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, val_loader, full_dataset
