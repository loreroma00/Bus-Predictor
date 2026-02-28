import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np


class BusDataset(Dataset):
    def __init__(self, file_path):
        # 1. READING DATA
        print(f"Reading dataset from {file_path}...")
        trips_df = pd.read_parquet(file_path)

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
            "segment_idx",
            "is_genuine",
            "can_avg_speed_ratio",
            "can_avg_traffic_speed",
        ]
        self.x1_dense = trips_df[self.x1_dense_cols].values.astype(np.float32)
        self.x2_dense = trips_df[self.x2_dense_cols].values.astype(np.float32)

        # 4. DEFINING LABELS
        self.y_time = trips_df["schedule_adherence"].values.astype(np.float32)
        self.y_crowd = trips_df["occupancy_status"].values.astype(np.int64)

        # 5. SLICING & RESHAPING
        self.num_trips = len(trips_df) // 100

        print(f"Reshaping data for {self.num_trips} trips...")

        # Trasformazione in Tensori 3D [Batch, Time, Feats]
        self.x1_cat = torch.tensor(
            trips_df[self.x1_cat_cols].iloc[::100].values.astype(np.int64)
        )
        self.x2_cat = torch.tensor(self.x2_cat.reshape(self.num_trips, 100, -1))
        self.x1_dense = torch.tensor(
            trips_df[self.x1_dense_cols].iloc[::100].values.astype(np.float32)
        )
        self.x2_dense = torch.tensor(self.x2_dense.reshape(self.num_trips, 100, -1))
        self.y_time = torch.tensor(self.y_time.reshape(self.num_trips, 100, 1))
        self.y_crowd = torch.tensor(self.y_crowd.reshape(self.num_trips, 100))

        # 6. METADATA FOR LSTM BUILDING
        self.x1_cat_cardinalities = [
            int(trips_df[c].max() + 1) for c in self.x1_cat_cols
        ]
        self.x2_cat_cardinalities = [
            int(trips_df[c].max() + 1) for c in self.x2_cat_cols
        ]

        self.n_x1_dense_features = len(self.x1_dense_cols)
        self.n_x2_dense_features = len(self.x2_dense_cols)

        print("Dataset initialized successfully.")

    def __len__(self):
        return self.num_trips

    def __getitem__(self, idx):
        """Restituisce la tupla per un singolo viaggio"""
        return (
            self.x1_cat[idx],
            self.x1_dense[idx],
            self.x2_cat[idx],
            self.x2_dense[idx],
            self.y_time[idx],
            self.y_crowd[idx],
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
