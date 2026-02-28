import os
import json
import math
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from model import BusLSTM

"""
DEPRECATED: This file is deprecated and will be removed in a future version.
    
    - For API usage, use main.py which imports from predictor.py
    - For CLI access, a new CLI tool will be created in frontend-thesis/
    
    This file is kept for backward compatibility only.
"""

import warnings

warnings.warn(
    "predict.py is deprecated. Use predictor.py for core logic or main.py for the API.",
    DeprecationWarning,
    stacklevel=2,
)

# 1. Calcolo dinamico della Root assoluta
# predict.py è in backend-thesis/application/model/
CURRENT_DIR = Path(__file__).resolve().parents[0]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Percorsi dei dati pre-processati
PARQUET_DIR = PROJECT_ROOT / "parquets"
ROUTE_ENCODER_PATH = PARQUET_DIR / "route_encoder.pkl"
STATIC_MAP_PATH = PARQUET_DIR / "canonical_route_map.parquet"
H3_ENCODER_PATH = PARQUET_DIR / "h3_encoding.json"

# 3. Percorsi generati dal Training (nella stessa cartella di predict.py)
HYPERPARAMS_PATH = CURRENT_DIR / "hyperparameters.json"
MODEL_WEIGHTS_PATH = CURRENT_DIR / "bus_model_trained.pth"


class Predictor:
    # 1. PARAMETRI DINAMICI NEL COSTRUTTORE
    def __init__(self, weights_path: str, config_path: str):
        """
        Il costruttore viene chiamato UNA SOLA VOLTA quando si avvia il server.
        """
        print(f"Creating predictor with brain: {os.path.basename(weights_path)}...")

        self.route_encoder = joblib.load(ROUTE_ENCODER_PATH)["route"]

        with open(H3_ENCODER_PATH, "r") as f:
            self.h3_encoder = json.load(f)

        self.static_map = pd.read_parquet(STATIC_MAP_PATH)

        # Usiamo i parametri passati al costruttore!
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model = BusLSTM(
            n_x1_dense_features=self.config["x1_dense_features"],
            n_x2_dense_features=self.config["x2_dense_features"],
            x1_cat_cardinalities=self.config["x1_cat_cards"],
            x2_cat_cardinalities=self.config["x2_cat_cards"],
            encoder_hidden_size=self.config["encoder_hidden_size"],
            lstm_hidden_size=self.config["decoder_hidden_size"],
            num_lstm_layers=self.config["num_lstm_layers"],
        )

        self.model.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

        print(
            f"Ready. Neurons: {self.config['encoder_hidden_size']}. Waiting for input..."
        )

    def _predict_tensors(
        self,
        x1_cat_raw: list,
        x1_dense_raw: list,
        x2_cat_raw: np.ndarray,
        x2_dense_raw: np.ndarray,
    ):
        # A. PREPROCESSING & SCALING
        try:
            x1_cat_raw[0] = self.route_encoder.transform([x1_cat_raw[0]])[0]
        except ValueError:
            x1_cat_raw[0] = 0

        x1_dense_raw[13] = np.clip(x1_dense_raw[13], 1, 3) / 3.0

        x2_dense_raw[:, 0] = np.clip(x2_dense_raw[:, 0], 0, 15000) / 15000.0
        x2_dense_raw[:, 1] = np.clip(x2_dense_raw[:, 1], 0, 1000) / 1000.0
        x2_dense_raw[:, 2] = x2_dense_raw[:, 2] / 99.0
        # Scaliamo le due nuove feature della velocità (Colonna 4 e 5)
        x2_dense_raw[:, 4] = np.clip(x2_dense_raw[:, 4], 0, 65.0) / 65.0
        # Lo speed_ratio (Colonna 5) è già normalizzato tra 0 e ~1, lo clippiamo per sicurezza
        x2_dense_raw[:, 5] = np.clip(x2_dense_raw[:, 5], 0.0, 2.0)

        # B. TENSORIZZAZIONE (Attenzione allo scoping corretto)
        x1_cat_tensor = torch.tensor(x1_cat_raw, dtype=torch.int64).unsqueeze(0)
        x1_dense_tensor = torch.tensor(x1_dense_raw, dtype=torch.float32).unsqueeze(0)
        x2_cat_tensor = torch.tensor(x2_cat_raw, dtype=torch.int64).unsqueeze(0)
        x2_dense_tensor = torch.tensor(x2_dense_raw, dtype=torch.float32).unsqueeze(0)

        # C. INFERENZA
        with torch.no_grad():
            pred_time_scaled, pred_crowd_logits = self.model(
                x1_cat_tensor, x1_dense_tensor, x2_cat_tensor, x2_dense_tensor
            )

        # D. TRADUZIONE
        real_time_delay_seconds = pred_time_scaled.squeeze().numpy() * 3600.0
        pred_crowd_classes = pred_crowd_logits.argmax(dim=-1).squeeze().numpy()

        return real_time_delay_seconds, pred_crowd_classes

    def get_trip_forecast(
        self,
        route_id: str,
        direction_id: int,
        time_seconds: float,
        day_type: int,
        weather_code: int,
        bus_type: int,
    ):
        trip_static = self.static_map[
            (self.static_map["route_id"] == route_id)
            & (self.static_map["direction_id"] == direction_id)
        ]

        if trip_static.empty or len(trip_static) != 100:
            raise ValueError(
                f"Mappa canonica non trovata o corrotta per la linea {route_id} (Dir: {direction_id})"
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

        # 2. INIEZIONE DELLE NUOVE FEATURE MANCANTI
        # Assumiamo condizioni ottimali di default: traffico scorrevole a 25 km/h, ratio 1.0
        base_speed = np.full(100, 25.0)
        speed_ratio = np.full(100, 1.0)

        x2_dense = np.column_stack(
            (shape_dist, dist_to_next, seg_idx, is_gen, base_speed, speed_ratio)
        )

        ritardi, folla = self._predict_tensors(x1_cat, x1_dense, x2_cat, x2_dense)

        return ritardi, folla, trip_static


# ==========================================
# TEST SUL BANCO INTERATTIVO
# ==========================================
if __name__ == "__main__":
    import glob

    print("\n--- SIMULAZIONE SERVER BACKEND ---")

    modelli_disponibili = glob.glob(os.path.join(CURRENT_DIR, "bus_model_*.pth"))

    if not modelli_disponibili:
        print("[ERRORE] Nessun modello trovato.")
        exit()

    print("\nModelli disponibili nel database:")
    for idx, path_modello in enumerate(modelli_disponibili):
        print(f"[{idx}] - {os.path.basename(path_modello)}")

    try:
        scelta = int(input("\nInserisci il numero del modello da caricare: "))
        pesi_scelti = modelli_disponibili[scelta]
        json_scelto = pesi_scelti.replace("bus_model_", "hyperparameters_").replace(
            ".pth", ".json"
        )

        # Inizializziamo passandogli la tua "cartuccia" d'oro!
        oracle = Predictor(weights_path=pesi_scelti, config_path=json_scelto)

        LINEA = "170"
        DIREZIONE = 0
        ORA_SECONDI = 30600

        print(
            f"\n[Richiesta API] Linea {LINEA} (Dir {DIREZIONE}) alle ore {ORA_SECONDI // 3600:02d}:{(ORA_SECONDI % 3600) // 60:02d}..."
        )

        ritardi, folla, mappa = oracle.get_trip_forecast(
            route_id=LINEA,
            direction_id=DIREZIONE,
            time_seconds=ORA_SECONDI,
            day_type=1,
            weather_code=2,
            bus_type=1,
        )

        print("\n[RISPOSTA DELL'ORACOLO]")
        for checkpoint in [10, 50, 99]:
            ritardo_minuti = ritardi[checkpoint] / 60.0
            distanza = mappa["can_shape_dist_travelled"].iloc[checkpoint]
            affollamento = folla[checkpoint]

            print(f"Al metro {distanza:.0f} (Segmento {checkpoint}):")
            print(
                f"  -> Ritardo: {ritardo_minuti:+.1f} min | Affollamento (0-6): {affollamento}"
            )

    except Exception as e:
        print(f"\n[ERRORE FATALE]: {e}")
