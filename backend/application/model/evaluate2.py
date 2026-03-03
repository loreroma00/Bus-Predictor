import os
import json
import torch
import numpy as np
import joblib
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict

from dataset import BusDataset
from model import BusLSTM, BusODELSTM

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# --- CONFIGURAZIONE PERCORSI FISSI ---
CURRENT_DIR = Path(__file__).resolve().parents[0]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = PROJECT_ROOT / "parquets"
TEST_FILE_PATH = PARQUET_DIR / "dataset_lstm_final_TEST.parquet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Caricamento del decoder per le linee


def evaluate_model(weights_path: Path, config_path: Path):
    print(f"\n--- AVVIO BANCO DI PROVA SU {DEVICE} ---")

    def _decode_time_from_sin_cos(sin_val, cos_val):
        angle = np.arctan2(float(sin_val), float(cos_val))
        if angle < 0:
            angle += 2 * np.pi
        minutes = int(round((angle / (2 * np.pi)) * 24 * 60)) % (24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    if not TEST_FILE_PATH.exists():
        print(f"[ERRORE] File non trovato: {TEST_FILE_PATH}")
        return

    # Caricamento del decoder per le linee
    encoder_path = PARQUET_DIR / "route_encoder.pkl"  # Adatta il percorso se serve
    try:
        route_decoder_obj = joblib.load(encoder_path)
        route_decoder = (
            route_decoder_obj.get("route")
            if isinstance(route_decoder_obj, dict)
            else route_decoder_obj
        )
        print("[*] Decoder linee caricato correttamente.")
    except Exception:
        route_decoder = None
        print("[!] Decoder non trovato, userò gli ID numerici.")

    # 1. CARICAMENTO DATASET (Shuffle=False per confronto coerente)
    test_dataset = BusDataset(str(TEST_FILE_PATH))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. CARICAMENTO MODELLO
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    arch = config.get("architecture", "lstm")
    ModelClass = BusODELSTM if arch == "ode_lstm" else BusLSTM

    model = ModelClass(
        n_x1_dense_features=config["x1_dense_features"],
        n_x2_dense_features=config["x2_dense_features"],
        x1_cat_cardinalities=config["x1_cat_cards"],
        x2_cat_cardinalities=config["x2_cat_cards"],
        encoder_hidden_size=config["encoder_hidden_size"],
        lstm_hidden_size=config["decoder_hidden_size"],
        num_lstm_layers=config.get("num_lstm_layers", 2) if arch == "lstm" else None,
    ).to(DEVICE)

    state_dict = torch.load(weights_path, map_location=DEVICE)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 3. METRICHE
    total_time_mae_seconds = 0.0
    total_time_mse_seconds = 0.0
    total_crowd_correct = 0
    total_samples = 0
    all_time_errors = []
    all_true_crowd, all_pred_crowd = [], []
    fotoromanzo_reale, fotoromanzo_predetto = None, None

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x1_cat, x1_dense, x2_cat, x2_dense, y_time, y_crowd = [
                b.to(DEVICE) for b in batch
            ]
            pred_time, pred_crowd_logits = model(x1_cat, x1_dense, x2_cat, x2_dense)

            # Conversione in secondi
            pred_time_sec = pred_time.squeeze(-1) * 3600.0
            true_time_sec = y_time.squeeze(-1) * 3600.0

            # Accumulo Errori
            err_batch = (pred_time_sec - true_time_sec).cpu().numpy().flatten()
            all_time_errors.extend(err_batch)
            total_time_mae_seconds += np.sum(np.abs(err_batch))
            total_time_mse_seconds += np.sum(err_batch**2)

            # Folla
            pred_crowd_classes = pred_crowd_logits.argmax(dim=-1)
            total_crowd_correct += (pred_crowd_classes == y_crowd).sum().item()
            all_true_crowd.append(y_crowd.cpu().numpy().flatten())
            all_pred_crowd.append(pred_crowd_classes.cpu().numpy().flatten())
            total_samples += y_time.numel()

            # --- ESTRAZIONE FOTOROMANZO (Soggetto Unico) ---
            if i == 0:
                route_name, s_time = "N/D", "N/D"
                try:
                    df_meta = pd.read_parquet(TEST_FILE_PATH).head(1)
                    row = df_meta.iloc[0]

                    route_col = (
                        "route_id"
                        if "route_id" in df_meta.columns
                        else "route_id - i64"
                    )
                    if route_col in df_meta.columns:
                        raw_id = int(row[route_col])
                        if route_decoder is not None:
                            route_name = route_decoder.inverse_transform([raw_id])[0]
                        else:
                            route_name = f"ID {raw_id}"

                    if "start_time" in df_meta.columns and pd.notna(row["start_time"]):
                        s_time = str(row["start_time"])
                    elif "scheduled_start_time" in df_meta.columns and pd.notna(
                        row["scheduled_start_time"]
                    ):
                        s_time = str(row["scheduled_start_time"])
                    elif (
                        "scheduled_start_time_sin" in df_meta.columns
                        and "scheduled_start_time_cos" in df_meta.columns
                        and pd.notna(row["scheduled_start_time_sin"])
                        and pd.notna(row["scheduled_start_time_cos"])
                    ):
                        s_time = _decode_time_from_sin_cos(
                            row["scheduled_start_time_sin"],
                            row["scheduled_start_time_cos"],
                        )
                except Exception as e:
                    print(f"[!] Impossibile leggere metadati fotoromanzo: {e}")

                fotoromanzo_reale = true_time_sec[0].cpu().numpy()
                fotoromanzo_predetto = pred_time_sec[0].cpu().numpy()

                print(f"\n[ISPEZIONE VISIVA] Linea: {route_name} | Partenza: {s_time}")
    # 4. REPORT FINALE
    avg_mae = total_time_mae_seconds / total_samples
    rmse = (total_time_mse_seconds / total_samples) ** 0.5
    errors_np = np.array(all_time_errors)

    print("\n" + "=" * 50)
    print("RISULTATI FINALI")
    print("=" * 50)
    print(f"MAE Tempo: {avg_mae / 60:.2f} min | RMSE: {rmse / 60:.2f} min")
    print(
        f"Deviazione Standard: {np.std(errors_np):.1f}s | Bias: {np.mean(errors_np):.1f}s"
    )
    print(f"Accuratezza Folla: {(total_crowd_correct / total_samples) * 100:.1f}%")
    print("=" * 50)

    if HAS_PLOTTING:
        nome = weights_path.stem
        plot_campana_errori(errors_np, CURRENT_DIR / f"campana_{nome}.png")
        plot_fotoromanzo_viaggio(
            np.arange(100),
            fotoromanzo_reale,
            fotoromanzo_predetto,
            CURRENT_DIR / f"fotoromanzo_{nome}.png",
        )


def evaluate_gbdt_model(pkl_path: Path):
    print(f"\n--- ANALISI GBDT: {pkl_path.name} ---")
    df = pd.read_parquet(TEST_FILE_PATH).iloc[::100]  # Un punto per viaggio

    X = df[["route", "time_sin", "time_cos", "day_type"]].copy()
    for col in ["route", "day_type"]:
        X[col] = X[col].astype("category")

    # FIX SCALING: Riportiamo a 0-9
    y_true = (df["bus_type"] * 9.0).round().astype(int).values

    model = joblib.load(pkl_path)
    y_pred = model.predict(X)

    print(f"Accuratezza Tipo Bus: {(y_pred == y_true).mean() * 100:.2f}%")
    # Qui il codice della matrice (omesso per brevità, ma usa y_true corretto)


# [INSERIRE QUI LE FUNZIONI plot_campana_errori E plot_fotoromanzo_viaggio DI PRIMA]

if __name__ == "__main__":
    print("\n--- LABORATORIO DI COLLAUDO ---")

    lstm_models = list(CURRENT_DIR.glob("bus_model_*.pth"))
    gbdt_models = list(CURRENT_DIR.glob("*.pkl"))

    all_models = []
    for p in lstm_models:
        all_models.append(("LSTM", p))
    for p in gbdt_models:
        all_models.append(("GBDT", p))

    if not all_models:
        print(
            "[ERRORE] Nessun modello trovato in questa cartella. Esegui prima train.py!"
        )
        exit()

    print("\nModelli disponibili:")
    for idx, (model_type, path_modello) in enumerate(all_models):
        print(f"[{idx}] [{model_type}] - {path_modello.name}")

    try:
        scelta = int(input("\nInserisci il numero: "))
        model_type, chosen_path = all_models[scelta]

        if model_type == "LSTM":
            nome_json = chosen_path.name.replace(
                "bus_model_", "hyperparameters_"
            ).replace(".pth", ".json")
            json_path = CURRENT_DIR / nome_json

            if not json_path.exists():
                print(
                    f"\n[ERRORE] Impossibile trovare il file di configurazione associato: {nome_json}"
                )
                exit()

            evaluate_model(weights_path=chosen_path, config_path=json_path)
        else:
            evaluate_gbdt_model(pkl_path=chosen_path)

    except (IndexError, ValueError):
        print("\n[ERRORE] Selezione non valida.")
    except KeyboardInterrupt:
        print("\n[!] Collaudo interrotto.")
