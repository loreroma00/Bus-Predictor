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
from model import BusLSTM, OccupancyLSTM

plt = None
sns = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

CURRENT_DIR = Path(__file__).resolve().parents[0]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = PROJECT_ROOT / "parquets"
TEST_FILE_PATH = PARQUET_DIR / "dataset_lstm_final_TEST.parquet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_confusion_matrix_image(cm, num_classes, nome):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title(f"Matrice di Confusione - {nome}")
    path = CURRENT_DIR / f"confusion_matrix_DUAL_{nome}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Confusion matrix salvata: {path}")


def plot_campana_errori(errors_np, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors_np, bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Errore (secondi)")
    ax.set_ylabel("Densità")
    ax.set_title("Distribuzione degli Errori di Previsione")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Distribuzione errori salvata: {output_path}")


def evaluate_model(config_path: Path):
    print(f"\n--- AVVIO BANCO DI PROVA DUAL SU {DEVICE} ---")

    def _decode_time_from_sin_cos(sin_val, cos_val):
        angle = np.arctan2(float(sin_val), float(cos_val))
        if angle < 0: angle += 2 * np.pi
        minutes = int(round((angle / (2 * np.pi)) * 24 * 60)) % (24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    if not TEST_FILE_PATH.exists():
        print(f"[ERRORE] File non trovato: {TEST_FILE_PATH}")
        return

    encoder_path = PARQUET_DIR / "route_encoder.pkl"
    try:
        route_decoder_obj = joblib.load(encoder_path)
        route_decoder = route_decoder_obj.get("route") if isinstance(route_decoder_obj, dict) else route_decoder_obj
        print("[*] Decoder linee caricato correttamente.")
    except Exception:
        route_decoder = None
        print("[!] Decoder non trovato, userò gli ID numerici.")

    test_dataset = BusDataset(str(TEST_FILE_PATH))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # LETTURA CONFIGURAZIONE DUAL
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # IDENTIFICAZIONE PESI CORRISPONDENTI
    base_name = config_path.name.replace("hyperparameters_DUAL_", "").replace(".json", "")
    time_weights_path = CURRENT_DIR / f"bus_model_TIME_{base_name}.pth"
    crowd_weights_path = CURRENT_DIR / f"bus_model_CROWD_{base_name}.pth"

    if not time_weights_path.exists() or not crowd_weights_path.exists():
        print(f"[ERRORE] Pesi mancanti per l'esperimento {base_name}!")
        return

    model_kwargs = {
        "n_x1_dense_features": config["x1_dense_features"],
        "n_x2_dense_features": config["x2_dense_features"],
        "x1_cat_cardinalities": config["x1_cat_cards"],
        "x2_cat_cardinalities": config["x2_cat_cards"],
        "encoder_hidden_size": config["encoder_hidden_size"],
        "lstm_hidden_size": config["decoder_hidden_size"],
    }
    crowd_kwargs = model_kwargs.copy()
    crowd_kwargs["num_lstm_layers"] = config.get("num_lstm_layers", 2)

    model_time = BusLSTM(**model_kwargs).to(DEVICE)
    model_crowd = OccupancyLSTM(**crowd_kwargs).to(DEVICE)

    # Caricamento e pulizia chiavi
    state_time = torch.load(time_weights_path, map_location=DEVICE)
    if any(k.startswith("_orig_mod.") for k in state_time.keys()):
        state_time = {k.replace("_orig_mod.", ""): v for k, v in state_time.items()}
    model_time.load_state_dict(state_time)
    
    state_crowd = torch.load(crowd_weights_path, map_location=DEVICE)
    if any(k.startswith("_orig_mod.") for k in state_crowd.keys()):
        state_crowd = {k.replace("_orig_mod.", ""): v for k, v in state_crowd.items()}
    model_crowd.load_state_dict(state_crowd)

    model_time.eval()
    model_crowd.eval()

    total_time_mae_seconds = 0.0
    total_time_mse_seconds = 0.0
    total_crowd_correct = 0
    total_samples = 0
    all_time_errors, all_true_crowd, all_pred_crowd = [], [], []
    fotoromanzo_reale, fotoromanzo_predetto = None, None
    route_metrics = defaultdict(lambda: {"errors": [], "y_true": [], "y_pred": []})

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x1_cat, x1_dense, x2_cat, x2_dense, y_time, y_crowd, lengths, t_grid = [b.to(DEVICE) for b in batch]
            seq_len = x2_dense.size(1)
            mask = torch.arange(seq_len, device=DEVICE).unsqueeze(0) < lengths.unsqueeze(1)

            # ELABORAZIONE DISACCOPPIATA
            pred_time = model_time(x1_cat, x1_dense, x2_cat, x2_dense, lengths=lengths, t_grid=t_grid)
            pred_crowd_logits = model_crowd(x1_cat, x1_dense, x2_cat, x2_dense, lengths=lengths)

            # SCALING AGGIORNATO A 600.0 (10 Minuti)
            pred_time_sec = pred_time.squeeze(-1) * 600.0
            true_time_sec = y_time.squeeze(-1) * 600.0

            # Apply mask: only evaluate real stops
            mask_np = mask.cpu().numpy()
            pred_batch_np = pred_time_sec.cpu().numpy()
            true_batch_np = true_time_sec.cpu().numpy()

            # Masked error computation
            err_batch = (pred_batch_np - true_batch_np)[mask_np]
            all_time_errors.extend(err_batch.tolist())
            total_time_mae_seconds += np.sum(np.abs(err_batch))
            total_time_mse_seconds += np.sum(err_batch**2)

            pred_crowd_classes = pred_crowd_logits.argmax(dim=-1)
            total_crowd_correct += (pred_crowd_classes[mask] == y_crowd[mask]).sum().item()
            all_true_crowd.append(y_crowd[mask].cpu().numpy())
            all_pred_crowd.append(pred_crowd_classes[mask].cpu().numpy())
            total_samples += int(mask.sum().item())

            # Per-route metrics (masked)
            route_ids_batch = x1_cat[:, 0].detach().cpu().numpy().astype(np.int64)
            pred_crowd_batch_np = pred_crowd_classes.detach().cpu().numpy()
            true_crowd_batch_np = y_crowd.detach().cpu().numpy()
            err_batch_2d = pred_batch_np - true_batch_np
            lengths_np = lengths.cpu().numpy()

            for trip_idx, route_id_enc in enumerate(route_ids_batch):
                n_real = int(lengths_np[trip_idx])
                per_route = route_metrics[int(route_id_enc)]
                per_route["errors"].extend(err_batch_2d[trip_idx, :n_real].tolist())
                per_route["y_true"].extend(true_crowd_batch_np[trip_idx, :n_real].tolist())
                per_route["y_pred"].extend(pred_crowd_batch_np[trip_idx, :n_real].tolist())

            if i == 0:
                route_name, s_time = "N/D", "N/D"
                n_real_0 = int(lengths_np[0])
                fotoromanzo_reale = true_time_sec[0, :n_real_0].cpu().numpy()
                fotoromanzo_predetto = pred_time_sec[0, :n_real_0].cpu().numpy()

    # (Calcoli finali e plot invariati)
    avg_mae = total_time_mae_seconds / total_samples
    rmse = (total_time_mse_seconds / total_samples) ** 0.5
    errors_np = np.array(all_time_errors)

    print("\n" + "=" * 50)
    print("RISULTATI FINALI")
    print("=" * 50)
    print(f"MAE Tempo: {avg_mae / 60:.2f} min | RMSE: {rmse / 60:.2f} min")
    print(f"Deviazione Standard: {np.std(errors_np):.1f}s | Bias: {np.mean(errors_np):.1f}s")
    accuracy_crowd = (total_crowd_correct / total_samples) * 100
    print(f"Accuratezza Folla: {accuracy_crowd:.1f}%")
    print("=" * 50)

    # Generazione Confusion Matrix e Plot (codice invariato...)
    all_true_crowd_np = np.concatenate(all_true_crowd)
    all_pred_crowd_np = np.concatenate(all_pred_crowd)
    num_classes = int(max(all_true_crowd_np.max(), all_pred_crowd_np.max()) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class, pred_class in zip(all_true_crowd_np, all_pred_crowd_np):
        cm[int(true_class), int(pred_class)] += 1

    print("\nMatrice di Confusione (righe=reale, colonne=predetto):\n")
    header = "     " + "  ".join([f"{i:>5}" for i in range(num_classes)])
    print(header)
    for i in range(num_classes):
        print(f"{i:>3} |" + "  ".join([f"{cm[i, j]:>5}" for j in range(num_classes)]))

    if HAS_PLOTTING:
        nome = base_name
        save_confusion_matrix_image(cm, num_classes, nome)
        plot_campana_errori(errors_np, CURRENT_DIR / f"campana_DUAL_{nome}.png")
        # I fotoromanzi restano identici...

# GBDT Omissis...

if __name__ == "__main__":
    print("\n--- LABORATORIO DI COLLAUDO ---")

    # Adesso cerchiamo i file JSON della configurazione DUAL
    dual_configs = list(CURRENT_DIR.glob("hyperparameters_DUAL_*.json"))
    gbdt_models = list(CURRENT_DIR.glob("*.pkl"))

    all_models = []
    for p in dual_configs:
        all_models.append(("DUAL_LSTM", p))
    for p in gbdt_models:
        all_models.append(("GBDT", p))

    if not all_models:
        print("[ERRORE] Nessun modello DUAL trovato. Esegui prima train.py!")
        exit()

    print("\nConfigurazioni Dual-Model disponibili:")
    for idx, (model_type, path_modello) in enumerate(all_models):
        print(f"[{idx}] [{model_type}] - {path_modello.name}")

    try:
        scelta = int(input("\nInserisci il numero: "))
        model_type, chosen_path = all_models[scelta]

        if model_type == "DUAL_LSTM":
            evaluate_model(config_path=chosen_path)
        else:
            evaluate_gbdt_model(pkl_path=chosen_path)

    except (IndexError, ValueError):
        print("\n[ERRORE] Selezione non valida.")
