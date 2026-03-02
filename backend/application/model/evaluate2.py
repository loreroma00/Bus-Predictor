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

# Il file con i dati nuovi (non visti in addestramento)
TEST_FILE_PATH = PARQUET_DIR / "dataset_lstm_final_TEST.parquet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(weights_path: Path, config_path: Path):
    print(f"\n--- AVVIO BANCO DI PROVA SU {DEVICE} ---")
    print(f"[*] Analisi in corso sul cervello: {weights_path.name}")

    # 1. Caricamento Dataset di Test
    if not TEST_FILE_PATH.exists():
        print(f"\n[ERRORE FATALE] File di test non trovato: {TEST_FILE_PATH}")
        print(
            "Devi creare un nuovo parquet con i dati freschi e chiamarlo 'dataset_lstm_final_TEST.parquet'!"
        )
        return

    print("Caricamento nuovi dati spaziali in corso...")
    test_dataset = BusDataset(str(TEST_FILE_PATH))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 2. Caricamento Architettura Dinamica
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    arch = config.get("architecture", "lstm")
    ModelClass = BusODELSTM if arch == "ode_lstm" else BusLSTM

    model_kwargs = {
        "n_x1_dense_features": config["x1_dense_features"],
        "n_x2_dense_features": config["x2_dense_features"],
        "x1_cat_cardinalities": config["x1_cat_cards"],
        "x2_cat_cardinalities": config["x2_cat_cards"],
        "encoder_hidden_size": config["encoder_hidden_size"],
        "lstm_hidden_size": config["decoder_hidden_size"],
    }

    if arch == "lstm":
        model_kwargs["num_lstm_layers"] = config.get("num_lstm_layers", 2)

    model = ModelClass(**model_kwargs).to(DEVICE)

    # Caricamento dei pesi specifici
    state_dict = torch.load(weights_path, map_location=DEVICE)

    # Handle torch.compile() wrapper - strip _orig_mod. prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print(
        f"Modello armato e pronto. Loss originale: {config.get('loss_type', 'N/A').upper()}"
    )

    # 3. Metriche Umane
    total_time_mae_seconds = 0.0
    total_time_mse_seconds = 0.0
    total_crowd_correct = 0
    total_samples = 0

    # Variabili per la matrice di confusione
    all_true_crowd = []
    all_pred_crowd = []
    
    # --- VARIABILI PER STORYTELLING GRAFICO ---
    all_time_errors = []
    fotoromanzo_reale = None
    fotoromanzo_predetto = None

    print("\nElaborazione dei viaggi inediti...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x1_cat, x1_dense, x2_cat, x2_dense, y_time, y_crowd = [
                b.to(DEVICE) for b in batch
            ]

            # Predizione
            pred_time, pred_crowd_logits = model(x1_cat, x1_dense, x2_cat, x2_dense)

            # --- CALCOLO ERRORI TEMPO ---
            pred_time_sec = pred_time.squeeze(-1) * 3600.0
            true_time_sec = y_time.squeeze(-1) * 3600.0
 
            # MAE e MSE
            mae = torch.abs(pred_time_sec - true_time_sec).sum().item()
            mse = ((pred_time_sec - true_time_sec) ** 2).sum().item()

            total_time_mae_seconds += mae
            total_time_mse_seconds += mse
            
            # --- CATTURA DATI PER LA CAMPANA ERRORI ---
            errori_batch = (pred_time_sec - true_time_sec).cpu().numpy().flatten()
            all_time_errors.extend(errori_batch)

            # --- CALCOLO ERRORI FOLLA ---
            pred_crowd_classes = pred_crowd_logits.argmax(dim=-1)
            correct_predictions = (pred_crowd_classes == y_crowd).sum().item()
            total_crowd_correct += correct_predictions

            # Salva per la matrice di confusione
            all_true_crowd.append(y_crowd.cpu().numpy().flatten())
            all_pred_crowd.append(pred_crowd_classes.cpu().numpy().flatten())

            total_samples += y_time.numel()

            # --- ESTRAZIONE VISIVA (Dal primo batch) ---
            if i == 0:
                # Salviamo la prima traiettoria per il fotoromanzo
                fotoromanzo_reale = true_time_sec[0].cpu().numpy()
                fotoromanzo_predetto = pred_time_sec[0].cpu().numpy()
                
                print("\n[ISPEZIONE VISIVA A CAMPIONE - Viaggio #0]")
                print(
                    f"{'Segmento':<10} | {'Reale (sec)':<15} | {'Previsto (sec)':<15} | {'Errore (sec)':<15} || {'Folla Reale':<15} | {'Folla Prevista':<15}"
                )
                print("-" * 95)
                for seg in [0, 25, 50, 75, 99]:
                    r_time = true_time_sec[0, seg].item()
                    p_time = pred_time_sec[0, seg].item()
                    err_time = abs(r_time - p_time)
                    r_crowd = y_crowd[0, seg].item()
                    p_crowd = pred_crowd_classes[0, seg].item()

                    print(
                        f"Seg {seg:02d}    | {r_time:>10.0f} s    | {p_time:>10.0f} s    | {err_time:>10.0f} s    || Classe {r_crowd:<8} | Classe {p_crowd:<8}"
                    )

    # 4. Statistiche Finali
    avg_mae = total_time_mae_seconds / total_samples
    rmse = (total_time_mse_seconds / total_samples) ** 0.5
    accuracy_crowd = (total_crowd_correct / total_samples) * 100

    # --- CALCOLO VARIANZA E DEVIAZIONE STANDARD ---
    all_time_errors_np = np.array(all_time_errors)
    variance_time = np.var(all_time_errors_np)
    std_dev_time = np.std(all_time_errors_np)
    mean_bias = np.mean(all_time_errors_np) # Questo ti dice se il modello tende a sovrastimare o sottostimare
    
    print("\n" + "=" * 50)
    print("RISULTATI SUL TEST SET INEDITO")
    print("=" * 50)
    print(f"Viaggi elaborati : {total_samples // 100}")
    print(f"Punti spaziali   : {total_samples}")
    print("-" * 50)
    print(
        f"Errore Medio (MAE) Tempo  : {avg_mae:.1f} secondi (+/- {avg_mae / 60:.1f} minuti)"
    )
    print(f"Errore Estremo (RMSE)     : {rmse:.1f} secondi")
    print(f"Accuratezza Folla         : {accuracy_crowd:.1f}%")
    print(f"Varianza Errore Tempo    : {variance_time:.1f} s^2")
    print(f"Deviazione Standard (sigma): {std_dev_time:.1f} secondi (+/- {std_dev_time / 60:.1f} minuti)")
    print(f"Bias Medio (Errore Medio) : {mean_bias:.1f} secondi")
    print("=" * 50)

    # 5. Matrice di Confusione per Occupancy Status
    print("\n--- MATRICE DI CONFUSIONE OCCUPANCY STATUS ---")

    all_true_crowd = np.concatenate(all_true_crowd)
    all_pred_crowd = np.concatenate(all_pred_crowd)

    # Calcola la matrice di confusione
    num_classes = 8
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(all_true_crowd, all_pred_crowd):
        cm[int(true), int(pred)] += 1

    # Stampa testo
    print("\nMatrice di Confusione (righe=reale, colonne=predetto):\n")
    header = "     " + "  ".join([f"{i:>5}" for i in range(num_classes)])
    print(header)
    print("     " + "-" * (6 * num_classes))
    for i in range(num_classes):
        row_str = f"{i:>3} |" + "  ".join(
            [f"{cm[i, j]:>5}" for j in range(num_classes)]
        )
        print(row_str)

    # Statistiche per classe
    print("\nStatistiche per classe:")
    print(
        f"{'Classe':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}"
    )
    print("-" * 55)

    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = cm[cls, :].sum()

        print(
            f"{cls:<8} | {precision:<10.3f} | {recall:<10.3f} | {f1:<10.3f} | {support:<10}"
        )

    # Macro average
    precisions = []
    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp

        if (tp + fp) > 0:
            precisions.append(tp / (tp + fp))
        if (tp + fn) > 0:
            recalls.append(tp / (tp + fn))
        if (tp + fp) > 0 and (tp + fn) > 0:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            if (p + r) > 0:
                f1s.append(2 * p * r / (p + r))

    print("-" * 55)
    print(
        f"{'Macro Avg':<8} | {np.mean(precisions):<10.3f} | {np.mean(recalls):<10.3f} | {np.mean(f1s):<10.3f} | {total_samples:<10}"
    )

    # Salva come immagine
    if HAS_PLOTTING:
        save_confusion_matrix_image(cm, num_classes, weights_path.stem)
        
        # --- GENERAZIONE NUOVI GRAFICI NARRATIVI ---
        print("\n--- GENERAZIONE GRAFICI DI STORYTELLING ---")
        nome_modello = weights_path.stem
        plot_campana_errori(np.array(all_time_errors), CURRENT_DIR / f"campana_errori_{nome_modello}.png")
        plot_fotoromanzo_viaggio(np.arange(100), fotoromanzo_reale, fotoromanzo_predetto, CURRENT_DIR / f"fotoromanzo_{nome_modello}.png")
    else:
        print("\n[INFO] matplotlib non disponibile - immagini non salvate")


def evaluate_gbdt_model(pkl_path: Path):
    print(f"\n--- AVVIO VALUTAZIONE GBDT ---")
    print(f"[*] Modello: {pkl_path.name}")

    if not TEST_FILE_PATH.exists():
        print(f"\n[ERRORE FATALE] File di test non trovato: {TEST_FILE_PATH}")
        return

    print("Caricamento dati di test...")
    df = pd.read_parquet(str(TEST_FILE_PATH))

    df = df.dropna(subset=["bus_type"])

    features = ["route_id", "time_sin", "time_cos", "day_type"]
    target = "bus_type"

    missing = [f for f in features + [target] if f not in df.columns]
    if missing:
        print(f"\n[ERRORE] Colonne mancanti nel dataset: {missing}")
        return

    df_trips = df.iloc[::100].copy()
    print(f"Trip unici estratti: {len(df_trips)}")

    X = df_trips[features].copy()
    y_true = (df_trips[target] * 9.0).round().astype(int).values

    categorical_features = ["route_id", "day_type"]
    for col in categorical_features:
        X[col] = X[col].astype("category")

    print("Caricamento modello GBDT...")
    model = joblib.load(pkl_path)

    print("Predizione in corso...")
    y_pred = model.predict(X)

    accuracy = (y_pred == y_true).mean()
    print(f"\n{'=' * 50}")
    print("RISULTATI GBDT SU TEST SET")
    print(f"{'=' * 50}")
    print(f"Campioni testati     : {len(y_true)}")
    print(f"Accuratezza Globale  : {accuracy * 100:.2f}%")
    print(f"{'=' * 50}")

    num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1

    print("\n--- MATRICE DI CONFUSIONE BUS TYPE ---")
    print("\nMatrice di Confusione (righe=reale, colonne=predetto):\n")
    header = "     " + "  ".join([f"{i:>5}" for i in range(num_classes)])
    print(header)
    print("     " + "-" * (6 * num_classes))
    for i in range(num_classes):
        row_str = f"{i:>3} |" + "  ".join(
            [f"{cm[i, j]:>5}" for j in range(num_classes)]
        )
        print(row_str)

    print("\nStatistiche per classe:")
    print(
        f"{'Classe':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}"
    )
    print("-" * 55)

    precisions = []
    recalls = []
    f1s = []

    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = cm[cls, :].sum()

        if (tp + fp) > 0:
            precisions.append(precision)
        if (tp + fn) > 0:
            recalls.append(recall)
        if (tp + fp) > 0 and (tp + fn) > 0 and (precision + recall) > 0:
            f1s.append(f1)

        print(
            f"{cls:<8} | {precision:<10.3f} | {recall:<10.3f} | {f1:<10.3f} | {support:<10}"
        )

    print("-" * 55)
    print(
        f"{'Macro Avg':<8} | {np.mean(precisions):<10.3f} | {np.mean(recalls):<10.3f} | {np.mean(f1s):<10.3f} | {len(y_true):<10}"
    )

    if HAS_PLOTTING:
        save_confusion_matrix_image(cm, num_classes, pkl_path.stem)
    else:
        print("\n[INFO] matplotlib non disponibile - immagine non salvata")


# =====================================================================
# MOTORI DI RENDERING GRAFICO
# =====================================================================

def save_confusion_matrix_image(cm: np.ndarray, num_classes: int, model_name: str):
    """Salva la matrice di confusione come immagine PNG."""
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        cbar_kws={"label": "Count"},
    )
    axes[0].set_xlabel("Predicted Class", fontsize=12)
    axes[0].set_ylabel("True Class", fontsize=12)
    axes[0].set_title(f"Confusion Matrix (Counts)\n{model_name}", fontsize=14)

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=axes[1],
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        cbar_kws={"label": "Proportion"},
    )
    axes[1].set_xlabel("Predicted Class", fontsize=12)
    axes[1].set_ylabel("True Class", fontsize=12)
    axes[1].set_title(f"Confusion Matrix (Normalized)\n{model_name}", fontsize=14)

    plt.tight_layout()
    output_path = CURRENT_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Matrice di confusione salvata: {output_path}")

def plot_campana_errori(errori_secondi, save_path):
    from scipy.stats import gaussian_kde
    print("[*] Generazione grafico a campana in corso...")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    limite = np.percentile(np.abs(errori_secondi), 95)
    errori_filtrati = errori_secondi[(errori_secondi >= -limite) & (errori_secondi <= limite)]
    
    ax.hist(errori_filtrati, bins=60, density=True, alpha=0.4, color='#3498db', edgecolor='white')
    
    kde = gaussian_kde(errori_filtrati)
    x_vals = np.linspace(min(errori_filtrati), max(errori_filtrati), 500)
    ax.plot(x_vals, kde(x_vals), color='#2980b9', linewidth=2.5, label='Densità di Probabilità')
    
    ax.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=2, zorder=5, label='Perfezione (Errore = 0s)')
    
    ax.set_title('Densità dell\'Errore di Predizione sul Ritardo', fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('Errore di Predizione (Secondi)', fontsize=12, weight='bold')
    ax.set_ylabel('Frequenza (Densità)', fontsize=12, weight='bold')
    
    mae_reale = np.mean(np.abs(errori_secondi))
    ax.annotate(f'MAE Globale: {mae_reale/60:.1f} min\nLa campana stretta dimostra\nche il grosso degli errori\nè concentrato vicino allo zero.',
                xy=(0.02, 0.8), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bdc3c7", lw=1),
                fontsize=10, color='#2c3e50')

    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Campana salvata in: {save_path.name}")

def plot_fotoromanzo_viaggio(segmenti, ritardo_reale, ritardo_predetto, save_path):
    print("[*] Generazione fotoromanzo spaziale in corso...")
    
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    ax.plot(segmenti, ritardo_reale, color='#7f8c8d', linestyle='--', linewidth=2.5, alpha=0.8, label='Ritardo Reale (Ground Truth)')
    ax.plot(segmenti, ritardo_predetto, color='#2c3e50', linestyle='-', linewidth=3, label='Predizione Modello')
    ax.fill_between(segmenti, ritardo_reale, ritardo_predetto, color='#e74c3c', alpha=0.15, label='Scarto (Errore)')
    
    ax.set_title('Inseguimento della Traiettoria: Analisi di un Singolo Viaggio', fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Avanzamento Corsa (Indice del Segmento Spaziale 0-100)', fontsize=12, weight='bold')
    ax.set_ylabel('Ritardo Cumulato (Secondi)', fontsize=12, weight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True)
    ax.set_xlim(0, max(segmenti))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Fotoromanzo salvato in: {save_path.name}")

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
