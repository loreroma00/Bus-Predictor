import os
import json
import torch
import numpy as np
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

    # Per la matrice di confusione
    all_true_crowd = []
    all_pred_crowd = []

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
    else:
        print("\n[INFO] matplotlib non disponibile - immagine non salvata")


def save_confusion_matrix_image(cm: np.ndarray, num_classes: int, model_name: str):
    """Salva la matrice di confusione come immagine PNG."""

    # Normalizza per riga per mostrare le percentuali
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Matrice grezza
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

    # Matrice normalizzata
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

    # Salva
    output_path = CURRENT_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Matrice di confusione salvata: {output_path}")


if __name__ == "__main__":
    print("\n--- LABORATORIO DI COLLAUDO ---")

    # 1. Scansione modelli
    modelli_disponibili = list(CURRENT_DIR.glob("bus_model_*.pth"))

    if not modelli_disponibili:
        print(
            "[ERRORE] Nessun modello trovato in questa cartella. Esegui prima train.py!"
        )
        exit()

    # 2. Menu Interattivo
    print("\nQuale Digital Twin vuoi testare?")
    for idx, path_modello in enumerate(modelli_disponibili):
        print(f"[{idx}] - {path_modello.name}")

    try:
        scelta = int(input("\nInserisci il numero: "))
        pesi_scelti = modelli_disponibili[scelta]

        # 3. Derivazione nome del JSON
        nome_json = pesi_scelti.name.replace("bus_model_", "hyperparameters_").replace(
            ".pth", ".json"
        )
        json_scelto = CURRENT_DIR / nome_json

        if not json_scelto.exists():
            print(
                f"\n[ERRORE] Impossibile trovare il file di configurazione associato: {nome_json}"
            )
            exit()

        # 4. Avvio valutazione
        evaluate_model(weights_path=pesi_scelti, config_path=json_scelto)

    except (IndexError, ValueError):
        print("\n[ERRORE] Selezione non valida.")
    except KeyboardInterrupt:
        print("\n[!] Collaudo interrotto.")
