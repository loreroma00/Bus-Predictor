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
    sum_true_profile = None
    sum_pred_profile = None
    n_trips_profile = 0
    route_metrics = defaultdict(lambda: {"errors": [], "y_true": [], "y_pred": []})

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x1_cat, x1_dense, x2_cat, x2_dense, y_time, y_crowd = [
                b.to(DEVICE) for b in batch
            ]
            pred_time, pred_crowd_logits = model(x1_cat, x1_dense, x2_cat, x2_dense)

            # Conversione in secondi
            pred_time_sec = pred_time.squeeze(-1) * 3600.0
            true_time_sec = y_time.squeeze(-1) * 3600.0

            pred_batch_np = pred_time_sec.cpu().numpy()
            true_batch_np = true_time_sec.cpu().numpy()

            if sum_true_profile is None:
                sum_true_profile = np.zeros(true_batch_np.shape[1], dtype=np.float64)
                sum_pred_profile = np.zeros(pred_batch_np.shape[1], dtype=np.float64)

            sum_true_profile += true_batch_np.sum(axis=0)
            sum_pred_profile += pred_batch_np.sum(axis=0)
            n_trips_profile += true_batch_np.shape[0]

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

            route_ids_batch = x1_cat[:, 0].detach().cpu().numpy().astype(np.int64)
            pred_crowd_batch_np = pred_crowd_classes.detach().cpu().numpy()
            true_crowd_batch_np = y_crowd.detach().cpu().numpy()
            err_batch_2d = pred_batch_np - true_batch_np

            for trip_idx, route_id_enc in enumerate(route_ids_batch):
                per_route = route_metrics[int(route_id_enc)]
                per_route["errors"].extend(err_batch_2d[trip_idx].tolist())
                per_route["y_true"].extend(true_crowd_batch_np[trip_idx].tolist())
                per_route["y_pred"].extend(pred_crowd_batch_np[trip_idx].tolist())

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
    accuracy_crowd = (total_crowd_correct / total_samples) * 100
    print(f"Accuratezza Folla: {accuracy_crowd:.1f}%")
    print("=" * 50)

    all_true_crowd_np = np.concatenate(all_true_crowd)
    all_pred_crowd_np = np.concatenate(all_pred_crowd)
    num_classes = int(max(all_true_crowd_np.max(), all_pred_crowd_np.max()) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class, pred_class in zip(all_true_crowd_np, all_pred_crowd_np):
        cm[int(true_class), int(pred_class)] += 1

    print("\n--- MATRICE DI CONFUSIONE OCCUPANCY STATUS ---")
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
        f"{'Macro Avg':<8} | {np.mean(precisions):<10.3f} | {np.mean(recalls):<10.3f} | {np.mean(f1s):<10.3f} | {len(all_true_crowd_np):<10}"
    )
    save_confusion_matrix_csv(cm, weights_path.stem)

    save_route_metrics_report(route_metrics, route_decoder, weights_path.stem)

    if HAS_PLOTTING:
        nome = weights_path.stem
        save_confusion_matrix_image(cm, num_classes, nome)
        plot_campana_errori(errors_np, CURRENT_DIR / f"campana_{nome}.png")
        plot_fotoromanzo_viaggio(
            np.arange(100),
            fotoromanzo_reale,
            fotoromanzo_predetto,
            CURRENT_DIR / f"fotoromanzo_{nome}.png",
        )
        if (
            n_trips_profile > 0
            and sum_true_profile is not None
            and sum_pred_profile is not None
        ):
            avg_true_profile = sum_true_profile / n_trips_profile
            avg_pred_profile = sum_pred_profile / n_trips_profile
            plot_fotoromanzo_medio(
                np.arange(len(avg_true_profile)),
                avg_true_profile,
                avg_pred_profile,
                CURRENT_DIR / f"fotoromanzo_medio_{nome}.png",
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


def _compute_macro_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.size == 0:
        return 0.0, 0.0, 0.0

    num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class, pred_class in zip(y_true, y_pred):
        cm[int(true_class), int(pred_class)] += 1

    precisions = []
    recalls = []
    f1_scores = []

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

        if (tp + fp) > 0:
            precisions.append(precision)
        if (tp + fn) > 0:
            recalls.append(recall)
        if (tp + fp) > 0 and (tp + fn) > 0 and (precision + recall) > 0:
            f1_scores.append(f1)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return macro_precision, macro_recall, macro_f1


def _decode_route_name(route_decoder, route_id_encoded: int):
    if route_decoder is None:
        return f"ID {route_id_encoded}"

    try:
        return str(route_decoder.inverse_transform([int(route_id_encoded)])[0])
    except Exception:
        return f"ID {route_id_encoded}"


def save_route_metrics_report(route_metrics, route_decoder, model_name: str):
    rows = []
    for route_id_encoded in sorted(route_metrics.keys()):
        route_data = route_metrics[route_id_encoded]
        errors = np.asarray(route_data["errors"], dtype=np.float64)
        y_true = np.asarray(route_data["y_true"], dtype=np.int64)
        y_pred = np.asarray(route_data["y_pred"], dtype=np.int64)

        if errors.size == 0:
            continue

        macro_precision, macro_recall, macro_f1 = _compute_macro_precision_recall_f1(
            y_true, y_pred
        )
        occupancy_precision_pct = (
            float((y_true == y_pred).mean() * 100.0) if y_true.size > 0 else 0.0
        )

        rows.append(
            {
                "route_id_encoded": int(route_id_encoded),
                "route": _decode_route_name(route_decoder, route_id_encoded),
                "samples": int(errors.size),
                "precision_macro": macro_precision,
                "recall_macro": macro_recall,
                "f1_macro": macro_f1,
                "sigma_s": float(np.std(errors)),
                "mae_s": float(np.mean(np.abs(errors))),
                "rmse_s": float(np.sqrt(np.mean(errors**2))),
                "bias_s": float(np.mean(errors)),
                "occupancy_precision_pct": occupancy_precision_pct,
            }
        )

    if not rows:
        print("[!] Nessuna metrica per-route da salvare.")
        return

    df_report = pd.DataFrame(rows).sort_values(by=["route_id_encoded"], ascending=True)

    csv_path = CURRENT_DIR / f"route_metrics_{model_name}.csv"
    md_path = CURRENT_DIR / f"route_metrics_{model_name}.md"
    df_report.to_csv(csv_path, index=False)

    md_lines = [
        f"# Report metriche per route - {model_name}",
        "",
        "Metriche occupancy (precision/recall/f1) calcolate con media macro, come nel report globale.",
        "",
        "| route_id_encoded | route | samples | precision_macro | recall_macro | f1_macro | sigma_s | mae_s | rmse_s | bias_s | occupancy_precision_pct |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in df_report.itertuples(index=False):
        md_lines.append(
            f"| {row.route_id_encoded} | {row.route} | {row.samples} | "
            f"{row.precision_macro:.4f} | {row.recall_macro:.4f} | {row.f1_macro:.4f} | "
            f"{row.sigma_s:.2f} | {row.mae_s:.2f} | {row.rmse_s:.2f} | {row.bias_s:.2f} | "
            f"{row.occupancy_precision_pct:.2f} |"
        )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Report metriche per-route salvato: {md_path}")
    print(f"[OK] Report CSV per-route salvato: {csv_path}")


def save_confusion_matrix_csv(cm: np.ndarray, model_name: str):
    output_path = CURRENT_DIR / f"confusion_matrix_{model_name}.csv"
    df_cm = pd.DataFrame(cm)
    df_cm.to_csv(output_path, index_label="true_class")
    print(f"[OK] Matrice di confusione CSV salvata: {output_path}")


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
    plt.style.use("seaborn-v0_8-whitegrid")

    limite = np.percentile(np.abs(errori_secondi), 95)
    errori_filtrati = errori_secondi[
        (errori_secondi >= -limite) & (errori_secondi <= limite)
    ]

    ax.hist(
        errori_filtrati,
        bins=60,
        density=True,
        alpha=0.4,
        color="#3498db",
        edgecolor="white",
    )

    kde = gaussian_kde(errori_filtrati)
    x_vals = np.linspace(min(errori_filtrati), max(errori_filtrati), 500)
    ax.plot(
        x_vals,
        kde(x_vals),
        color="#2980b9",
        linewidth=2.5,
        label="Densità di Probabilità",
    )

    ax.axvline(
        x=0,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        zorder=5,
        label="Perfezione (Errore = 0s)",
    )

    ax.set_title(
        "Densità dell'Errore di Predizione sul Ritardo",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.set_xlabel("Errore di Predizione (Secondi)", fontsize=12, weight="bold")
    ax.set_ylabel("Frequenza (Densità)", fontsize=12, weight="bold")

    mae_reale = np.mean(np.abs(errori_secondi))
    ax.annotate(
        f"MAE Globale: {mae_reale / 60:.1f} min\nLa campana stretta dimostra\nche il grosso degli errori\nè concentrato vicino allo zero.",
        xy=(0.02, 0.8),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bdc3c7", lw=1),
        fontsize=10,
        color="#2c3e50",
    )

    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Campana salvata in: {save_path.name}")


def plot_fotoromanzo_viaggio(segmenti, ritardo_reale, ritardo_predetto, save_path):
    print("[*] Generazione fotoromanzo spaziale in corso...")

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid")

    ax.plot(
        segmenti,
        ritardo_reale,
        color="#7f8c8d",
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label="Ritardo Reale (Ground Truth)",
    )
    ax.plot(
        segmenti,
        ritardo_predetto,
        color="#2c3e50",
        linestyle="-",
        linewidth=3,
        label="Predizione Modello",
    )
    ax.fill_between(
        segmenti,
        ritardo_reale,
        ritardo_predetto,
        color="#e74c3c",
        alpha=0.15,
        label="Scarto (Errore)",
    )

    ax.set_title(
        "Inseguimento della Traiettoria: Analisi di un Singolo Viaggio",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.set_xlabel(
        "Avanzamento Corsa (Indice del Segmento Spaziale 0-100)",
        fontsize=12,
        weight="bold",
    )
    ax.set_ylabel("Ritardo Cumulato (Secondi)", fontsize=12, weight="bold")

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0, max(segmenti))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Fotoromanzo salvato in: {save_path.name}")


def plot_fotoromanzo_medio(
    segmenti, ritardo_reale_medio, ritardo_predetto_medio, save_path
):
    print("[*] Generazione fotoromanzo medio in corso...")

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid")

    ax.plot(
        segmenti,
        ritardo_reale_medio,
        color="#7f8c8d",
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label="Ritardo Reale Medio (Ground Truth)",
    )
    ax.plot(
        segmenti,
        ritardo_predetto_medio,
        color="#2c3e50",
        linestyle="-",
        linewidth=3,
        label="Predizione Media del Modello",
    )
    ax.fill_between(
        segmenti,
        ritardo_reale_medio,
        ritardo_predetto_medio,
        color="#e74c3c",
        alpha=0.15,
        label="Scarto Medio (Errore)",
    )

    ax.set_title(
        "Traiettoria Media: Predizione Media vs Viaggio Medio",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.set_xlabel(
        "Avanzamento Corsa (Indice del Segmento Spaziale 0-100)",
        fontsize=12,
        weight="bold",
    )
    ax.set_ylabel("Ritardo Cumulato Medio (Secondi)", fontsize=12, weight="bold")

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0, max(segmenti))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Fotoromanzo medio salvato in: {save_path.name}")


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
