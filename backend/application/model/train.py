"""Defines the training algorithm for the model."""

import os
import torch
import torch.nn as nn
import torch.optim as optim

import json
import argparse

from dataset import load_dataset
from model import BusLSTM, OccupancyLSTM

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PARQUET_DIR = os.path.join(PROJECT_ROOT, "parquets")
FILE_PATH = os.path.join(PARQUET_DIR, "dataset_lstm_final.parquet")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
EPOCHS = 100
ENCODER_HIDDEN_SIZE = 128
DECODER_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
TIME_LOSS_WEIGHT = 1.0


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """Creates a Logit Adjusted Cross Entropy Loss Function."""

    def __init__(
        self,
        freq: torch.Tensor,
        reduction: str = "mean",
        tau: float = 1.0,
        ignore_index: int = 7,
    ):
        """Initialize the loss function."""
        super().__init__()
        adjusted_freq = freq + 1e-8
        log_freq = torch.log(adjusted_freq)
        self.register_buffer("pi", log_freq)
        self.reduction = reduction
        self.tau = tau
        self.ignore_index: int = ignore_index

    def forward(self, logits, targets):
        """Define the forward chain of the function."""
        adjusted_logits = logits + (self.tau * self.pi)
        return nn.functional.cross_entropy(
            adjusted_logits,
            targets,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )


def train(
    loss_type: str,
    hyperparameter_iteration: int = 1,
    preloaded_data=None,
    config_modello=None,
):
    # 1. GESTIONE DELLA MEMORIA (Caricamento o Iniezione)
    if preloaded_data is None:
        print("Loading data from disk...")
        train_loader, val_loader, full_dataset, counts = load_dataset(
            FILE_PATH, train_size=0.8
        )
    else:
        print("Using preloaded data from RAM...")
        train_loader, val_loader, full_dataset, counts = preloaded_data

    # Estrazione Metadati
    n_x1_dense = full_dataset.n_x1_dense_features
    n_x2_dense = full_dataset.n_x2_dense_features
    x1_cat_cards = full_dataset.x1_cat_cardinalities
    x2_cat_cards = full_dataset.x2_cat_cardinalities

    # 2. INIZIALIZZAZIONE MODELLO
    print(f"Initializing model for iteration {hyperparameter_iteration}...")

    model_time = BusLSTM(
        n_x1_dense_features=n_x1_dense,
        n_x2_dense_features=n_x2_dense,
        x1_cat_cardinalities=x1_cat_cards,
        x2_cat_cardinalities=x2_cat_cards,
        encoder_hidden_size=ENCODER_HIDDEN_SIZE,
        lstm_hidden_size=DECODER_HIDDEN_SIZE,
    ).to(DEVICE)
    model_occupancy = OccupancyLSTM(
        n_x1_dense_features=n_x1_dense,
        n_x2_dense_features=n_x2_dense,
        x1_cat_cardinalities=x1_cat_cards,
        x2_cat_cardinalities=x2_cat_cards,
        encoder_hidden_size=ENCODER_HIDDEN_SIZE,
        lstm_hidden_size=DECODER_HIDDEN_SIZE,
        num_lstm_layers=LSTM_LAYERS,
    ).to(DEVICE)

    # 3. INTERRUTTORE DELLE LOSS
    match loss_type:
        case "mae":
            criterion_time = nn.L1Loss()
        case "huber":
            criterion_time = nn.HuberLoss()
        case "nll":
            criterion_time = nn.GaussianNLLLoss()
        case _:
            criterion_time = nn.MSELoss()

    count_amounts: torch.Tensor = counts.sum()
    frequencies: torch.Tensor = counts / count_amounts
    # OCCUPANCY_WEIGHTS = torch.tensor(
    #    [0.01, 0.05, 1.0, 5.0, 10.0, 10.0, 20.0], dtype=torch.float32
    # ).to(DEVICE)
    criterion_crowd = LogitAdjustedCrossEntropyLoss(frequencies).to(DEVICE)

    # 4. DUE OTTIMIZZATORI E DUE SCHEDULER INDIPENDENTI
    opt_time = optim.Adam(model_time.parameters(), lr=LEARNING_RATE)
    opt_crowd = optim.Adam(model_occupancy.parameters(), lr=LEARNING_RATE)

    sch_time = optim.lr_scheduler.CosineAnnealingLR(
        opt_time, T_max=EPOCHS, eta_min=1e-6
    )
    sch_crowd = optim.lr_scheduler.CosineAnnealingLR(
        opt_crowd, T_max=EPOCHS, eta_min=1e-6
    )

    print(f"Starting training on {DEVICE}...")
    print("Using dual-model architecture (BusLSTM + OccupancyLSTM)...")
    print(f"Using loss function: {loss_type.upper()}")

    best_val_loss_time = float("inf")
    best_val_loss_crowd = float("inf")
    patience = 12
    patience_counter_time = 0
    patience_counter_crowd = 0

    # Nomi file dinamici per non sovrascrivere
    best_time_path = f"bus_model_TIME_{loss_type}_{hyperparameter_iteration}.pth"
    best_crowd_path = f"bus_model_CROWD_{loss_type}_{hyperparameter_iteration}.pth"
    json_filename = f"hyperparameters_DUAL_{loss_type}_{hyperparameter_iteration}.json"

    # Flag per capire se un modello ha finito
    time_done = False
    crowd_done = False

    # 5. TRAINING LOOP
    for epoch in range(EPOCHS):
        if time_done and crowd_done:
            print(
                f"\n[!] ENTRAMBI I MODELLI HANNO RAGGIUNTO L'EARLY STOPPING. Addestramento concluso all'epoca {epoch}."
            )
            break

        model_time.train() if not time_done else model_time.eval()
        model_occupancy.train() if not crowd_done else model_occupancy.eval()

        running_loss_time = 0.0
        running_loss_crowd = 0.0

        for batch in train_loader:
            x1_cat, x1_dense, x2_cat, x2_dense, y_time, y_crowd, lengths, t_grid = [
                b.to(DEVICE) for b in batch
            ]
            seq_len = x2_dense.size(1)
            mask = torch.arange(seq_len, device=DEVICE).unsqueeze(
                0
            ) < lengths.unsqueeze(1)

            # --- CATENA RITARDO (Aggiorna solo se non in early stop) ---
            if not time_done:
                opt_time.zero_grad()
                pred_time = model_time(
                    x1_cat, x1_dense, x2_cat, x2_dense, lengths=lengths, t_grid=t_grid
                )

                pred_time_masked = pred_time.squeeze(-1)[mask]
                y_time_masked = y_time.squeeze(-1)[mask]

                if loss_type == "nll":
                    dummy_variance = torch.ones_like(pred_time_masked)
                    loss_time = criterion_time(
                        pred_time_masked, y_time_masked, dummy_variance
                    )
                else:
                    loss_time = criterion_time(pred_time_masked, y_time_masked)

                loss_time.backward()
                opt_time.step()
                running_loss_time += loss_time.item()

            # --- CATENA PASSEGGERI (Aggiorna solo se non in early stop) ---
            if not crowd_done:
                opt_crowd.zero_grad()
                pred_crowd = model_occupancy(
                    x1_cat, x1_dense, x2_cat, x2_dense, lengths=lengths
                )

                pred_crowd_masked = pred_crowd[mask]  # [N_valid, 7]
                y_crowd_masked = y_crowd[mask]  # [N_valid]

                loss_crowd = criterion_crowd(pred_crowd_masked, y_crowd_masked)
                loss_crowd.backward()
                opt_crowd.step()
                running_loss_crowd += loss_crowd.item()

        # --- VALIDATION ---
        model_time.eval()
        model_occupancy.eval()
        val_loss_time = 0.0
        val_loss_crowd = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x1_cat, x1_dense, x2_cat, x2_dense, y_time, y_crowd, lengths, t_grid = [
                    b.to(DEVICE) for b in batch
                ]
                seq_len = x2_dense.size(1)
                mask = torch.arange(seq_len, device=DEVICE).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)

                # Valutazione Tempo
                if not time_done:
                    pred_time = model_time(
                        x1_cat,
                        x1_dense,
                        x2_cat,
                        x2_dense,
                        lengths=lengths,
                        t_grid=t_grid,
                    )
                    pred_time_masked = pred_time.squeeze(-1)[mask]
                    y_time_masked = y_time.squeeze(-1)[mask]
                    if loss_type == "nll":
                        dummy_variance = torch.ones_like(pred_time_masked)
                        l_time = criterion_time(
                            pred_time_masked, y_time_masked, dummy_variance
                        )
                    else:
                        l_time = criterion_time(pred_time_masked, y_time_masked)
                    val_loss_time += l_time.item()

                # Valutazione Passeggeri
                if not crowd_done:
                    pred_crowd = model_occupancy(
                        x1_cat, x1_dense, x2_cat, x2_dense, lengths=lengths
                    )
                    pred_crowd_masked = pred_crowd[mask]  # [N_valid, 7]
                    y_crowd_masked = y_crowd[mask]  # [N_valid]
                    l_crowd = criterion_crowd(pred_crowd_masked, y_crowd_masked)
                    val_loss_crowd += l_crowd.item()

        # Medie Train/Val
        avg_train_time = running_loss_time / len(train_loader) if not time_done else 0
        avg_train_crowd = (
            running_loss_crowd / len(train_loader) if not crowd_done else 0
        )
        avg_val_time = (
            val_loss_time / len(val_loader) if not time_done else best_val_loss_time
        )
        avg_val_crowd = (
            val_loss_crowd / len(val_loader) if not crowd_done else best_val_loss_crowd
        )

        if not time_done:
            sch_time.step()
        if not crowd_done:
            sch_crowd.step()

        curr_lr_time = opt_time.param_groups[0]["lr"] if not time_done else 0
        curr_lr_crowd = opt_crowd.param_groups[0]["lr"] if not crowd_done else 0

        # --- REPORT TELEMETRIA DUAL ---
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        if not time_done:
            print(
                f"  [TIME]  LR: {curr_lr_time:.6f} | Train: {avg_train_time:.4f} | Val: {avg_val_time:.4f}"
            )
        if not crowd_done:
            print(
                f"  [CROWD] LR: {curr_lr_crowd:.6f} | Train: {avg_train_crowd:.4f} | Val: {avg_val_crowd:.4f}"
            )

        # --- EARLY STOPPING INDIPENDENTE ---
        # 1. RITARDO
        if not time_done:
            if avg_val_time < best_val_loss_time:
                best_val_loss_time = avg_val_time
                patience_counter_time = 0
                torch.save(model_time.state_dict(), best_time_path)
                print(f"    [+] Modello TIME migliorato e salvato!")
            else:
                patience_counter_time += 1
                if patience_counter_time >= patience:
                    print(f"    [!] EARLY STOPPING INNESCATO per il modello TIME.")
                    time_done = True

        # 2. PASSEGGERI
        if not crowd_done:
            if avg_val_crowd < best_val_loss_crowd:
                best_val_loss_crowd = avg_val_crowd
                patience_counter_crowd = 0
                torch.save(model_occupancy.state_dict(), best_crowd_path)
                print(f"    [+] Modello CROWD migliorato e salvato!")
            else:
                patience_counter_crowd += 1
                if patience_counter_crowd >= patience:
                    print(f"    [!] EARLY STOPPING INNESCATO per il modello CROWD.")
                    crowd_done = True

    # Salvataggio Metadati finale
    model_config = {
        "x1_dense_features": n_x1_dense,
        "x2_dense_features": n_x2_dense,
        "x1_cat_cards": x1_cat_cards,
        "x2_cat_cards": x2_cat_cards,
        "max_stops": full_dataset.max_stops,
        "epochs_run": epoch + 1,
        "loss_type": loss_type,
        "encoder_hidden_size": ENCODER_HIDDEN_SIZE,
        "decoder_hidden_size": DECODER_HIDDEN_SIZE,
        "num_lstm_layers": LSTM_LAYERS,
        "Learning_Rate": LEARNING_RATE,
        "time_model_stopped": time_done,
        "crowd_model_stopped": crowd_done,
    }

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)


# ==========================================
# GESTORE DEL BATCH
# ==========================================
def batch_train():
    print("Initializing batch training...")

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    GRID_SEARCH_PATH = os.path.join(CURRENT_DIR, "grid_search.json")

    try:
        # 2. Usiamo il percorso assoluto per aprirlo
        with open(GRID_SEARCH_PATH, "r") as f:
            experiments = json.load(f)
    except FileNotFoundError:
        print(f"[ERRORE] File mancante nel percorso esatto: {GRID_SEARCH_PATH}")
        return

    print("Caricamento Spaziale Globale (una tantum)...")
    # Carichiamo i dati una volta sola
    preloaded_data = load_dataset(FILE_PATH, train_size=0.8)

    # Eseguiamo gli esperimenti in sequenza
    for idx, exp in enumerate(experiments):
        loss_corrente = exp.get("loss", "mse")
        print(f"\n{'=' * 50}")
        print(f"STARTING EXPERIMENT No: {idx} | LOSS: {loss_corrente.upper()}")
        print(f"{'=' * 50}")

        train(
            loss_type=loss_corrente,
            hyperparameter_iteration=idx,
            preloaded_data=preloaded_data,
            config_modello=exp,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loss", type=str, default="mse", choices=["mse", "mae", "huber", "nll"]
    )
    parser.add_argument("--batch-train", action="store_true")

    args = parser.parse_args()

    if args.batch_train:
        batch_train()
    else:
        train(loss_type=args.loss, hyperparameter_iteration=99)
