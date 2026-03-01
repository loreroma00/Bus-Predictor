import os
import torch
import torch.nn as nn
import torch.optim as optim

import json
import argparse

from dataset import load_dataset
from model import BusLSTM, BusODELSTM

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


def train(loss_type: str, hyperparameter_iteration: int = 1, preloaded_data=None, config_modello=None):
    
    # 1. GESTIONE DELLA MEMORIA (Caricamento o Iniezione)
    if preloaded_data is None:
        print("Loading data from disk...")
        train_loader, val_loader, full_dataset = load_dataset(FILE_PATH, train_size=0.8)
    else:
        print("Using preloaded data from RAM...")
        train_loader, val_loader, full_dataset = preloaded_data

    # Estrazione Metadati
    n_x1_dense = full_dataset.n_x1_dense_features
    n_x2_dense = full_dataset.n_x2_dense_features
    x1_cat_cards = full_dataset.x1_cat_cardinalities
    x2_cat_cards = full_dataset.x2_cat_cardinalities

    # 2. INIZIALIZZAZIONE MODELLO
    print(f"Initializing model for iteration {hyperparameter_iteration}...")

    if args.arch == "ode_lstm":
        model = BusODELSTM(
            n_x1_dense_features=n_x1_dense, 
            n_x2_dense_features=n_x2_dense,
            x1_cat_cardinalities=x1_cat_cards, 
            x2_cat_cardinalities=x2_cat_cards,
            encoder_hidden_size=ENCODER_HIDDEN_SIZE,
            lstm_hidden_size=DECODER_HIDDEN_SIZE,
            num_lstm_layers=LSTM_LAYERS
        ).to(DEVICE)
    if args.arch == "lstm":
        model = BusLSTM(
            n_x1_dense_features=n_x1_dense, 
            n_x2_dense_features=n_x2_dense,
            x1_cat_cardinalities=x1_cat_cards, 
            x2_cat_cardinalities=x2_cat_cards,
            encoder_hidden_size=ENCODER_HIDDEN_SIZE,
            lstm_hidden_size=DECODER_HIDDEN_SIZE,
            num_lstm_layers=LSTM_LAYERS
        ).to(DEVICE)


    CROWD_LOSS_WEIGHT: float = 1.0
    
    # 3. INTERRUTTORE DELLE LOSS
    match loss_type:
        case "mae":
            criterion_time = nn.L1Loss()
            CROWD_LOSS_WEIGHT = 0.25
        case "huber":
            criterion_time = nn.HuberLoss()
            CROWD_LOSS_WEIGHT = 0.025
        case "nll":
            criterion_time = nn.GaussianNLLLoss()
            CROWD_LOSS_WEIGHT = 0.03
        case _:
            criterion_time = nn.MSELoss()
            CROWD_LOSS_WEIGHT = 0.05
            
    criterion_crowd = nn.CrossEntropyLoss(ignore_index=7) 

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,       # Il numero totale di epoche (es. 100). Indica quando la curva tocca il fondo.
        eta_min=1e-6        # Il pavimento assoluto. Evita che il learning rate arrivi esattamente a 0.
    )
    
    print(f"Starting training on {DEVICE}...")
    print(f"Using architecture {args.arch}...")
    print(f"Using loss function: {loss_type.upper()}")

    best_val_loss = float('inf') 
    patience = 12                
    patience_counter = 0         
    
    # Nomi file dinamici per non sovrascrivere
    best_model_path = f"bus_model_{loss_type}_{hyperparameter_iteration}.pth"
    json_filename = f"hyperparameters_{loss_type}_{hyperparameter_iteration}.json"

    # 4. TRAINING LOOP
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_loss_time = 0.0   
        running_loss_crowd = 0.0  

        for batch in train_loader:
            x1_cat, x1_dense, x2_cat, x2_dense, y_time, y_crowd = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            pred_time, pred_crowd = model(x1_cat, x1_dense, x2_cat, x2_dense)

            # --- IL TRICK DELLA NLL ---
            if loss_type == "nll":
                dummy_variance = torch.ones_like(pred_time)
                loss_time = criterion_time(pred_time, y_time, dummy_variance)
            else:
                loss_time = criterion_time(pred_time, y_time)
            # --------------------------
            
            pred_crowd_permuted = pred_crowd.permute(0, 2, 1) 
            loss_crowd = criterion_crowd(pred_crowd_permuted, y_crowd)

            total_loss = (loss_time * TIME_LOSS_WEIGHT) + (loss_crowd * CROWD_LOSS_WEIGHT)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_loss_time += loss_time.item()   
            running_loss_crowd += loss_crowd.item() 

        # --- CALCOLO MEDIE TRAIN ---
        avg_train_loss = running_loss / len(train_loader)
        avg_train_loss_time = running_loss_time / len(train_loader)
        avg_train_loss_crowd = running_loss_crowd / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_loss_time = 0.0
        val_loss_crowd = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x1_cat, x1_dense, x2_cat, x2_dense, y_time, y_crowd = [b.to(DEVICE) for b in batch]
                pred_time, pred_crowd = model(x1_cat, x1_dense, x2_cat, x2_dense)

                if loss_type == "nll":
                    dummy_variance = torch.ones_like(pred_time)
                    l_time = criterion_time(pred_time, y_time, dummy_variance)
                else:
                    l_time = criterion_time(pred_time, y_time)
                
                pred_crowd_permuted = pred_crowd.permute(0, 2, 1)
                l_crowd = criterion_crowd(pred_crowd_permuted, y_crowd)
                
                val_loss += (l_time + (l_crowd * CROWD_LOSS_WEIGHT)).item()
                val_loss_time += l_time.item()
                val_loss_crowd += l_crowd.item()

        # --- CALCOLO MEDIE VAL ---
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_time = val_loss_time / len(val_loader)
        avg_val_loss_crowd = val_loss_crowd / len(val_loader)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- REPORT TELEMETRIA ---
        print(f"Epoch [{epoch + 1}/{EPOCHS}] (LR: {current_lr:.6f}) | "
              f"Tot Train: {avg_train_loss:.4f} (Val: {avg_val_loss:.4f}) | "
              f"Time {loss_type.upper()}: {avg_train_loss_time:.4f} (Val: {avg_val_loss_time:.4f}) | "
              f"Crowd CE: {avg_train_loss_crowd:.4f} (Val: {avg_val_loss_crowd:.4f})")
        
        # --- EARLY STOPPING E SALVATAGGIO METADATI ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  [+] Modello migliorato e salvato! (Best Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  [-] Nessun miglioramento da {patience_counter} epoche.")
            if patience_counter >= patience:
                print(f"\n[!] EARLY STOPPING INNESCATO all'epoca {epoch + 1}.")
                
                model_config = {
                    "x1_dense_features": n_x1_dense,
                    "x2_dense_features": n_x2_dense,
                    "x1_cat_cards": x1_cat_cards,
                    "x2_cat_cards": x2_cat_cards,
                    "epochs": epoch + 1,
                    "loss_type": loss_type,
                    "encoder_hidden_size": ENCODER_HIDDEN_SIZE,
                    "decoder_hidden_size": DECODER_HIDDEN_SIZE,
                    "num_lstm_layers": LSTM_LAYERS,
                    "Learning_Rate": LEARNING_RATE
                }
                
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(model_config, f, indent=2, ensure_ascii=False)
                break 
 
# ==========================================
# GESTORE DEL BATCH (LA FABBRICA DEI MODELLI)
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
        print(f"\n{'='*50}")
        print(f"STARTING EXPERIMENT No: {idx} | LOSS: {loss_corrente.upper()}")
        print(f"{'='*50}")
        
        train(
            loss_type=loss_corrente,
            hyperparameter_iteration=idx,
            preloaded_data=preloaded_data,
            config_modello=exp
        )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "mae", "huber", "nll"],
        help="Choose loss function"
    )
    parser.add_argument(
        "--batch-train",
        action="store_true",
        help="Uses 'grid_search.json' to batch train several models"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="lstm",
        choices=["lstm", "ode_lstm"],
        help="Architecture to use"
    )
    
    args = parser.parse_args()

    if args.batch_train:
        batch_train()
    else:
        train(loss_type=args.loss, hyperparameter_iteration=99) # Il 99 indica che è una run isolata
