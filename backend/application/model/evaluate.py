import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import BusDataset
from model import BusLSTM

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
        print("Devi creare un nuovo parquet con i dati freschi e chiamarlo 'dataset_lstm_final_TEST.parquet'!")
        return

    print("Caricamento nuovi dati spaziali in corso...")
    test_dataset = BusDataset(str(TEST_FILE_PATH))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    # 2. Caricamento Architettura Dinamica
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = BusLSTM(
        n_x1_dense_features=config["x1_dense_features"], 
        n_x2_dense_features=config["x2_dense_features"],
        x1_cat_cardinalities=config["x1_cat_cards"],
        x2_cat_cardinalities=config["x2_cat_cards"],
        encoder_hidden_size=config["encoder_hidden_size"],
        lstm_hidden_size=config["decoder_hidden_size"],
        num_lstm_layers=config["num_lstm_layers"]
    ).to(DEVICE)
    
    # Caricamento dei pesi specifici
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    print(f"Modello armato e pronto. Loss originale: {config.get('loss_type', 'N/A').upper()}")

    # 3. Metriche Umane
    total_time_mae_seconds = 0.0
    total_time_mse_seconds = 0.0
    total_crowd_correct = 0
    total_samples = 0
    
    print("\nElaborazione dei viaggi inediti...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x1_cat, x1_dense, x2_cat, x2_dense, y_time, y_crowd = [b.to(DEVICE) for b in batch]
            
            # Predizione
            pred_time, pred_crowd_logits = model(x1_cat, x1_dense, x2_cat, x2_dense)
            
            # --- CALCOLO ERRORI TEMPO ---
            pred_time_sec = pred_time.squeeze(-1) * 3600.0
            true_time_sec = y_time.squeeze(-1) * 3600.0
            
            # MAE e MSE
            mae = torch.abs(pred_time_sec - true_time_sec).sum().item()
            mse = ((pred_time_sec - true_time_sec)**2).sum().item()
            
            total_time_mae_seconds += mae
            total_time_mse_seconds += mse
            
            # --- CALCOLO ERRORI FOLLA ---
            pred_crowd_classes = pred_crowd_logits.argmax(dim=-1)
            correct_predictions = (pred_crowd_classes == y_crowd).sum().item()
            total_crowd_correct += correct_predictions
            
            total_samples += y_time.numel() 

            # --- ESTRAZIONE VISIVA (Dal primo batch) ---
            if i == 0:
                print("\n[ISPEZIONE VISIVA A CAMPIONE - Viaggio #0]")
                print(f"{'Segmento':<10} | {'Reale (sec)':<15} | {'Previsto (sec)':<15} | {'Errore (sec)':<15} || {'Folla Reale':<15} | {'Folla Prevista':<15}")
                print("-" * 95)
                for seg in [0, 25, 50, 75, 99]:
                    r_time = true_time_sec[0, seg].item()
                    p_time = pred_time_sec[0, seg].item()
                    err_time = abs(r_time - p_time)
                    r_crowd = y_crowd[0, seg].item()
                    p_crowd = pred_crowd_classes[0, seg].item()
                    
                    print(f"Seg {seg:02d}    | {r_time:>10.0f} s    | {p_time:>10.0f} s    | {err_time:>10.0f} s    || Classe {r_crowd:<8} | Classe {p_crowd:<8}")

    # 4. Statistiche Finali
    avg_mae = total_time_mae_seconds / total_samples
    rmse = (total_time_mse_seconds / total_samples) ** 0.5
    accuracy_crowd = (total_crowd_correct / total_samples) * 100

    print("\n" + "="*50)
    print("RISULTATI SUL TEST SET INEDITO")
    print("="*50)
    print(f"Viaggi elaborati : {total_samples // 100}")
    print(f"Punti spaziali   : {total_samples}")
    print("-" * 50)
    print(f"Errore Medio (MAE) Tempo  : {avg_mae:.1f} secondi (+/- {avg_mae/60:.1f} minuti)")
    print(f"Errore Estremo (RMSE)     : {rmse:.1f} secondi")
    print(f"Accuratezza Folla         : {accuracy_crowd:.1f}%")
    print("="*50)


if __name__ == "__main__":
    print("\n--- LABORATORIO DI COLLAUDO ---")
    
    # 1. Scansione modelli
    modelli_disponibili = list(CURRENT_DIR.glob("bus_model_*.pth"))
    
    if not modelli_disponibili:
        print("[ERRORE] Nessun modello trovato in questa cartella. Esegui prima train.py!")
        exit()
        
    # 2. Menu Interattivo
    print("\nQuale Digital Twin vuoi testare?")
    for idx, path_modello in enumerate(modelli_disponibili):
        print(f"[{idx}] - {path_modello.name}")
        
    try:
        scelta = int(input("\nInserisci il numero: "))
        pesi_scelti = modelli_disponibili[scelta]
        
        # 3. Derivazione nome del JSON
        nome_json = pesi_scelti.name.replace("bus_model_", "hyperparameters_").replace(".pth", ".json")
        json_scelto = CURRENT_DIR / nome_json
        
        if not json_scelto.exists():
            print(f"\n[ERRORE] Impossibile trovare il file di configurazione associato: {nome_json}")
            exit()
            
        # 4. Avvio valutazione
        evaluate_model(weights_path=pesi_scelti, config_path=json_scelto)
        
    except (IndexError, ValueError):
        print("\n[ERRORE] Selezione non valida.")
    except KeyboardInterrupt:
        print("\n[!] Collaudo interrotto.")
