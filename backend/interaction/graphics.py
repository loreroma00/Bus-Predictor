import os
import re
import glob
import matplotlib.pyplot as plt

def elabora_e_inquadra(log_filepath):
    print(f"\n[*] Analisi del segnale per: {log_filepath}")
    
    # Deriviamo il nome del file di output e il titolo dal nome del file txt
    base_name = os.path.splitext(os.path.basename(log_filepath))[0]
    save_filepath = f"loss_{base_name}.png"
    titolo_grafico = f"Dinamica di Apprendimento - {base_name.replace('_', ' ').upper()}"

    # --- LAYER INTERNO: Estrazione del Segnale ---
    epochs, train_loss, val_loss = [], [], []
    pattern = r"Epoch \[(\d+)/\d+\].*?Tot Train: ([\d.]+) \(Val: ([\d.]+)\)"
    
    with open(log_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        match = re.search(pattern, line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))

    if not epochs:
        print(f"[!] Nessun dato compatibile trovato in {log_filepath}. Salto il file.")
        return False

    best_epoch_idx = val_loss.index(min(val_loss))
    best_epoch = epochs[best_epoch_idx]
    best_val = val_loss[best_epoch_idx]

    # --- LAYER ESTERNO: Composizione ed Estetica ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Tracciamento curve
    ax.plot(epochs, train_loss, label='Training Loss', color='#2c3e50', linewidth=2.5, alpha=0.85)
    ax.plot(epochs, val_loss, label='Validation Loss', color='#e74c3c', linewidth=2.5, alpha=0.9)

    # Linea e marcatore di Early Stopping
    ax.axvline(x=best_epoch, color='#7f8c8d', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.scatter(best_epoch, best_val, color='#c0392b', s=80, zorder=5) 
    
    # Annotazione spostata dinamicamente
    offset_x = 2 if best_epoch < (max(epochs) - 10) else -15
    ax.annotate(f'Early Stopping\nEpoca {best_epoch}',
                xy=(best_epoch, best_val),
                xytext=(best_epoch + offset_x, best_val + (max(val_loss)-min(val_loss))*0.05),
                arrowprops=dict(facecolor='#34495e', arrowstyle='->', alpha=0.8),
                fontsize=11, weight='bold', color='#2c3e50')

    # Estetica assi e griglia
    ax.set_title(titolo_grafico, fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('Epoche di Addestramento', fontsize=12, weight='bold', labelpad=10)
    ax.set_ylabel('Loss Complessiva (MSE / CE)', fontsize=12, weight='bold', labelpad=10)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', frameon=True, fontsize=11, shadow=True)

    plt.tight_layout()
    plt.savefig(save_filepath, format='png', bbox_inches='tight')
    plt.close()
    
    print(f"[+] Render completato: {save_filepath}")
    return True

def main():
    print("=== MOTORE DI RENDERING GRAFICI LOSS ===")
    
    # Cerca tutti i file txt nella cartella corrente
    txt_files = glob.glob("*.txt")
    
    # Escludiamo il requirements.txt per non fare confusione
    if "requirements.txt" in txt_files:
        txt_files.remove("requirements.txt")
        
    if not txt_files:
        print("[!] Nessun file .txt trovato in questa cartella.")
        return

    print("\nFile di log rilevati:")
    print("[0] - GENERA GRAFICI PER TUTTI I FILE")
    for i, file in enumerate(txt_files, start=1):
        print(f"[{i}] - {file}")

    scelta = input("\nInserisci il numero dell'opzione desiderata: ")

    try:
        scelta_idx = int(scelta)
        if scelta_idx == 0:
            print("\n[*] Avvio rendering batch per tutti i log...")
            for file in txt_files:
                elabora_e_inquadra(file)
            print("\n[OK] Processo batch completato.")
        elif 1 <= scelta_idx <= len(txt_files):
            file_scelto = txt_files[scelta_idx - 1]
            elabora_e_inquadra(file_scelto)
        else:
            print("[!] Scelta non valida.")
    except ValueError:
        print("[!] Input non valido. Inserisci un numero.")

if __name__ == "__main__":
    main()
