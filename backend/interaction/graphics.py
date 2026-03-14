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

def generate_fotoromanzo(predicted_df, actual_df, stops_map, route_id, trip_id, output_path):
    """Generate a comparison chart: predicted delay vs actual delay vs scheduled baseline.

    Args:
        predicted_df: DataFrame with stop_sequence, predicted_delay_sec (may be None/empty)
        actual_df: DataFrame with stop_sequence, schedule_adherence (may be None/empty)
        stops_map: dict stop_sequence → {stop_name, ...} from TopologyLedger
        route_id: route identifier string
        trip_id: trip identifier string
        output_path: where to save the PNG
    """
    import numpy as np

    has_predicted = predicted_df is not None and not predicted_df.empty
    has_actual = actual_df is not None and not actual_df.empty

    if not has_predicted and not has_actual:
        print(f"[!] No data to plot for trip {trip_id}")
        return False

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    # Scheduled baseline (on-time = 0 delay)
    all_seqs = set()
    if has_predicted:
        all_seqs.update(predicted_df['stop_sequence'].values)
    if has_actual:
        all_seqs.update(actual_df['stop_sequence'].values)
    seq_range = sorted(all_seqs)
    ax.axhline(y=0, color='#7f8c8d', linestyle='--', linewidth=1.5, alpha=0.8, label='Scheduled (on-time)')

    # Predicted curve
    if has_predicted:
        pred_sorted = predicted_df.sort_values('stop_sequence')
        ax.plot(
            pred_sorted['stop_sequence'], pred_sorted['predicted_delay_sec'],
            label='Predicted delay', color='#2c3e50', linewidth=2.5, alpha=0.85,
        )

    # Actual measurements (median per stop_sequence since there can be multiple pings)
    rmse = None
    if has_actual:
        actual_grouped = actual_df.groupby('stop_sequence')['schedule_adherence'].median().reset_index()
        actual_grouped = actual_grouped.sort_values('stop_sequence')
        ax.scatter(
            actual_grouped['stop_sequence'], actual_grouped['schedule_adherence'],
            label='Actual delay', color='#e74c3c', s=40, zorder=5, alpha=0.9,
        )

        # Compute RMSE if both are available
        if has_predicted:
            merged = actual_grouped.merge(
                predicted_df[['stop_sequence', 'predicted_delay_sec']],
                on='stop_sequence', how='inner',
            )
            if not merged.empty:
                errors = (merged['predicted_delay_sec'] - merged['schedule_adherence']) ** 2
                rmse = float(np.sqrt(errors.mean()))

    # X-axis: stop names
    if stops_map and seq_range:
        tick_seqs = seq_range[::max(1, len(seq_range) // 15)]  # ~15 ticks max
        tick_labels = [
            stops_map.get(s, {}).get('stop_name', str(s))[:18]
            for s in tick_seqs
        ]
        ax.set_xticks(tick_seqs)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    # Y-axis: format as mm:ss
    def fmt_delay(x, _):
        sign = '-' if x < 0 else '+'
        mins, secs = divmod(abs(int(x)), 60)
        return f"{sign}{mins}:{secs:02d}"

    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_delay))

    # Title
    short_trip = trip_id[:30] + '...' if len(trip_id) > 30 else trip_id
    rmse_str = f" | RMSE: {rmse:.1f}s" if rmse is not None else ""
    ax.set_title(
        f"Fotoromanzo - Route {route_id} | {short_trip}{rmse_str}",
        fontsize=13, weight='bold', pad=15,
    )
    ax.set_xlabel('Stop Sequence', fontsize=11, weight='bold', labelpad=8)
    ax.set_ylabel('Delay (mm:ss)', fontsize=11, weight='bold', labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='best', frameon=True, fontsize=10, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close()

    print(f"[+] Fotoromanzo saved: {output_path}")
    return True


if __name__ == "__main__":
    main()
