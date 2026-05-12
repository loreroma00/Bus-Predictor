import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_route_metrics(csv_ode: str, csv_lstm: str, output_name: str):
    print("[*] Caricamento dati per singola linea in corso...")
    
    # 1. Caricamento Dati
    df_ode = pd.read_csv(csv_ode)
    df_lstm = pd.read_csv(csv_lstm)
    
    # Aggiunta etichette per il grafico
    df_ode['Modello'] = 'ODE-LSTM (Continuo)'
    df_lstm['Modello'] = 'LSTM (Discreto)'
    
    # Unione dei due dataset
    df_all = pd.concat([df_ode, df_lstm], ignore_index=True)
    
    # Conversione da secondi a minuti per maggiore leggibilità
    df_all['mae_min'] = df_all['mae_s'] / 60.0
    df_all['bias_min'] = df_all['bias_s'] / 60.0
    
    # 2. Impostazione Estetica (Stile Accademico pulito)
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # Proprietà dei marcatori per la media (il pallino bianco)
    mean_props = {
        "marker": "o", 
        "markerfacecolor": "white", 
        "markeredgecolor": "black", 
        "markersize": "8"
    }

    # --- GRAFICO 1: Distribuzione del MAE ---
    sns.boxplot(
        data=df_all, 
        x='Modello', 
        y='mae_min', 
        palette=['#3498db', '#e74c3c'],  # Colori: Blu (ODE) e Rosso (LSTM)
        ax=axes[0],
        width=0.4,
        showmeans=True,
        meanprops=mean_props
    )
    axes[0].set_title("Distribuzione dell'Errore Assoluto Medio (MAE)", fontsize=14, weight='bold', pad=15)
    axes[0].set_ylabel('MAE (Minuti)', fontsize=12, weight='bold')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelsize=12)

    # --- GRAFICO 2: Distribuzione del BIAS ---
    sns.boxplot(
        data=df_all, 
        x='Modello', 
        y='bias_min', 
        palette=['#3498db', '#e74c3c'], 
        ax=axes[1],
        width=0.4,
        showmeans=True,
        meanprops=mean_props
    )
    # Linea dello zero per far capire dove sta la perfezione imparziale
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Bias Zero (Imparziale)')
    axes[1].set_title("Distribuzione dell'Errore Sistematico (Bias)", fontsize=14, weight='bold', pad=15)
    axes[1].set_ylabel('Bias (Minuti)', fontsize=12, weight='bold')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].legend(loc='lower right')

    # Titolo globale
    fig.suptitle('Impatto della Topologia Urbana sulle Prestazioni Predittive', fontsize=18, weight='bold', y=1.05)

    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Grafico salvato con successo come: {output_name}")

if __name__ == "__main__":
    # Assicurati che i nomi corrispondano esattamente ai file nella tua cartella
    FILE_ODE = "route_metrics_bus_model_mse_ODE.csv"
    FILE_LSTM = "route_metrics_bus_model_mse_99 LSTM.csv"
    
    plot_route_metrics(FILE_ODE, FILE_LSTM, "boxplot_confronto_linee.png")