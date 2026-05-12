import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mae_vs_sigma_scatter(csv_ode: str, csv_lstm: str, output_name: str):
    print("[*] Caricamento dati per l'analisi a dispersione...")
    
    # 1. Caricamento Dati
    df_ode = pd.read_csv(csv_ode)
    df_lstm = pd.read_csv(csv_lstm)
    
    # Aggiunta etichette per distinguere i modelli nel grafico
    df_ode['Modello'] = 'ODE-LSTM (Continuo)'
    df_lstm['Modello'] = 'LSTM (Discreto)'
    
    # Unione dei dataset
    df_all = pd.concat([df_ode, df_lstm], ignore_index=True)
    
    # 2. Trasformazione Dati (Il Layer Interno -> What visibile)
    # Convertiamo i secondi in minuti per rendere gli assi facilmente leggibili
    df_all['mae_min'] = df_all['mae_s'] / 60.0
    df_all['sigma_min'] = df_all['sigma_s'] / 60.0
    
    # 3. Impostazione Estetica
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 8), dpi=300)
    
    # --- CREAZIONE DELLO SCATTERPLOT ---
    sns.scatterplot(
        data=df_all, 
        x='mae_min', 
        y='sigma_min', 
        hue='Modello', 
        palette=['#3498db', '#e74c3c'],  # Blu per ODE, Rosso per LSTM
        edgecolor='w',                   # Bordo bianco intorno ai pallini per staccarli
        s=100,                           # Dimensione dei pallini
        alpha=0.75                       # Leggera trasparenza per vedere i pallini sovrapposti
    )
    
    # --- AGGIUNTA DEI "MIRINI" (Le Medie Globali) ---
    # Calcoliamo il baricentro (media) di ogni modello
    mean_ode_mae = df_ode['mae_s'].mean() / 60.0
    mean_ode_sigma = df_ode['sigma_s'].mean() / 60.0
    
    mean_lstm_mae = df_lstm['mae_s'].mean() / 60.0
    mean_lstm_sigma = df_lstm['sigma_s'].mean() / 60.0
    
    # Disegniamo il mirino Blu per l'ODE
    plt.axvline(x=mean_ode_mae, color='#3498db', linestyle='--', alpha=0.5, linewidth=2)
    plt.axhline(y=mean_ode_sigma, color='#3498db', linestyle='--', alpha=0.5, linewidth=2)
    
    # Disegniamo il mirino Rosso per l'LSTM
    plt.axvline(x=mean_lstm_mae, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=2)
    plt.axhline(y=mean_lstm_sigma, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=2)
    
    # 4. Testi e Titoli (Storytelling)
    plt.title("Relazione tra Errore Medio (MAE) e Dispersione (Sigma)\nper Singola Linea Urbana", fontsize=16, weight='bold', pad=15)
    
    # Spieghiamo gli assi in modo chiaro anche per chi legge
    plt.xlabel("Errore Assoluto Medio (MAE) in Minuti\n(Distanza dalla Perfezione)", fontsize=12, weight='bold')
    plt.ylabel("Deviazione Standard (Sigma) in Minuti\n(Incertezza/Instabilità del Modello)", fontsize=12, weight='bold')
    
    plt.legend(title='Architettura di Rete', title_fontsize='13', fontsize='11', loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 5. Salvataggio
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Scatterplot salvato con successo come: {output_name}")

if __name__ == "__main__":
    FILE_ODE = "route_metrics_bus_model_mse_ODE.csv"
    FILE_LSTM = "route_metrics_bus_model_mse_99 LSTM.csv"
    
    plot_mae_vs_sigma_scatter(FILE_ODE, FILE_LSTM, "scatterplot_mae_sigma.png")