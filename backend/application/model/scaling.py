import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PARQUET_DIR = os.path.join(PROJECT_ROOT, "parquets")


def scaling(input_parquet: str, output_parquet: str, scaler_path: str):
    print("Reading Dataset...")
    df: pd.DataFrame = pd.read_parquet(input_parquet)

    # 1. Definiamo quali colonne dense scalare (devono coincidere con il tuo DataLoader)
    x1_dense_cols = [
        'deposit_grottarossa', 'deposit_magliana', 'deposit_tor_sapienza', 
        'deposit_portonaccio', 'deposit_monte_sacro', 'deposit_tor_pagnotta', 
        'deposit_tor_cervara', 'deposit_maglianella', 'deposit_costi', 
        'deposit_trastevere', 'deposit_acilia', 'deposit_tor_vergata', 
        'deposit_porta_maggiore', 'door_number', 'scheduled_start_time_sin', 
        'scheduled_start_time_cos'
    ]
    
    x2_dense_cols = ['can_shape_dist_travelled', 'can_distance_to_next_stop', 'segment_idx']
    
    all_feature_cols = x1_dense_cols + x2_dense_cols

    # La label di regressione che il modello deve prevedere
    target_col = ['schedule_adherence']
    
    # NOTA: occupancy_status non va scalata perché è una categoria per la classificazione!

    print("Initializing Scalers...")
    # Creiamo due scaler separati
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Inizializziamo il LabelEncoder per la linea
    route_encoder = LabelEncoder() 

    print("Encoding categorical strings...")
    # Trasforma "120F", "170" ecc. in interi 0, 1, 2...
    df['route_id'] = route_encoder.fit_transform(df['route_id'])

    print("Scaling input features...")
    # Scaliamo tutte le feature dense in un colpo solo
    df[all_feature_cols] = feature_scaler.fit_transform(df[all_feature_cols])

    print("Scaling target (schedule_adherence)...")
    # reshape(-1, 1) è richiesto da sklearn quando si scala una singola colonna
    df[target_col] = target_scaler.fit_transform(df[target_col].values.reshape(-1, 1))

    print("Saving scalers...")
    joblib.dump({
        'features': feature_scaler,
        'target': target_scaler,
        'route': route_encoder # <-- Salvato!
    }, scaler_path)

    print("Saving scaled dataset...")
    df.to_parquet(output_parquet)

    print("Done.")


def descaling(value: float, scaler_path: str):
    # Carichiamo il dizionario contenente gli scaler
    scalers = joblib.load(scaler_path)
    
    # Estraiamo SOLO lo scaler che era stato allenato sulla label
    target_scaler = scalers['target']

    # Effettuiamo la trasformazione inversa per ottenere i secondi reali [5, 6]
    prediction_matrix = np.array([[value]])
    real_prediction = target_scaler.inverse_transform(prediction_matrix)

    return real_prediction


if __name__ == "__main__":
    scaling(
        input_parquet=os.path.join(PARQUET_DIR, "dataset_lstm_unscaled.parquet"),
        output_parquet=os.path.join(PARQUET_DIR, "dataset_lstm_final.parquet"),
        scaler_path=os.path.join(PARQUET_DIR, "lstm_scalers.pkl"), # Cambiato nome al plurale
    )
