import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PARQUET_DIR = os.path.join(PROJECT_ROOT, "parquets")


def scaling(input_parquet: str, output_parquet: str, encoder_path: str):
    print("Reading Dataset...")
    df = pd.read_parquet(input_parquet)

    # ==========================================
    # 0. CHIRURGIA DEGLI OUTLIER (Salva il Reshape)
    # ==========================================
    print("Individuazione anomalie nei ritardi...")
    # Troviamo i viaggi in cui c'è almeno un ritardo/anticipo maggiore di 1 ora
    viaggi_anomali = df[
        (df['schedule_adherence'] > 3600) | (df['schedule_adherence'] < -3600)
    ]['trip_id'].unique()

    print(f"Rimozione di {len(viaggi_anomali)} viaggi interi per guasti/anomalie...")
    # Teniamo solo i viaggi sani, per non rompere i blocchi da 100 step
    df = df[~df['trip_id'].isin(viaggi_anomali)].copy()
    df.reset_index(drop=True, inplace=True)

    # ==========================================
    # 1. ENCODING CATEGORICO
    # ==========================================
    print("Encoding categorical strings...")
    route_encoder = LabelEncoder()
    df['route_id'] = route_encoder.fit_transform(df['route_id'])

    # ==========================================
    # 2. SCALING FISICO / ARTIGIANALE
    # ==========================================
    print("Applicazione dello scaling logico...")
    
    # A. Gradini Fissi (Normalizzati tra 0 e 1)
    df['door_number'] = df['door_number'].clip(1, 3) / 3.0
    df['segment_idx'] = df['segment_idx'] / 99.0
    df['bus_type'] = df['bus_type'] / 9.0
    
    # B. Cinematica Spaziale (Taglio degli outlier GPS e compressione 0-1)
    df['can_shape_dist_travelled'] = df['can_shape_dist_travelled'].clip(0, 15000) / 15000.0
    df['can_distance_to_next_stop'] = df['can_distance_to_next_stop'].clip(0, 1000) / 1000.0
    df['can_avg_traffic_speed'] = df['can_avg_traffic_speed'].clip(0, 65) / 65.0
    
    # C. Il Target (Ritardi tra -1.0 e +1.0)
    df['schedule_adherence'] = df['schedule_adherence'] / 3600.0

    # NOTA: I depositi (0 o 1) e i time_sin/cos (-1 a 1) NON si toccano!

    # ==========================================
    # 3. SALVATAGGIO
    # ==========================================
    print("Salvataggio dell'Encoder della Linea...")
    # Non serve più salvare feature_scaler e target_scaler!
    joblib.dump({'route': route_encoder}, encoder_path)

    print("Salvataggio dataset pulito e scalato...")
    df.to_parquet(output_parquet)
    print("Fatto.")


def descaling(value: float, scaler_path: str = None):
    """
    La decodifica ora è pura matematica. Se la rete sputa 0.1, 
    sono 360 secondi di ritardo.
    """
    return value * 3600.0


if __name__ == "__main__":
    scaling(
        input_parquet=os.path.join(PARQUET_DIR, "dataset_lstm_unscaled.parquet"),
        output_parquet=os.path.join(PARQUET_DIR, "dataset_lstm_final.parquet"),
        encoder_path=os.path.join(PARQUET_DIR, "route_encoder.pkl"), # Cambiato nome per chiarezza
    )
