import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_bus_type_predictor(parquet_path: str, model_save_path: str):
    print("Inizializzazione della pipeline LightGBM...")
    
    # 1. CARICAMENTO DATI
    # Assicurati di puntare al dataset pulito (quello dal 22 Febbraio 2026 in poi!)
    print(f"Caricamento dati da {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=['bus_type'])
    
    # 2. FEATURE ENGINEERING (Il Layer Interno)
    # Selezioniamo solo le colonne che sappiamo a priori prima che il viaggio inizi
    features = ['route_id', 'time_sin', 'time_cos', 'day_type']
    target = 'bus_type'
    
    # Assicurati che le colonne esistano nel tuo dataframe, altrimenti adattale
    # Estraiamo X e y
    X = df[features].copy()
    y = df[target].copy()
    y = y.astype(int)
    
    # TRUCCO MAGICO DI LIGHTGBM: Convertiamo le stringhe/ID in tipo 'category'
    # Così l'albero capisce che la linea 170 non è "maggiore" della linea 85, sono solo etichette diverse.
    categorical_features = ['route_id', 'day_type']
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    # 3. SPLIT DEL DATASET
    # Teniamo il 20% dei dati per validare se il modello sta imparando o barando
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. CONFIGURAZIONE DEL MODELLO (Il Layer Esterno)
    print("Addestramento del Gradient Boosting Decision Tree...")
    clf = lgb.LGBMClassifier(
        n_estimators=150,        # Numero di alberi nella foresta
        learning_rate=0.1,       # Quanto velocemente impara
        max_depth=7,             # Profondità massima dell'albero (evita l'overfitting)
        random_state=42,
        n_jobs=-1                # Usa tutti i core della tua CPU
    )
    
    # 5. ADDESTRAMENTO
    clf.fit(X_train, y_train)
    
    # 6. VALIDAZIONE (Il What)
    print("Valutazione delle performance...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Accuratezza Globale: {acc * 100:.2f}%\n")
    
    # Stampiamo il report dettagliato per ogni tipo di bus
    print("Report di Classificazione:")
    print(classification_report(y_test, y_pred))
    
    # 7. SALVATAGGIO
    joblib.dump(clf, model_save_path)
    print(f"[+] Modello salvato con successo in: {model_save_path}")

if __name__ == "__main__":
    # Inserisci i percorsi corretti del tuo ambiente
    PERCORSO_DATASET = "dataset_lstm_final.parquet" 
    PERCORSO_SALVATAGGIO = "bus_type_predictor.pkl"
    
    train_bus_type_predictor(PERCORSO_DATASET, PERCORSO_SALVATAGGIO)
