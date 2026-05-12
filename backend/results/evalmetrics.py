import numpy as np

def parse_confusion_matrix(filepath):
    """
    Catena Estetica/Funzionale 1: Trasforma il testo (Segnale) in Dati (Sistema).
    """
    matrix_data = []
    class_names = []
    capture = False
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Condizione di innesco: troviamo l'inizio della matrice
            if line.startswith("Actual:"):
                capture = True
                continue
            
            # Se stiamo catturando e la riga è vuota (o contiene altro), fermati
            if capture:
                if not line or line.startswith("==="):
                    break
                
                # Scomponiamo la riga (es: "EMPTY   368160   3225 ...")
                parts = line.split()
                if len(parts) > 1:
                    class_name = parts[0]
                    # Convertiamo tutti i valori successivi in interi
                    values = [int(x) for x in parts[1:]]
                    
                    class_names.append(class_name)
                    matrix_data.append(values)
                    
    # Trasformiamo la lista in un tensore Numpy per sfruttare la matematica vettoriale
    return np.array(matrix_data), class_names


def calculate_and_print_metrics(matrix, class_names):
    """
    Catena Estetica/Funzionale 2: Estrae il significato matematico.
    """
    # Atomi Base (One-vs-All)
    tp = np.diag(matrix)                  # La diagonale (Veri Positivi)
    actual_totals = np.sum(matrix, axis=1) # Somma delle righe (Eventi Reali = TP + FN)
    predicted_totals = np.sum(matrix, axis=0) # Somma delle colonne (Predizioni = TP + FP)
    
    total_events = np.sum(matrix)
    
    # 1. ACCURACY GLOBALE
    accuracy = np.sum(tp) / total_events if total_events > 0 else 0
    print("=" * 50)
    print(f"ACCURACY GLOBALE DEL SISTEMA: {accuracy:.4%} ({np.sum(tp)}/{total_events})")
    print("=" * 50)
    
    # 2. METRICHE PER CLASSE (Who & Why locale)
    for i, class_name in enumerate(class_names):
        support = actual_totals[i] # Quante volte questa classe è avvenuta davvero
        
        # Evitiamo divisioni per zero
        recall = tp[i] / support if support > 0 else 0.0
        precision = tp[i] / predicted_totals[i] if predicted_totals[i] > 0 else 0.0
        
        f1_score = 0.0
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        print(f"Classe: {class_name: <8} | Supporto Reale: {support}")
        print(f"  - Recall:    {recall:.2%}")
        print(f"  - Precision: {precision:.2%}")
        print(f"  - F1-Score:  {f1_score:.2%}")
        print("-" * 30)

# Punto di ingresso dello script
if __name__ == "__main__":
    # Inserisci qui il nome del tuo file report
    file_path = "validation_live_20260311_report.txt" 
    
    try:
        matrice, classi = parse_confusion_matrix(file_path)
        if len(matrice) > 0:
            calculate_and_print_metrics(matrice, classi)
        else:
            print("Matrice non trovata. Controlla il formato del file.")
    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato nella cartella corrente.")