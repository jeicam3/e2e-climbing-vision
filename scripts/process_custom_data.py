import os
import cv2
import pandas as pd
from tqdm import tqdm

# --- KONFIGURACJA ŚCIEŻEK ---

# 1. Lokalizacja tego skryptu (np. /TwójProjekt/scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Główny folder projektu (jeden poziom wyżej niż 'scripts', czyli /TwójProjekt)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo-dataset1")
# 3. Ścieżki wyjściowe (powstaną na tym samym poziomie co folder 'scripts')
OUTPUT_IMG_DIR = os.path.join(DATASET_DIR, "full-data")
OUTPUT_CSV = os.path.join(DATASET_DIR, "labels.csv")

# 4. Ścieżki do Twojego nowego zbioru
# Zakładam, że custom_dataset masz w folderze 'data' w głównym projekcie
# Jeśli masz go gdzie indziej, np. bezpośrednio w PROJECT_ROOT, usuń "data" z os.path.join
CUSTOM_DIR = os.path.join(PROJECT_ROOT, "data", "custom_dataset") 
VIDEOS_DIR = os.path.join(CUSTOM_DIR, "videos_yolo")
LABELS_DIR = os.path.join(CUSTOM_DIR, "labels")
STATUS_CSV = os.path.join(CUSTOM_DIR, "labeling_status.csv")

# Parametr wyciągania klatek
FRAME_STEP = 5

def main():
    # Upewnij się, że docelowy folder obrazów istnieje
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    # 1. Odczytanie statusu
    if not os.path.exists(STATUS_CSV):
        print(f"❌ Nie znaleziono pliku statusu: {STATUS_CSV}")
        return

    status_df = pd.read_csv(STATUS_CSV)
    
    # Bezpieczne wyczyszczenie stringów i szukanie gotowych filmów (100%)
    #status_df['progress'] = status_df['progress'].astype(str).str.strip().str.replace('%', '')
    completed_videos = status_df[status_df['labeled'] == 1]['video'].tolist()

    if not completed_videos:
        print("⚠️ Brak filmów oznaczonych w 100%. Sprawdź plik labeling_status.csv.")
        return

    print(f"✅ Znaleziono {len(completed_videos)} gotowych filmów do przetworzenia: {completed_videos}")

    new_records = []

    # 2. Przetwarzanie wyselekcjonowanych filmów
    for video_name in completed_videos:
        video_base = os.path.splitext(video_name)[0] # np. "IMG_0894"
        video_path = os.path.join(VIDEOS_DIR, f"{video_base}.mp4")
        label_path = os.path.join(LABELS_DIR, f"{video_base}_labels.csv")

        if not os.path.exists(video_path):
            print(f"⚠️ Pomijam. Brak pliku wideo: {video_path}")
            continue
        if not os.path.exists(label_path):
            print(f"⚠️ Pomijam. Brak pliku etykiet: {label_path}")
            continue

        labels_df = pd.read_csv(label_path)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Przetwarzanie wideo: {video_name} ({total_frames} klatek)...")

        # Pętla po wszystkich klatkach z paskiem postępu tqdm
        for frame_idx in tqdm(range(total_frames), desc=f"Zapisywanie klatek {video_base}", leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            # Bierzemy pod uwagę tylko co FRAME_STEP-tą klatkę
            if frame_idx % FRAME_STEP == 0:
                # Szukamy, czy w pliku customowego labelingu jest oznaczona ta klatka
                label_row = labels_df[labels_df['frame'] == frame_idx]

                if not label_row.empty:
                    lh = int(label_row.iloc[0]['left_hand'])
                    rh = int(label_row.iloc[0]['right_hand'])
                    lf = int(label_row.iloc[0]['left_foot'])
                    rf = int(label_row.iloc[0]['right_foot'])

                    # Nazwa zgodna z formatem (np. IMG_0894_frame_0020.jpg)
                    img_filename = f"{video_base}_frame_{frame_idx:04d}.jpg"
                    img_path = os.path.join(OUTPUT_IMG_DIR, img_filename)

                    # Zapis pliku fizycznego
                    cv2.imwrite(img_path, frame)

                    # Dodanie do listy pamięciowej
                    new_records.append({
                        'frame_id': img_filename,
                        'left_hand': lh,
                        'right_hand': rh,
                        'left_foot': lf,
                        'right_foot': rf
                    })

        cap.release()

    # 3. Dodawanie zebranych rekordów do ostatecznego CSV
    if new_records:
        new_df = pd.DataFrame(new_records)
        
        # Sprawdzenie, czy oryginalny procesor (process_data.py) już utworzył plik docelowy
        if os.path.exists(OUTPUT_CSV):
            # Tryb 'a' = append (dopisywanie na koniec pliku). Brak nagłówka (header=False).
            new_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
            print(f"🎉 Sukces! Złączono datasety. Dopisano {len(new_records)} nowych klatek do {OUTPUT_CSV}.")
        else:
            # Tworzymy nowy plik z nagłówkami
            new_df.to_csv(OUTPUT_CSV, mode='w', header=True, index=False)
            print(f"🎉 Utworzono {OUTPUT_CSV} i zapisano do niego {len(new_records)} klatek.")
    else:
        print("⚠️ Nie wyekstrahowano żadnych klatek. Upewnij się, że frame_index w Twoich CSV pokrywa się ze skokiem (FRAME_STEP).")

if __name__ == "__main__":
    main()