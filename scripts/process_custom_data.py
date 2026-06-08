import os
import cv2
import pandas as pd
from tqdm import tqdm

# --- KONFIGURACJA ŚCIEŻEK ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- NOWE ŚCIEŻKI I PARAMETRY WYCINANIA ---
OUTPUT_IMG_DIR = os.path.join(PROJECT_ROOT, "cropped_dataset", "full-data")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "cropped_dataset", "labels.csv")
BBOX_DIR = os.path.join(PROJECT_ROOT, "climber_bboxes")

CUSTOM_DIR = os.path.join(PROJECT_ROOT, "data", "custom_dataset") 
VIDEOS_DIR = os.path.join(CUSTOM_DIR, "videos")
LABELS_DIR = os.path.join(CUSTOM_DIR, "labels")
STATUS_CSV = os.path.join(CUSTOM_DIR, "labeling_status.csv")

FRAME_STEP = 5
CROP = True
MARGIN_RATIO = 0.15

# --- NOWE FUNKCJE ---
def load_bboxes(csv_path):
    bboxes = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            val = str(row['x1']).strip().upper()
            if val != 'NULL' and pd.notna(row['x1']):
                bboxes[int(row['frame'])] = (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']))
    return bboxes

def crop_and_pad_frame(frame, bbox, margin=0.15):
    if bbox is None:
        return frame
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin
    
    crop_x1 = max(0, int(x1 - mx))
    crop_y1 = max(0, int(y1 - my))
    crop_x2 = min(w_img, int(x2 + mx))
    crop_y2 = min(h_img, int(y2 + my))
    
    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    ch, cw = cropped.shape[:2]
    if ch <= 0 or cw <= 0:
        return frame
        
    max_dim = max(ch, cw)
    top = (max_dim - ch) // 2
    bottom = max_dim - ch - top
    left = (max_dim - cw) // 2
    right = max_dim - cw - left
    return cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# --- GŁÓWNA PĘTLA ---
def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    if not os.path.exists(STATUS_CSV):
        print(f"❌ Nie znaleziono pliku statusu: {STATUS_CSV}")
        return

    status_df = pd.read_csv(STATUS_CSV)
    completed_videos = status_df[status_df['labeled'] == 1]['video'].tolist()

    if not completed_videos:
        print("⚠️ Brak filmów oznaczonych w 100%. Sprawdź plik labeling_status.csv.")
        return

    print(f"✅ Znaleziono {len(completed_videos)} gotowych filmów do przetworzenia: {completed_videos}")
    new_records = []

    for video_name in completed_videos:
        video_base = os.path.splitext(video_name)[0]
        video_path = os.path.join(VIDEOS_DIR, video_name)
        label_path = os.path.join(LABELS_DIR, f"{video_base}_labels.csv")

        # Wczytanie bboxów: nazwa pliku to np. IMG_0894.csv
        bbox_path = os.path.join(BBOX_DIR, f"{video_base}.csv")
        bboxes = load_bboxes(bbox_path) if CROP else {}

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

        for frame_idx in tqdm(range(total_frames), desc=f"Zapisywanie klatek {video_base}", leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP == 0:
                label_row = labels_df[labels_df['frame'] == frame_idx]

                if not label_row.empty:
                    lh = int(label_row.iloc[0]['left_hand'])
                    rh = int(label_row.iloc[0]['right_hand'])
                    lf = int(label_row.iloc[0]['left_foot'])
                    rf = int(label_row.iloc[0]['right_foot'])

                    # WYCINANIE BBOXA (JEŚLI CROP == TRUE)
                    if CROP and frame_idx in bboxes:
                        frame = crop_and_pad_frame(frame, bboxes[frame_idx], MARGIN_RATIO)

                    img_filename = f"{video_base}_frame_{frame_idx:04d}.jpg"
                    img_path = os.path.join(OUTPUT_IMG_DIR, img_filename)

                    cv2.imwrite(img_path, frame)

                    new_records.append({
                        'frame_id': img_filename,
                        'left_hand': lh,
                        'right_hand': rh,
                        'left_foot': lf,
                        'right_foot': rf
                    })

        cap.release()

    if new_records:
        new_df = pd.DataFrame(new_records)
        if os.path.exists(OUTPUT_CSV):
            new_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
            print(f"🎉 Sukces! Złączono datasety. Dopisano {len(new_records)} nowych klatek do {OUTPUT_CSV}.")
        else:
            new_df.to_csv(OUTPUT_CSV, mode='w', header=True, index=False)
            print(f"🎉 Utworzono {OUTPUT_CSV} i zapisano do niego {len(new_records)} klatek.")
    else:
        print("⚠️ Nie wyekstrahowano żadnych klatek. Upewnij się, że frame_index w Twoich CSV pokrywa się ze skokiem (FRAME_STEP).")

if __name__ == "__main__":
    main()