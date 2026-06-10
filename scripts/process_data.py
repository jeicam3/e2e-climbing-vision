import cv2
import pandas as pd
import os

# konfiguracja
BASE_DATASET_PATH = "data/dataset"
PARTICIPANTS = ["p1", "p2a", "p2b", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
TARGET_SIZE = (224, 224)
FRAME_STEP = 20

# --- NOWE PARAMETRY WYCINANIA ---
CROP = True
MARGIN_RATIO = 0.15
BBOX_DIR = "../climber_bboxes" # Ścieżka relatywna od skryptu
OUTPUT_FOLDER = "../cropped_dataset/full-data"
LABELS_CSV = "../cropped_dataset/labels.csv"

# Wymuś ścieżki absolutne z poziomu skryptu
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BBOX_DIR = os.path.join(PROJECT_ROOT, "climber_bboxes")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "cropped_dataset", "full-data")
LABELS_CSV = os.path.join(PROJECT_ROOT, "cropped_dataset", "labels.csv")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

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

# --- STARE FUNKCJE ---
def parse_hold_usage(csv_path):
    df = pd.read_csv(csv_path, sep=',', header=None, dtype=str)
    df.columns = ['limb', 'startFrame', 'endFrame'] + [f'extra_{i}' for i in range(len(df.columns)-3)]
    valid_limbs = ['lh', 'rh', 'lf', 'rf']
    df = df[df['limb'].str.contains('|'.join(valid_limbs), case=False, na=False)].copy()
    df['startFrame'] = pd.to_numeric(df['startFrame'], errors='coerce').fillna(0).astype(int)
    df['endFrame'] = pd.to_numeric(df['endFrame'], errors='coerce').fillna(0).astype(int)
    return df

def get_4_bits(frame_idx, usage_df):
    bits = [0, 0, 0, 0]
    limbs = ['lh', 'rh', 'lf', 'rf']
    for i, limb in enumerate(limbs):
        active = usage_df[(usage_df['limb'].str.lower() == limb) & 
                          (usage_df['startFrame'] <= frame_idx) & 
                          (usage_df['endFrame'] >= frame_idx)]
        if not active.empty:
            bits[i] = 1
    return bits

all_data = []

# --- GŁÓWNA PĘTLA ---
for p_id in PARTICIPANTS:
    current_p_path = os.path.join(PROJECT_ROOT, BASE_DATASET_PATH, p_id)
    if not os.path.exists(current_p_path):
        continue
        
    video_files = [f for f in os.listdir(current_p_path) if f.endswith('.mp4') and 'large' not in f]
    
    for video_name in video_files:
        route_name = video_name.replace('.mp4', '')
        video_path = os.path.join(current_p_path, video_name)
        csv_path = os.path.join(current_p_path, f"{route_name}_holdUsage.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        # Wczytanie bboxów: nazwa pliku to np. p1_orange.csv
        bbox_csv_name = f"{p_id}_{route_name}.csv"
        bbox_path = os.path.join(BBOX_DIR, bbox_csv_name)
        bboxes = load_bboxes(bbox_path) if CROP else {}

        usage_df = parse_hold_usage(csv_path)
        cap = cv2.VideoCapture(video_path)
        
        frame_idx = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            if frame_idx % FRAME_STEP == 0:
                bits = get_4_bits(frame_idx, usage_df)
                
                # WYCINANIE BBOXA (JEŚLI CROP == TRUE)
                if CROP and frame_idx in bboxes:
                    frame_to_process = crop_and_pad_frame(frame, bboxes[frame_idx], MARGIN_RATIO)
                else:
                    frame_to_process = frame
                    
                resized_frame = cv2.resize(frame_to_process, TARGET_SIZE)
                
                img_name = f"{p_id}_{route_name}_f{frame_idx:05d}.jpg"
                img_path = os.path.join(OUTPUT_FOLDER, img_name)
                cv2.imwrite(img_path, resized_frame)
                
                all_data.append([img_name] + bits)
                saved_count += 1
            frame_idx += 1
        cap.release()

final_df = pd.DataFrame(all_data, columns=['frame_id', 'left_hand', 'right_hand', 'left_foot', 'right_foot'])
final_df.to_csv(LABELS_CSV, index=False)
print("Zakończono process_data.py")