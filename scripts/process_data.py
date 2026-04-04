import cv2
import pandas as pd
import os

# konfiguracja
BASE_DATASET_PATH = "data/dataset"
PARTICIPANTS = ["p1", "p2a", "p2b", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
OUTPUT_FOLDER = "full-data"
LABELS_CSV = "labels.csv"
TARGET_SIZE = (224, 224)
FRAME_STEP = 15

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def parse_hold_usage(csv_path):
    #przetwarzanie pliku csv
    df = pd.read_csv(csv_path, sep=',', header=None, dtype=str)
    
    df.columns = ['limb', 'startFrame', 'endFrame'] + [f'extra_{i}' for i in range(len(df.columns)-3)]

    valid_limbs = ['lh', 'rh', 'lf', 'rf']
    df = df[df['limb'].str.contains('|'.join(valid_limbs), case=False, na=False)].copy()

    df['startFrame'] = pd.to_numeric(df['startFrame'], errors='coerce').fillna(0).astype(int)
    df['endFrame'] = pd.to_numeric(df['endFrame'], errors='coerce').fillna(0).astype(int)
    df['limb'] = df['limb'].str.strip().str.lower()
    
    return df

def get_4_bits(frame_idx, usage_df):
    bits = [0, 0, 0, 0]
    limbs = ['lh', 'rh', 'lf', 'rf']
    
    for i, limb_code in enumerate(limbs):
        #musi być contains bo w csv np. 'lf 1'
        mask = (
            (usage_df['limb'].str.contains(limb_code, na=False)) & 
            (frame_idx >= usage_df['startFrame']) & 
            (frame_idx <= usage_df['endFrame'])
        )
        
        if mask.any():
            bits[i] = 1
    return bits

all_data = []

for p_id in PARTICIPANTS:
    current_p_path = os.path.join(BASE_DATASET_PATH, p_id)
    
    if not os.path.exists(current_p_path):
        print(f"Pominięto {p_id} - folder nie istnieje.")
        continue

    video_files = [f for f in os.listdir(current_p_path) if f.endswith('.mp4') and 'large' not in f]
    
    for video_name in video_files:
        route_name = video_name.replace('.mp4', '')
        video_path = os.path.join(current_p_path, video_name)
        csv_path = os.path.join(current_p_path, f"{route_name}_holdUsage.csv")
        
        if not os.path.exists(csv_path):
            continue

        usage_df = parse_hold_usage(csv_path)
        cap = cv2.VideoCapture(video_path)
        
        frame_idx = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            if frame_idx % FRAME_STEP == 0:
                bits = get_4_bits(frame_idx, usage_df)
                resized_frame = cv2.resize(frame, TARGET_SIZE)
                
                # nazwa pliku = id uczestnika + nazwa trasy + numer klatki
                img_name = f"{p_id}_{route_name}_f{frame_idx:05d}.jpg"
                img_path = os.path.join(OUTPUT_FOLDER, img_name)
                cv2.imwrite(img_path, resized_frame)
                
                all_data.append([img_name] + bits)
                saved_count += 1
            frame_idx += 1
            
        cap.release()
        print(f"{p_id} | {video_name}: zapisano {saved_count} klatek")

#tutaj zapis na google drive
df_final = pd.DataFrame(all_data, columns=['filename', 'lh', 'rh', 'lf', 'rf'])
df_final.to_csv(LABELS_CSV, index=False)
print(f"\nZapisano {len(df_final)} klatek")