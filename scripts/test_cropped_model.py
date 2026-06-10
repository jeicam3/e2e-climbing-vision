import argparse
from bisect import bisect_right
import csv
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "cropped_model.pth"
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "dataset" / "IMG_0899.mp4"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "dataset" / "test_cropped.mp4"
DEFAULT_BBOX_DIR = PROJECT_ROOT / "data" / "climber_bboxes"

CLASS_NAMES = ["LH", "RH", "LF", "RF"]
MODEL_RESOLUTIONS = {
    "b0": 224,
    "b1": 240,
    "b2": 260,
}

BAR_WIDTH = 150
BAR_HEIGHT = 20
START_X = 20
START_Y = 300


def resolve_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_bboxes(csv_path):
    bboxes = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            normalized_row = {key.strip().lower(): value for key, value in row.items() if key is not None}
            raw_x1 = str(normalized_row.get("x1", "")).strip()
            if raw_x1.upper() in {"", "NULL", "NAN", "NONE"}:
                continue

            try:
                frame_idx = int(float(normalized_row["frame"]))
                bboxes[frame_idx] = (
                    float(normalized_row["x1"]),
                    float(normalized_row["y1"]),
                    float(normalized_row["x2"]),
                    float(normalized_row["y2"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

    return bboxes


def guess_bbox_path(video_path, bbox_dir):
    candidates = [
        bbox_dir / f"{video_path.stem}.csv",
        bbox_dir / f"{video_path.parent.name}_{video_path.stem}.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def crop_and_pad_frame(frame, bbox, margin=0.15):
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
        return None, None

    max_dim = max(ch, cw)
    top = (max_dim - ch) // 2
    bottom = max_dim - ch - top
    left = (max_dim - cw) // 2
    right = max_dim - cw - left
    padded = cv2.copyMakeBorder(
        cropped,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return padded, (crop_x1, crop_y1, crop_x2, crop_y2)


def build_model(model_name, dropout_rate):
    model_name = model_name.lower()
    creators = {
        "b0": models.efficientnet_b0,
        "b1": models.efficientnet_b1,
        "b2": models.efficientnet_b2,
    }
    model = creators[model_name](weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_ftrs, len(CLASS_NAMES)),
        nn.Sigmoid(),
    )
    return model


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_trained_model(path, model_name, dropout_rate, device):
    model = build_model(model_name, dropout_rate)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def build_preprocess(target_size):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def predict(cropped_frame, model, preprocess, device):
    rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    frame_t = preprocess(rgb_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(frame_t)
        probs = output.squeeze(0).detach().cpu().tolist()

    return probs


def visualize_probs(frame, probs, origin=(START_X, START_Y)):
    start_x, start_y = origin
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        y = start_y + i * 30
        cv2.rectangle(frame, (start_x, y), (start_x + BAR_WIDTH, y + BAR_HEIGHT), (50, 50, 50), -1)

        color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)
        current_width = int(BAR_WIDTH * max(0.0, min(1.0, prob)))
        cv2.rectangle(frame, (start_x, y), (start_x + current_width, y + BAR_HEIGHT), color, -1)

        text = f"{name}: {int(prob * 100)}%"
        cv2.putText(
            frame,
            text,
            (start_x + BAR_WIDTH + 10, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


def draw_crop_info(frame, crop_rect, bbox_source):
    if crop_rect is None:
        return

    x1, y1, x2, y2 = crop_rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.putText(
        frame,
        f"crop bbox: {bbox_source}",
        (max(10, x1), max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 200, 0),
        2,
    )


def draw_crop_preview(frame, cropped_frame, size=160):
    if cropped_frame is None:
        return

    frame_h, frame_w = frame.shape[:2]
    if frame_w < size + 30 or frame_h < size + 30:
        return

    preview = cv2.resize(cropped_frame, (size, size))
    x1 = frame_w - size - 20
    y1 = 20
    x2 = x1 + size
    y2 = y1 + size

    frame[y1:y2, x1:x2] = preview
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(
        frame,
        "model input",
        (x1, y2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def draw_status(frame, text):
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 255),
        2,
    )


def get_bbox_for_frame(frame_idx, bboxes, bbox_frames, missing_bbox_mode):
    bbox = bboxes.get(frame_idx)
    if bbox is not None:
        return bbox, str(frame_idx)

    if missing_bbox_mode == "previous":
        previous_pos = bisect_right(bbox_frames, frame_idx) - 1
        if previous_pos >= 0:
            previous_idx = bbox_frames[previous_pos]
            return bboxes[previous_idx], f"{previous_idx} -> {frame_idx}"

    return None, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual test for a climbing model using cropped climber bboxes as model input."
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to .pth model weights.")
    parser.add_argument("--video-path", default=str(DEFAULT_VIDEO_PATH), help="Path to the input video.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Path for the annotated output video.")
    parser.add_argument("--bbox-csv", default=None, help="Path to bbox CSV. If omitted, it is guessed from the video name.")
    parser.add_argument("--bbox-dir", default=str(DEFAULT_BBOX_DIR), help="Directory with bbox CSV files.")
    parser.add_argument("--model-name", choices=MODEL_RESOLUTIONS.keys(), default="b2", help="EfficientNet variant.")
    parser.add_argument("--dropout-rate", type=float, default=0.6, help="Dropout used in the classifier head.")
    parser.add_argument("--margin-ratio", type=float, default=0.15, help="Crop margin ratio, same meaning as in process_data.py.")
    parser.add_argument(
        "--missing-bbox",
        choices=["skip", "previous", "full"],
        default="skip",
        help="What to do when a frame has no bbox. 'skip' avoids full-frame inference.",
    )
    parser.add_argument("--no-crop-preview", action="store_true", help="Do not draw the model-input crop preview.")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = resolve_path(args.model_path)
    video_path = resolve_path(args.video_path)
    output_path = resolve_path(args.output_path)
    bbox_dir = resolve_path(args.bbox_dir)
    bbox_path = resolve_path(args.bbox_csv) if args.bbox_csv else guess_bbox_path(video_path, bbox_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not bbox_path.exists():
        raise FileNotFoundError(
            f"Bbox file not found: {bbox_path}. Pass it with --bbox-csv or --bbox-dir."
        )

    bboxes = load_bboxes(bbox_path)
    if not bboxes:
        raise ValueError(f"Bbox file has no valid frame rows: {bbox_path}")
    bbox_frames = sorted(bboxes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_size = MODEL_RESOLUTIONS[args.model_name]
    preprocess = build_preprocess(target_size)
    model = load_trained_model(model_path, args.model_name, args.dropout_rate, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    frame_idx = 0
    predicted_frames = 0
    skipped_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bbox, bbox_source = get_bbox_for_frame(frame_idx, bboxes, bbox_frames, args.missing_bbox)

        if bbox is None and args.missing_bbox == "full":
            cropped_frame = frame
            crop_rect = None
            bbox_source = "full frame"
        elif bbox is None:
            cropped_frame = None
            crop_rect = None
        else:
            cropped_frame, crop_rect = crop_and_pad_frame(frame, bbox, args.margin_ratio)

        if cropped_frame is None:
            skipped_frames += 1
            draw_status(frame, f"frame {frame_idx}: no bbox - prediction skipped")
        else:
            probs = predict(cropped_frame, model, preprocess, device)
            predicted_frames += 1
            draw_crop_info(frame, crop_rect, bbox_source)
            if not args.no_crop_preview:
                draw_crop_preview(frame, cropped_frame)
            visualize_probs(frame, probs)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Saved output: {output_path}")
    print(f"Predicted {predicted_frames} frames; skipped {skipped_frames} frames without bbox.")


if __name__ == "__main__":
    main()
