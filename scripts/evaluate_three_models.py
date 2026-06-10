from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CLASS_NAMES = ["LH", "RH", "LF", "RF"]
LABEL_COLUMNS = ["left_hand", "right_hand", "left_foot", "right_foot"]

TEST_VIDEO_DIR = PROJECT_ROOT / "data" / "test_videos"
RAW_VIDEO_DIR = TEST_VIDEO_DIR / "raw"
MASKED_VIDEO_DIR = TEST_VIDEO_DIR / "masked"
LABEL_DIR = TEST_VIDEO_DIR / "labels"
BBOX_DIR = TEST_VIDEO_DIR / "bboxes"
FALLBACK_BBOX_DIR = PROJECT_ROOT / "data" / "climber_bboxes"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation_results"
VIDEO_EXTENSIONS = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"]

# Default test videos for models without their own test_videos list. Names may
# include or omit the extension. Raw videos are matched in data/test_videos/raw
# by stem, so IMG_0898.MOV and IMG_0898 both resolve to the same test sample.
TEST_VIDEO_FILES = [
    "IMG_0898.MOV",
    "IMG_0912.MOV",
]

# Fill in model_path before running the final evaluation. All paths may be
# absolute or relative to the repository root.
MODEL_CONFIGS = [
    {
        "name": "e2e",
        "description": "Full-frame EfficientNet baseline",
        "model_path": "checkpoints/two_stage_b2_run.pth",
        "input_mode": "raw",
        "model_name": "b2",
        "dropout_rate": 0.6,
        "test_videos": [
            "p9_orange",
            "p9_green",
            "p4_orange",
            "p4_green",
            "IMG_0903",
            "IMG_0895",
            "p5_orange",
            "p5_green",
            "p10_green",
            "p10_orange",
        ],
    },
    {
        "name": "masks_bbox",
        "description": "Frames with hold masks and climber bbox overlay",
        "model_path": "checkpoints/yolo_climbing_model.pth",
        "input_mode": "masked",
        "model_name": "b2",
        "dropout_rate": 0.6,
        "test_videos": [
            "IMG_0903",
            "IMG_0899",
        ],
    },
    {
        "name": "bbox_crop",
        "description": "Climber crop based on bbox CSV",
        "model_path": "checkpoints/last.pth",
        "input_mode": "crop",
        "model_name": "b2",
        "dropout_rate": 0.6,
        "test_videos": TEST_VIDEO_FILES,
    },
]

MODEL_RESOLUTIONS = {
    "b0": 224,
    "b1": 240,
    "b2": 260,
}


@dataclass
class EvalExample:
    video_name: str
    frame_idx: int
    labels: list[int] | None


class BinaryStats:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, truth, pred):
        if truth == 1 and pred == 1:
            self.tp += 1
        elif truth == 0 and pred == 0:
            self.tn += 1
        elif truth == 0 and pred == 1:
            self.fp += 1
        elif truth == 1 and pred == 0:
            self.fn += 1

    @property
    def total(self):
        return self.tp + self.tn + self.fp + self.fn

    def accuracy(self):
        return safe_div(self.tp + self.tn, self.total)

    def precision(self):
        return safe_div(self.tp, self.tp + self.fp)

    def recall(self):
        return safe_div(self.tp, self.tp + self.fn)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return safe_div(2 * p * r, p + r)


def safe_div(num, den):
    return 0.0 if den == 0 else num / den


def resolve_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_layout():
    for directory in [RAW_VIDEO_DIR, MASKED_VIDEO_DIR, LABEL_DIR, BBOX_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_test_videos(config):
    return list(config.get("test_videos") or TEST_VIDEO_FILES)


def selected_test_videos(configs):
    videos = []
    seen_stems = set()
    for config in configs:
        for video_name in get_test_videos(config):
            stem = Path(video_name).stem.lower()
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            videos.append(video_name)
    return videos


def find_video_path(video_name, directory):
    exact_path = directory / video_name
    if exact_path.exists():
        return exact_path

    stem = Path(video_name).stem
    for suffix in VIDEO_EXTENSIONS:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    matches = sorted(path for path in directory.glob(f"{stem}.*") if path.is_file())
    return matches[0] if matches else exact_path


def copy_test_videos_from(source_dir, video_names):
    source_dir = resolve_path(source_dir)
    copied = []
    missing = []
    for file_name in video_names:
        existing_target = find_video_path(file_name, RAW_VIDEO_DIR)
        if existing_target.exists():
            continue

        source = find_video_path(file_name, source_dir)
        target = RAW_VIDEO_DIR / source.name
        if source.exists():
            shutil.copy2(source, target)
            copied.append(source.name)
        else:
            missing.append(file_name)
    return copied, missing


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.reader(file))


def read_dict_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def normalize_key_map(row):
    return {str(key).strip().lower(): value for key, value in row.items() if key is not None}


def parse_int(value):
    return int(float(str(value).strip()))


def load_frame_labels(video_name):
    stem = Path(video_name).stem
    candidates = [
        LABEL_DIR / f"{stem}_labels.csv",
        LABEL_DIR / f"{stem}.csv",
        LABEL_DIR / f"{stem}_holdUsage.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.name.endswith("_holdUsage.csv"):
                return load_hold_usage_labels(candidate), candidate
            return load_direct_labels(candidate), candidate
    return {}, None


def load_direct_labels(path):
    labels = {}
    rows = read_dict_rows(path)
    for row in rows:
        normalized = normalize_key_map(row)
        if "frame" not in normalized:
            continue
        try:
            frame_idx = parse_int(normalized["frame"])
            labels[frame_idx] = [
                parse_int(normalized["left_hand"]),
                parse_int(normalized["right_hand"]),
                parse_int(normalized["left_foot"]),
                parse_int(normalized["right_foot"]),
            ]
        except (KeyError, TypeError, ValueError):
            continue
    return labels


def load_hold_usage_labels(path):
    intervals = []
    for row in read_csv_rows(path):
        if len(row) < 3:
            continue
        limb = row[0].strip().lower()
        try:
            start_frame = parse_int(row[1])
            end_frame = parse_int(row[2])
        except ValueError:
            continue
        intervals.append((limb, start_frame, end_frame))

    if not intervals:
        return {}

    labels = {}
    min_frame = min(start for _, start, _ in intervals)
    max_frame = max(end for _, _, end in intervals)
    for frame_idx in range(min_frame, max_frame + 1):
        labels[frame_idx] = labels_from_intervals(frame_idx, intervals)
    return labels


def labels_from_intervals(frame_idx, intervals):
    labels = [0, 0, 0, 0]
    limb_index = {"lh": 0, "rh": 1, "lf": 2, "rf": 3}
    for limb, start_frame, end_frame in intervals:
        for marker, index in limb_index.items():
            if marker in limb and start_frame <= frame_idx <= end_frame:
                labels[index] = 1
    return labels


def load_bboxes(video_name):
    stem = Path(video_name).stem
    candidates = [
        BBOX_DIR / f"{stem}.csv",
        FALLBACK_BBOX_DIR / f"{stem}.csv",
    ]
    bbox_path = next((path for path in candidates if path.exists()), None)
    if bbox_path is None:
        return {}, None

    bboxes = {}
    for row in read_dict_rows(bbox_path):
        normalized = normalize_key_map(row)
        raw_x1 = str(normalized.get("x1", "")).strip().upper()
        if raw_x1 in {"", "NULL", "NAN", "NONE"}:
            continue
        try:
            frame_idx = parse_int(normalized["frame"])
            bboxes[frame_idx] = (
                float(normalized["x1"]),
                float(normalized["y1"]),
                float(normalized["x2"]),
                float(normalized["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            continue

    return bboxes, bbox_path


def crop_and_pad_frame(frame, bbox, margin=0.15):
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    mx = width * margin
    my = height * margin

    crop_x1 = max(0, int(x1 - mx))
    crop_y1 = max(0, int(y1 - my))
    crop_x2 = min(w_img, int(x2 + mx))
    crop_y2 = min(h_img, int(y2 + my))
    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    ch, cw = cropped.shape[:2]
    if ch <= 0 or cw <= 0:
        return None

    max_dim = max(ch, cw)
    top = (max_dim - ch) // 2
    bottom = max_dim - ch - top
    left = (max_dim - cw) // 2
    right = max_dim - cw - left
    return cv2.copyMakeBorder(
        cropped,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def build_model(model_name, dropout_rate):
    model_name = model_name.lower()
    creators = {
        "b0": models.efficientnet_b0,
        "b1": models.efficientnet_b1,
        "b2": models.efficientnet_b2,
    }
    if model_name not in creators:
        raise ValueError(f"Unsupported model_name: {model_name}")

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
        for key in ["model_state_dict", "state_dict"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_model(config, device):
    model_path_value = config.get("model_path", "")
    if not model_path_value:
        raise ValueError(
            f"Model path for '{config['name']}' is empty. Fill MODEL_CONFIGS at the top of this file "
            "or pass --models-json."
        )

    model_path = resolve_path(model_path_value)
    if not model_path.exists():
        checkpoint_dir = model_path.parent
        available = sorted(path.name for path in checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []
        available_text = ", ".join(available) if available else "none"
        raise FileNotFoundError(
            f"Model file for '{config['name']}' not found: {model_path}. "
            f"Available .pth files in {checkpoint_dir}: {available_text}"
        )

    model = build_model(config.get("model_name", "b2"), float(config.get("dropout_rate", 0.6)))
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def build_preprocess(model_name):
    resolution = MODEL_RESOLUTIONS[model_name.lower()]
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def predict_frame(frame, model, preprocess, device):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(tensor).squeeze(0).detach().cpu().tolist()
    return [float(value) for value in probs]


def predict_batch(frames, model, preprocess, device):
    tensors = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensors.append(preprocess(rgb_frame))

    batch = torch.stack(tensors).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.inference_mode():
        output = model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    return output.detach().cpu().tolist(), elapsed


def bce_loss(labels, probs, eps=1e-7):
    total = 0.0
    for truth, prob in zip(labels, probs):
        p = min(max(prob, eps), 1.0 - eps)
        total += -(truth * math.log(p) + (1 - truth) * math.log(1.0 - p))
    return total / len(labels)


def collect_examples(args, config):
    examples = []
    data_rows = []
    model_name = config["name"]
    video_names = get_test_videos(config)
    video_dir = MASKED_VIDEO_DIR if config["input_mode"] == "masked" else RAW_VIDEO_DIR

    for video_name in video_names:
        video_path = find_video_path(video_name, video_dir)
        resolved_video_name = video_path.name if video_path.exists() else video_name
        labels_by_frame, label_path = load_frame_labels(video_name)
        bboxes, bbox_path = load_bboxes(video_name)

        if not video_path.exists():
            data_rows.append(
                {
                    "model": model_name,
                    "input_mode": config["input_mode"],
                    "video_request": video_name,
                    "video": resolved_video_name,
                    "status": "missing_video",
                    "frames_total": 0,
                    "frames_loaded": 0,
                    "frames_labeled": 0,
                    "label_path": path_to_str(label_path),
                    "bbox_path": path_to_str(bbox_path),
                    "bbox_coverage": "",
                    "LH_positive": "",
                    "RH_positive": "",
                    "LF_positive": "",
                    "RF_positive": "",
                }
            )
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            data_rows.append(
                {
                    "model": model_name,
                    "input_mode": config["input_mode"],
                    "video_request": video_name,
                    "video": resolved_video_name,
                    "status": "cannot_open_video",
                    "frames_total": 0,
                    "frames_loaded": 0,
                    "frames_labeled": 0,
                    "label_path": path_to_str(label_path),
                    "bbox_path": path_to_str(bbox_path),
                    "bbox_coverage": "",
                    "LH_positive": "",
                    "RH_positive": "",
                    "LF_positive": "",
                    "RF_positive": "",
                }
            )
            continue

        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if labels_by_frame:
            frame_indices = sorted(labels_by_frame)
        else:
            frame_indices = list(range(0, frames_total, args.unlabeled_frame_step))

        positives = [0, 0, 0, 0]
        bbox_hits = 0
        valid_frame_indices = [frame_idx for frame_idx in frame_indices if 0 <= frame_idx < frames_total]
        for frame_idx in valid_frame_indices:

            labels = labels_by_frame.get(frame_idx)
            if labels is not None:
                for i, value in enumerate(labels):
                    positives[i] += value
            if frame_idx in bboxes:
                bbox_hits += 1

            examples.append(EvalExample(video_name=resolved_video_name, frame_idx=frame_idx, labels=labels))

        cap.release()
        loaded = len(valid_frame_indices)
        labeled_count = sum(1 for frame_idx in valid_frame_indices if frame_idx in labels_by_frame)
        data_rows.append(
            {
                "model": model_name,
                "input_mode": config["input_mode"],
                "video_request": video_name,
                "video": resolved_video_name,
                "status": "ok",
                "frames_total": frames_total,
                "frames_loaded": loaded,
                "frames_labeled": labeled_count,
                "label_path": path_to_str(label_path),
                "bbox_path": path_to_str(bbox_path),
                "bbox_coverage": round(safe_div(bbox_hits, loaded), 6),
                "LH_positive": positives[0],
                "RH_positive": positives[1],
                "LF_positive": positives[2],
                "RF_positive": positives[3],
                "LH_positive_rate": round(safe_div(positives[0], labeled_count), 6),
                "RH_positive_rate": round(safe_div(positives[1], labeled_count), 6),
                "LF_positive_rate": round(safe_div(positives[2], labeled_count), 6),
                "RF_positive_rate": round(safe_div(positives[3], labeled_count), 6),
            }
        )

    return examples, data_rows


def get_model_input(example, config, args, masked_cache, raw_cache, bbox_cache):
    mode = config["input_mode"]
    if mode == "raw":
        frame = read_raw_frame(example, raw_cache)
        if frame is None:
            return None, {"used_bbox": False, "used_masked_video": False, "skipped_reason": "missing_raw_frame"}
        return frame, {"used_bbox": False, "used_masked_video": False, "skipped_reason": ""}

    if mode == "masked":
        frame = read_masked_frame(example, masked_cache)
        if frame is None:
            return None, {
                "used_bbox": False,
                "used_masked_video": False,
                "skipped_reason": "missing_masked_video_or_frame",
            }
        return frame, {"used_bbox": False, "used_masked_video": True, "skipped_reason": ""}

    if mode == "crop":
        raw_frame = read_raw_frame(example, raw_cache)
        if raw_frame is None:
            return None, {"used_bbox": False, "used_masked_video": False, "skipped_reason": "missing_raw_frame"}

        bboxes, _ = bbox_cache.get(Path(example.video_name).stem, ({}, None))
        bbox = bboxes.get(example.frame_idx)
        if bbox is None:
            if args.missing_bbox == "skip":
                return None, {"used_bbox": False, "used_masked_video": False, "skipped_reason": "missing_bbox"}
            return raw_frame, {"used_bbox": False, "used_masked_video": False, "skipped_reason": ""}

        cropped = crop_and_pad_frame(raw_frame, bbox, args.crop_margin)
        if cropped is None:
            if args.missing_bbox == "skip":
                return None, {"used_bbox": False, "used_masked_video": False, "skipped_reason": "invalid_bbox_crop"}
            return raw_frame, {"used_bbox": False, "used_masked_video": False, "skipped_reason": ""}
        return cropped, {"used_bbox": True, "used_masked_video": False, "skipped_reason": ""}

    raise ValueError(f"Unsupported input_mode: {mode}")


def read_masked_frame(example, masked_cache):
    video_path = find_masked_video_path(example.video_name)
    if not video_path.exists():
        return None
    return read_cached_video_frame(video_path.name, example.frame_idx, masked_cache, video_path)


def find_masked_video_path(raw_video_name):
    return find_video_path(raw_video_name, MASKED_VIDEO_DIR)


def read_raw_frame(example, raw_cache):
    video_path = find_video_path(example.video_name, RAW_VIDEO_DIR)
    if not video_path.exists():
        return None
    return read_cached_video_frame(video_path.name, example.frame_idx, raw_cache, video_path)


def read_cached_video_frame(cache_key, frame_idx, cache, video_path):
    entry = cache.get(cache_key)
    if entry is None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        entry = {"cap": cap, "next_frame": 0}
        cache[cache_key] = entry

    cap = entry["cap"]
    if entry["next_frame"] != frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()
    if ret:
        entry["next_frame"] = frame_idx + 1
    return frame if ret else None


def evaluate_model(config, examples, args, device):
    model = load_model(config, device)
    preprocess = build_preprocess(config.get("model_name", "b2"))
    bbox_cache = {Path(video_name).stem: load_bboxes(video_name) for video_name in get_test_videos(config)}
    masked_cache = {}
    raw_cache = {}

    limb_stats = {name: BinaryStats() for name in CLASS_NAMES}
    video_stats = {}
    predictions = []
    bce_values = []
    exact_matches = 0
    hamming_errors = 0
    labeled_examples = 0
    evaluated_examples = 0
    skipped_examples = 0
    bbox_used = 0
    masked_used = 0
    inference_seconds = 0.0
    batch_items = []

    def record_prediction(example, meta, probs):
        nonlocal labeled_examples
        nonlocal exact_matches
        nonlocal hamming_errors

        preds = [1 if prob >= args.threshold else 0 for prob in probs]

        labels = example.labels
        if labels is not None:
            labeled_examples += 1
            bce_values.append(bce_loss(labels, probs))
            exact_matches += int(preds == labels)
            hamming_errors += sum(1 for truth, pred in zip(labels, preds) if truth != pred)

            per_video = video_stats.setdefault(example.video_name, new_video_stats())
            per_video["labeled_examples"] += 1
            per_video["bce_values"].append(bce_values[-1])
            per_video["exact_matches"] += int(preds == labels)
            per_video["hamming_errors"] += sum(1 for truth, pred in zip(labels, preds) if truth != pred)

            for idx, class_name in enumerate(CLASS_NAMES):
                limb_stats[class_name].add(labels[idx], preds[idx])
                per_video["limb_stats"][class_name].add(labels[idx], preds[idx])

        predictions.append(prediction_row(config, example, probs, preds, meta, args.threshold))

    def flush_batch():
        nonlocal inference_seconds

        if not batch_items:
            return

        batch_frames = [item["frame"] for item in batch_items]
        batch_probs, elapsed = predict_batch(batch_frames, model, preprocess, device)
        inference_seconds += elapsed

        for item, probs in zip(batch_items, batch_probs):
            record_prediction(item["example"], item["meta"], probs)

        batch_items.clear()

    for example in examples:
        model_input, meta = get_model_input(example, config, args, masked_cache, raw_cache, bbox_cache)
        if model_input is None:
            skipped_examples += 1
            predictions.append(
                prediction_row(config, example, None, None, meta, args.threshold)
            )
            continue

        evaluated_examples += 1
        bbox_used += int(meta["used_bbox"])
        masked_used += int(meta["used_masked_video"])
        batch_items.append({"example": example, "frame": model_input, "meta": meta})

        if len(batch_items) >= args.batch_size:
            flush_batch()

    flush_batch()

    for entry in masked_cache.values():
        entry["cap"].release()
    for entry in raw_cache.values():
        entry["cap"].release()

    per_limb_rows = build_limb_rows(config, limb_stats)
    per_video_rows = build_video_rows(config, video_stats)
    confusion_rows = build_confusion_rows(config, limb_stats)
    summary_row = build_summary_row(
        config,
        bce_values,
        limb_stats,
        labeled_examples,
        evaluated_examples,
        skipped_examples,
        exact_matches,
        hamming_errors,
        bbox_used,
        masked_used,
        inference_seconds,
    )
    return summary_row, per_limb_rows, per_video_rows, confusion_rows, predictions


def prediction_row(config, example, probs, preds, meta, threshold):
    row = {
        "model": config["name"],
        "input_mode": config["input_mode"],
        "video": example.video_name,
        "frame": example.frame_idx,
        "threshold": threshold,
        "has_labels": example.labels is not None,
        "used_bbox": meta["used_bbox"],
        "used_masked_video": meta["used_masked_video"],
        "skipped_reason": meta["skipped_reason"],
    }
    labels = example.labels or ["", "", "", ""]
    probs = probs or ["", "", "", ""]
    preds = preds or ["", "", "", ""]
    for idx, class_name in enumerate(CLASS_NAMES):
        row[f"{class_name}_label"] = labels[idx]
        row[f"{class_name}_prob"] = probs[idx]
        row[f"{class_name}_pred"] = preds[idx]
    return row


def new_video_stats():
    return {
        "labeled_examples": 0,
        "bce_values": [],
        "exact_matches": 0,
        "hamming_errors": 0,
        "limb_stats": {name: BinaryStats() for name in CLASS_NAMES},
    }


def build_limb_rows(config, limb_stats):
    rows = []
    for class_name, stats in limb_stats.items():
        rows.append(
            {
                "model": config["name"],
                "input_mode": config["input_mode"],
                "class": class_name,
                "support": stats.tp + stats.fn,
                "total": stats.total,
                "accuracy": round(stats.accuracy(), 6),
                "precision": round(stats.precision(), 6),
                "recall": round(stats.recall(), 6),
                "f1": round(stats.f1(), 6),
                "tp": stats.tp,
                "tn": stats.tn,
                "fp": stats.fp,
                "fn": stats.fn,
            }
        )
    return rows


def build_video_rows(config, video_stats):
    rows = []
    for video_name, stats in sorted(video_stats.items()):
        limb_f1 = [limb_stats.f1() for limb_stats in stats["limb_stats"].values()]
        rows.append(
            {
                "model": config["name"],
                "input_mode": config["input_mode"],
                "video": video_name,
                "bce_loss": round(mean(stats["bce_values"]), 6),
                "exact_match_accuracy": round(safe_div(stats["exact_matches"], stats["labeled_examples"]), 6),
                "hamming_loss": round(
                    safe_div(stats["hamming_errors"], stats["labeled_examples"] * len(CLASS_NAMES)), 6
                ),
                "macro_f1": round(mean(limb_f1), 6),
                "labeled_examples": stats["labeled_examples"],
            }
        )
    return rows


def build_confusion_rows(config, limb_stats):
    rows = []
    for class_name, stats in limb_stats.items():
        rows.append(
            {
                "model": config["name"],
                "input_mode": config["input_mode"],
                "class": class_name,
                "tp": stats.tp,
                "tn": stats.tn,
                "fp": stats.fp,
                "fn": stats.fn,
            }
        )
    return rows


def build_summary_row(
    config,
    bce_values,
    limb_stats,
    labeled_examples,
    evaluated_examples,
    skipped_examples,
    exact_matches,
    hamming_errors,
    bbox_used,
    masked_used,
    inference_seconds,
):
    total_tp = sum(stats.tp for stats in limb_stats.values())
    total_fp = sum(stats.fp for stats in limb_stats.values())
    total_fn = sum(stats.fn for stats in limb_stats.values())
    precision_micro = safe_div(total_tp, total_tp + total_fp)
    recall_micro = safe_div(total_tp, total_tp + total_fn)
    micro_f1 = safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro)
    macro_f1 = mean([stats.f1() for stats in limb_stats.values()])

    return {
        "model": config["name"],
        "description": config.get("description", ""),
        "input_mode": config["input_mode"],
        "bce_loss": round(mean(bce_values), 6),
        "exact_match_accuracy": round(safe_div(exact_matches, labeled_examples), 6),
        "hamming_loss": round(safe_div(hamming_errors, labeled_examples * len(CLASS_NAMES)), 6),
        "macro_f1": round(macro_f1, 6),
        "micro_f1": round(micro_f1, 6),
        "labeled_examples": labeled_examples,
        "evaluated_examples": evaluated_examples,
        "skipped_examples": skipped_examples,
        "bbox_usage_rate": round(safe_div(bbox_used, evaluated_examples), 6),
        "masked_video_usage_rate": round(safe_div(masked_used, evaluated_examples), 6),
        "avg_inference_ms": round(safe_div(inference_seconds * 1000.0, evaluated_examples), 6),
        "fps": round(safe_div(evaluated_examples, inference_seconds), 6),
    }


def mean(values):
    return safe_div(sum(values), len(values))


def path_to_str(path):
    return "" if path is None else str(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path, summary_rows, per_limb_rows, per_video_rows, data_rows):
    lines = [
        "# Evaluation report",
        "",
        "## Test videos",
        "",
        "| Model | Mode | Video | Status | Loaded frames | Labeled frames | Bbox coverage | LH+ | RH+ | LF+ | RF+ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data_rows:
        lines.append(
            "| {model} | {input_mode} | {video} | {status} | {frames_loaded} | {frames_labeled} | {bbox_coverage} | "
            "{LH_positive} | {RH_positive} | {LF_positive} | {RF_positive} |".format(
                model=row.get("model", ""),
                input_mode=row.get("input_mode", ""),
                video=row.get("video", ""),
                status=row.get("status", ""),
                frames_loaded=row.get("frames_loaded", ""),
                frames_labeled=row.get("frames_labeled", ""),
                bbox_coverage=row.get("bbox_coverage", ""),
                LH_positive=row.get("LH_positive", ""),
                RH_positive=row.get("RH_positive", ""),
                LF_positive=row.get("LF_positive", ""),
                RF_positive=row.get("RF_positive", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Model summary",
            "",
            "| Model | Mode | BCE loss | Exact match acc | Hamming loss | Macro F1 | Micro F1 | FPS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {model} | {input_mode} | {bce_loss} | {exact_match_accuracy} | {hamming_loss} | "
            "{macro_f1} | {micro_f1} | {fps} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Per-limb F1",
            "",
            "| Model | LH | RH | LF | RF |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_name in sorted({row["model"] for row in per_limb_rows}):
        by_class = {row["class"]: row for row in per_limb_rows if row["model"] == model_name}
        lines.append(
            "| {model} | {LH} | {RH} | {LF} | {RF} |".format(
                model=model_name,
                LH=by_class.get("LH", {}).get("f1", ""),
                RH=by_class.get("RH", {}).get("f1", ""),
                LF=by_class.get("LF", {}).get("f1", ""),
                RF=by_class.get("RF", {}).get("f1", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Per-video results",
            "",
            "| Model | Video | BCE loss | Exact match acc | Hamming loss | Macro F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_video_rows:
        lines.append(
            "| {model} | {video} | {bce_loss} | {exact_match_accuracy} | {hamming_loss} | {macro_f1} |".format(
                **row
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model_configs(path):
    if path is None:
        return MODEL_CONFIGS
    with resolve_path(path).open(encoding="utf-8") as file:
        return json.load(file)


def validate_model_configs(configs, selected_models):
    selected = set(selected_models or [])
    valid = []
    for config in configs:
        if selected and config["name"] not in selected:
            continue
        valid.append(config)
    if not valid:
        raise ValueError("No model configs selected for evaluation.")
    return valid


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch installation cannot use it. "
                f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}. "
                "Install a CUDA-enabled PyTorch build or run with --device cpu."
            )
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    print(
        "CUDA is not available for this Python environment; using CPU. "
        f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}"
    )
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the three EfficientNet climbing-contact models on their configured video test sets."
    )
    parser.add_argument("--models-json", default=None, help="Optional JSON file with MODEL_CONFIGS replacement.")
    parser.add_argument("--only", nargs="*", default=None, help="Evaluate only selected model names.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Evaluation device. Use 'cuda' to fail fast if GPU support is not configured.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of preprocessed frames evaluated in one model forward pass.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for every output label.")
    parser.add_argument("--crop-margin", type=float, default=0.15, help="Crop margin around climber bbox.")
    parser.add_argument(
        "--missing-bbox",
        choices=["full", "skip"],
        default="full",
        help="For crop model, use full frame or skip frame when bbox is missing.",
    )
    parser.add_argument(
        "--unlabeled-frame-step",
        type=int,
        default=30,
        help="Frame step used for prediction-only videos without labels.",
    )
    parser.add_argument(
        "--copy-test-videos-from",
        default=None,
        help="Optional source directory. Missing selected test videos are copied into data/test_videos/raw.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    ensure_layout()
    configs = validate_model_configs(load_model_configs(args.models_json), args.only)

    if args.copy_test_videos_from:
        copied, missing = copy_test_videos_from(args.copy_test_videos_from, selected_test_videos(configs))
        if copied:
            print(f"Copied test videos: {', '.join(copied)}")
        if missing:
            print(f"Missing in source directory: {', '.join(missing)}")

    device = resolve_device(args.device)

    summary_rows = []
    per_limb_rows = []
    per_video_rows = []
    confusion_rows = []
    prediction_rows = []
    data_rows = []

    for config in configs:
        examples, model_data_rows = collect_examples(args, config)
        data_rows.extend(model_data_rows)
        unavailable = [row["video_request"] for row in model_data_rows if row["status"] != "ok"]
        if unavailable:
            print(f"Warning: {config['name']} has unavailable test videos: {', '.join(unavailable)}")
        print(f"Evaluating {config['name']} on {device} with {len(examples)} frames...")
        summary, limbs, videos, confusion, predictions = evaluate_model(config, examples, args, device)
        summary_rows.append(summary)
        per_limb_rows.extend(limbs)
        per_video_rows.extend(videos)
        confusion_rows.extend(confusion)
        prediction_rows.extend(predictions)

    write_csv(RESULTS_DIR / "data_stats.csv", data_rows)
    write_csv(RESULTS_DIR / "summary_metrics.csv", summary_rows)
    write_csv(RESULTS_DIR / "per_limb_metrics.csv", per_limb_rows)
    write_csv(RESULTS_DIR / "per_video_metrics.csv", per_video_rows)
    write_csv(RESULTS_DIR / "confusion_matrices.csv", confusion_rows)
    write_csv(RESULTS_DIR / "predictions.csv", prediction_rows)
    write_markdown_report(
        RESULTS_DIR / "report.md",
        summary_rows=summary_rows,
        per_limb_rows=per_limb_rows,
        per_video_rows=per_video_rows,
        data_rows=data_rows,
    )

    print(f"Saved evaluation outputs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
