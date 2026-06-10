import argparse
import csv
import json
from pathlib import Path


SUPPORTED_EXTENSIONS = {".mov", ".mp4"}
NULL_CSV = "NULL"


def find_video_files(input_dir):
    input_path = Path(input_dir)
    return sorted(
        [
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def select_primary_climber(result):
    boxes = result.boxes
    if boxes is None:
        return None

    bboxes = boxes.xyxy.tolist()
    if not bboxes:
        return None

    if boxes.id is not None:
        track_ids = boxes.id.tolist()
    else:
        track_ids = [None] * len(bboxes)

    def area(item):
        bbox, _ = item
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    bbox, track_id = max(zip(bboxes, track_ids), key=area)
    return {
        "bbox": [float(coord) for coord in bbox],
        "climber_id": int(track_id) if track_id is not None else None,
    }


def detect_climber_bbox(model, frame, reset_tracker):
    result = model.track(
        frame,
        classes=[0],
        verbose=False,
        persist=not reset_tracker,
        tracker="botsort.yaml",
        conf=0.3,
    )[0]
    return select_primary_climber(result)


def read_video_bboxes(model, video_path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        climber = detect_climber_bbox(model, frame, reset_tracker=frame_idx == 0)
        if climber is None:
            frame_record = {
                "frame": frame_idx,
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None,
                "climber_id": None,
            }
        else:
            x1, y1, x2, y2 = climber["bbox"]
            frame_record = {
                "frame": frame_idx,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "climber_id": climber["climber_id"],
            }

        frames.append(frame_record)

        frame_idx += 1
        if frame_idx % 100 == 0:
            if total_frames > 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  frames: {frame_idx}/{total_frames} ({progress:.1f}%)")
            else:
                print(f"  frames: {frame_idx}")

    cap.release()
    return {
        "video": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
    }


def csv_value(value):
    return NULL_CSV if value is None else value


def save_csv(payload, output_path):
    fieldnames = ["frame", "x1", "y1", "x2", "y2", "climber_id"]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in payload["frames"]:
            writer.writerow({key: csv_value(frame[key]) for key in fieldnames})


def save_json(payload, output_path):
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, ensure_ascii=False)


def save_payload(payload, output_dir, output_format):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(payload["video"]).stem

    if output_format in {"csv", "both"}:
        csv_path = output_dir / f"{stem}.csv"
        save_csv(payload, csv_path)
        print(f"  saved: {csv_path}")

    if output_format in {"json", "both"}:
        json_path = output_dir / f"{stem}.json"
        save_json(payload, json_path)
        print(f"  saved: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export one climber bbox per frame for each video in input_dir."
    )
    parser.add_argument(
        "--input",
        default="data/input/input_dir",
        help="Directory with .mov/.mp4 videos.",
    )
    parser.add_argument(
        "--output",
        default="data/output/climber_bboxes",
        help="Output directory for bbox files.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="csv",
        help="Output format.",
    )
    parser.add_argument(
        "--model",
        default="models/yolov8n.pt",
        help="YOLO model used for person detection.",
    )
    parser.add_argument(
        "--video",
        help="Optional single video path. When set, --input is ignored.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.video:
        video_files = [Path(args.video)]
    else:
        video_files = find_video_files(args.input)

    if not video_files:
        print("No videos found.")
        return

    print(f"Found {len(video_files)} video(s).")
    print(f"Loading model: {args.model}")
    from ultralytics import YOLO

    model = YOLO(args.model)

    output_dir = Path(args.output)
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] {video_path.name}")
        try:
            payload = read_video_bboxes(model, video_path)
            save_payload(payload, output_dir, args.format)
        except Exception as exc:
            print(f"  ERROR: {video_path.name}: {exc}")


if __name__ == "__main__":
    main()
