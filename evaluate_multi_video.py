#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from statistics import mean

import cv2

from src.main import DriverFatigueMonitoringPipeline
from src.config import (
    EYE_MODEL_PATH,
    YAWN_MODEL_PATH,
    DISTRACTION_MODEL_PATH,
    DROWSINESS_MODEL_PATH,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    INFERENCE_SKIP_FRAMES,
)


def infer_label_from_name(name: str) -> str:
    n = name.lower()
    if "yawn" in n:
        return "yawning"
    if "talk" in n:
        return "talking"
    if "normal" in n:
        return "normal"
    return "other"


def evaluate_video(pipeline, video_path: Path, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = 0
    inference_frames = 0
    face_detected = 0
    alerts = 0

    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames > 0 and frame_count >= max_frames:
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        if frame_count % INFERENCE_SKIP_FRAMES == 0:
            _, state = pipeline.process_frame(frame)
            inference_frames += 1
            if state is not None:
                face_detected += 1
                if state.get("should_alert"):
                    alerts += 1
        else:
            state = pipeline.last_state_dict
            if state is not None:
                face_detected += 1

        frame_count += 1

    cap.release()

    elapsed = max(time.time() - start, 1e-6)
    fps = frame_count / elapsed
    alert_rate_1k = (alerts / frame_count) * 1000 if frame_count else 0.0
    face_rate = (face_detected / frame_count) * 100 if frame_count else 0.0

    return {
        "video": video_path.name,
        "label": infer_label_from_name(video_path.name),
        "frames": frame_count,
        "inference_frames": inference_frames,
        "alerts": alerts,
        "alert_rate_per_1000_frames": round(alert_rate_1k, 3),
        "face_detection_rate_pct": round(face_rate, 2),
        "fps": round(fps, 2),
    }


def summarize(rows):
    if not rows:
        return {}

    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    def aggregate(group):
        total_frames = sum(r["frames"] for r in group)
        total_alerts = sum(r["alerts"] for r in group)
        return {
            "videos": len(group),
            "total_frames": total_frames,
            "total_alerts": total_alerts,
            "mean_fps": round(mean(r["fps"] for r in group), 2),
            "mean_face_detection_rate_pct": round(mean(r["face_detection_rate_pct"] for r in group), 2),
            "alert_rate_per_1000_frames": round((total_alerts / total_frames) * 1000, 3) if total_frames else 0.0,
        }

    summary = {
        "overall": aggregate(rows),
        "by_label": {label: aggregate(group) for label, group in by_label.items()},
        "false_alert_proxy": {},
    }

    normal_group = by_label.get("normal", [])
    if normal_group:
        normal_total_frames = sum(r["frames"] for r in normal_group)
        normal_alerts = sum(r["alerts"] for r in normal_group)
        summary["false_alert_proxy"] = {
            "definition": "alerts occurring in videos labeled as normal",
            "normal_videos": len(normal_group),
            "normal_alerts": normal_alerts,
            "normal_frames": normal_total_frames,
            "false_alert_rate_per_1000_frames": round((normal_alerts / normal_total_frames) * 1000, 3) if normal_total_frames else 0.0,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-video benchmark for fatigue monitoring")
    parser.add_argument("--videos_dir", type=str, default="dataset/evaluation_videos")
    parser.add_argument("--max_videos", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--calibration_seconds", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="results/multi_video_benchmark.json")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    videos = sorted(list(videos_dir.glob("*.avi")) + list(videos_dir.glob("*.mp4")))
    videos = videos[: args.max_videos]

    rows = []
    for video in videos:
        pipeline = DriverFatigueMonitoringPipeline(
            eye_model_path=str(EYE_MODEL_PATH),
            yawn_model_path=str(YAWN_MODEL_PATH),
            distraction_model_path=str(DISTRACTION_MODEL_PATH),
            drowsiness_model_path=str(DROWSINESS_MODEL_PATH),
        )
        if hasattr(pipeline.monitoring_system, 'calibration'):
            pipeline.monitoring_system.calibration.calibration_seconds = max(0.0, float(args.calibration_seconds))
        result = evaluate_video(pipeline, video, args.max_frames)
        if result is not None:
            rows.append(result)
            print(
                f"{result['video']}: label={result['label']} frames={result['frames']} "
                f"alerts={result['alerts']} fps={result['fps']}"
            )

    summary = summarize(rows)

    output = {
        "config": {
            "videos_dir": str(videos_dir),
            "evaluated_videos": len(rows),
            "max_videos": args.max_videos,
            "max_frames_per_video": args.max_frames,
            "calibration_seconds": args.calibration_seconds,
            "inference_skip_frames": INFERENCE_SKIP_FRAMES,
            "resolution": [VIDEO_WIDTH, VIDEO_HEIGHT],
        },
        "summary": summary,
        "per_video": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
