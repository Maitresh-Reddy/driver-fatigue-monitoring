#!/usr/bin/env python3
import json
from pathlib import Path


def main():
    source = Path("results/multi_video_benchmark.json")
    target = Path("results/multi_video_benchmark_table.md")

    data = json.loads(source.read_text(encoding="utf-8"))
    summary = data["summary"]

    lines = []
    lines.append("# Multi-Video Benchmark Summary")
    lines.append("")
    lines.append("| Scope | Videos | Frames | Alerts | Mean FPS | Mean Face Detect % | Alerts/1000 frames |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    overall = summary["overall"]
    lines.append(
        f"| Overall | {overall['videos']} | {overall['total_frames']} | {overall['total_alerts']} | "
        f"{overall['mean_fps']} | {overall['mean_face_detection_rate_pct']} | {overall['alert_rate_per_1000_frames']} |"
    )

    for label in sorted(summary["by_label"].keys()):
        group = summary["by_label"][label]
        lines.append(
            f"| {label} | {group['videos']} | {group['total_frames']} | {group['total_alerts']} | "
            f"{group['mean_fps']} | {group['mean_face_detection_rate_pct']} | {group['alert_rate_per_1000_frames']} |"
        )

    false_proxy = summary.get("false_alert_proxy", {})
    if false_proxy:
        lines.append("")
        lines.append("## False Alert Proxy")
        lines.append("")
        lines.append(f"- Definition: {false_proxy.get('definition', '')}")
        lines.append(f"- Normal videos: {false_proxy.get('normal_videos', 0)}")
        lines.append(f"- Normal alerts: {false_proxy.get('normal_alerts', 0)}")
        lines.append(f"- Normal frames: {false_proxy.get('normal_frames', 0)}")
        lines.append(
            f"- False alert rate per 1000 frames: {false_proxy.get('false_alert_rate_per_1000_frames', 0)}"
        )

    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {target}")


if __name__ == "__main__":
    main()
