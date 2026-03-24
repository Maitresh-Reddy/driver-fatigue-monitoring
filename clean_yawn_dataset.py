#!/usr/bin/env python3
from pathlib import Path
import shutil
import cv2
import mediapipe as mp
import numpy as np

BASE = Path("dataset/yawn")
YAWN_DIR = BASE / "yawn"
NON_YAWN_DIR = BASE / "non_yawn"

MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def mar_from_landmarks(landmarks, w, h):
    points = []
    for idx in MOUTH_INDICES[:20]:
        lm = landmarks[idx]
        points.append([lm.x * w, lm.y * h])

    points = np.array(points, dtype=np.float32)

    a = np.linalg.norm(points[2] - points[10])
    b = np.linalg.norm(points[4] - points[8])
    c = np.linalg.norm(points[0] - points[16])
    if c <= 1e-6:
        return 0.0
    return float((a + b) / (2.0 * c))


def unique_dest(dest_dir: Path, src_name: str) -> Path:
    dst = dest_dir / src_name
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def relabel_file(src: Path, target_dir: Path):
    dst = unique_dest(target_dir, src.name)
    shutil.move(str(src), str(dst))


def main():
    if not YAWN_DIR.exists() or not NON_YAWN_DIR.exists():
        print("dataset/yawn/{yawn,non_yawn} not found")
        return 1

    all_yawn = list(YAWN_DIR.glob("*.jpg")) + list(YAWN_DIR.glob("*.png"))
    all_non = list(NON_YAWN_DIR.glob("*.jpg")) + list(NON_YAWN_DIR.glob("*.png"))

    if hasattr(mp, "solutions"):
        face_mesh_module = mp.solutions.face_mesh
    else:
        from mediapipe.python import solutions as mp_solutions
        face_mesh_module = mp_solutions.face_mesh

    mp_mesh = face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    moved_y2n = 0
    moved_n2y = 0
    noface = 0

    yawn_keep_threshold = 0.53
    non_yawn_to_yawn_threshold = 0.62

    print(f"Scanning yawn images: {len(all_yawn)}")
    for img_path in all_yawn:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_mesh.process(rgb)
        if not result.multi_face_landmarks:
            noface += 1
            continue

        mar = mar_from_landmarks(result.multi_face_landmarks[0].landmark, w, h)
        if mar < yawn_keep_threshold:
            relabel_file(img_path, NON_YAWN_DIR)
            moved_y2n += 1

    all_non = list(NON_YAWN_DIR.glob("*.jpg")) + list(NON_YAWN_DIR.glob("*.png"))
    print(f"Scanning non_yawn images: {len(all_non)}")
    for img_path in all_non:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_mesh.process(rgb)
        if not result.multi_face_landmarks:
            noface += 1
            continue

        mar = mar_from_landmarks(result.multi_face_landmarks[0].landmark, w, h)
        if mar > non_yawn_to_yawn_threshold:
            relabel_file(img_path, YAWN_DIR)
            moved_n2y += 1

    mp_mesh.close()

    final_yawn = sum(1 for _ in YAWN_DIR.glob("*.jpg")) + sum(1 for _ in YAWN_DIR.glob("*.png"))
    final_non = sum(1 for _ in NON_YAWN_DIR.glob("*.jpg")) + sum(1 for _ in NON_YAWN_DIR.glob("*.png"))

    print("--- Cleanup summary ---")
    print(f"Moved yawn -> non_yawn: {moved_y2n}")
    print(f"Moved non_yawn -> yawn: {moved_n2y}")
    print(f"No-face images skipped: {noface}")
    print(f"Final yawn count: {final_yawn}")
    print(f"Final non_yawn count: {final_non}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
