#!/usr/bin/env python3
from pathlib import Path
import shutil
import cv2
import mediapipe as mp
import numpy as np

BASE = Path('dataset/yawn')
YAWN_DIR = BASE / 'yawn'
NON_YAWN_DIR = BASE / 'non_yawn'
TMP_DIR = BASE / '_curated_tmp'
TMP_YAWN = TMP_DIR / 'yawn'
TMP_NON = TMP_DIR / 'non_yawn'

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
        return None
    return float((a + b) / (2.0 * c))


def scan_mar(paths):
    if hasattr(mp, 'solutions'):
        face_mesh_module = mp.solutions.face_mesh
    else:
        raise RuntimeError('MediaPipe solutions API not available')

    mesh = face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    rows = []
    skipped = 0

    for img_path in paths:
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            skipped += 1
            continue

        mar = mar_from_landmarks(result.multi_face_landmarks[0].landmark, w, h)
        if mar is None:
            skipped += 1
            continue

        rows.append((img_path, mar))

    mesh.close()
    return rows, skipped


def copy_unique(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    if dst.exists():
        stem = src.stem
        suffix = src.suffix
        i = 1
        while True:
            candidate = dst_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1
    shutil.copy2(str(src), str(dst))


def main():
    if not YAWN_DIR.exists() or not NON_YAWN_DIR.exists():
        print('dataset/yawn/yawn or dataset/yawn/non_yawn not found')
        return 1

    files = list(YAWN_DIR.glob('*.jpg')) + list(YAWN_DIR.glob('*.png')) + list(NON_YAWN_DIR.glob('*.jpg')) + list(NON_YAWN_DIR.glob('*.png'))
    print(f'Total candidate images: {len(files)}')

    rows, skipped = scan_mar(files)
    if len(rows) < 500:
        print(f'Not enough face-detected images: {len(rows)}, skipped: {skipped}')
        return 2

    mar_values = np.array([r[1] for r in rows], dtype=np.float32)
    low_thr = float(np.percentile(mar_values, 35))
    high_thr = float(np.percentile(mar_values, 65))

    print(f'MAR thresholds => low: {low_thr:.4f}, high: {high_thr:.4f}')

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_YAWN.mkdir(parents=True, exist_ok=True)
    TMP_NON.mkdir(parents=True, exist_ok=True)

    yawn_rows = [r for r in rows if r[1] >= high_thr]
    non_rows = [r for r in rows if r[1] <= low_thr]

    # balance classes
    target = min(len(yawn_rows), len(non_rows), 3000)
    rng = np.random.default_rng(42)

    if len(yawn_rows) > target:
        idx = rng.choice(len(yawn_rows), size=target, replace=False)
        yawn_rows = [yawn_rows[i] for i in idx]
    if len(non_rows) > target:
        idx = rng.choice(len(non_rows), size=target, replace=False)
        non_rows = [non_rows[i] for i in idx]

    for src, _ in yawn_rows:
        copy_unique(src, TMP_YAWN)
    for src, _ in non_rows:
        copy_unique(src, TMP_NON)

    # replace dataset folders
    shutil.rmtree(YAWN_DIR)
    shutil.rmtree(NON_YAWN_DIR)
    shutil.move(str(TMP_YAWN), str(YAWN_DIR))
    shutil.move(str(TMP_NON), str(NON_YAWN_DIR))
    shutil.rmtree(TMP_DIR)

    print('--- Curated dataset rebuilt ---')
    print(f'Face-detected images used: {len(rows)}')
    print(f'Skipped images: {skipped}')
    print(f'Final yawn images: {sum(1 for _ in YAWN_DIR.iterdir())}')
    print(f'Final non_yawn images: {sum(1 for _ in NON_YAWN_DIR.iterdir())}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
