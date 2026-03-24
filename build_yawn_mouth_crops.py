#!/usr/bin/env python3
from pathlib import Path
import shutil
import cv2
import mediapipe as mp

BASE = Path('dataset/yawn')
SRC_Y = BASE / 'yawn'
SRC_N = BASE / 'non_yawn'
TMP = BASE / '_mouth_tmp'
TMP_Y = TMP / 'yawn'
TMP_N = TMP / 'non_yawn'

MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def crop_mouth(img, landmarks):
    h, w = img.shape[:2]
    pts = []
    for idx in MOUTH_INDICES:
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    pad_x = int((x1 - x0) * 0.35) + 8
    pad_y = int((y1 - y0) * 0.6) + 8

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    if x1 <= x0 or y1 <= y0:
        return None

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)


def process_class(src_dir: Path, dst_dir: Path, mesh):
    dst_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    skip = 0

    files = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png'))
    for i, img_path in enumerate(files):
        img = cv2.imread(str(img_path))
        if img is None:
            skip += 1
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        if not res.multi_face_landmarks:
            skip += 1
            continue

        crop = crop_mouth(img, res.multi_face_landmarks[0].landmark)
        if crop is None:
            skip += 1
            continue

        out = dst_dir / f"{img_path.stem}_mouth.jpg"
        cv2.imwrite(str(out), crop)
        ok += 1

    return ok, skip, len(files)


def main():
    if not SRC_Y.exists() or not SRC_N.exists():
        print('dataset/yawn folders missing')
        return 1

    if not hasattr(mp, 'solutions'):
        print('mediapipe solutions API unavailable')
        return 2

    if TMP.exists():
        shutil.rmtree(TMP)

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    y_ok, y_skip, y_total = process_class(SRC_Y, TMP_Y, mesh)
    n_ok, n_skip, n_total = process_class(SRC_N, TMP_N, mesh)

    mesh.close()

    # Keep balanced subset
    y_files = list(TMP_Y.glob('*.jpg'))
    n_files = list(TMP_N.glob('*.jpg'))
    target = min(len(y_files), len(n_files), 3000)

    if len(y_files) > target:
        for f in y_files[target:]:
            f.unlink(missing_ok=True)
    if len(n_files) > target:
        for f in n_files[target:]:
            f.unlink(missing_ok=True)

    # Replace source dirs with cropped data
    shutil.rmtree(SRC_Y)
    shutil.rmtree(SRC_N)
    shutil.move(str(TMP_Y), str(SRC_Y))
    shutil.move(str(TMP_N), str(SRC_N))
    shutil.rmtree(TMP)

    print('--- Mouth crop dataset build summary ---')
    print(f'yawn: total={y_total}, kept={y_ok}, skipped={y_skip}')
    print(f'non_yawn: total={n_total}, kept={n_ok}, skipped={n_skip}')
    print(f'balanced final size per class: {target}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
