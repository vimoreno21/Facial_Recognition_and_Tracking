import os
import pickle
from pathlib import Path

import cv2
import numpy as np

from src.detector.yolo_face import YOLOFaceDetector
from src.recognition.arcface_embedder import ArcFaceEmbedder
from src.main import get_best_device



DATA_DIR = Path("data/known")
OUT_PATH = Path("data/known_embeddings.pkl")


def get_images():
    persons = {}
    for person_dir in DATA_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        imgs = []
        for img_path in person_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            imgs.append(img_path)
        if imgs:
            persons[name] = imgs
    return persons


def main():
    device = get_best_device()
    print(f"[info] Using device: {device}")

    detector = YOLOFaceDetector(device=device, conf_threshold=0.5)
    embedder = ArcFaceEmbedder(ctx_id=-1)  # CPU for ArcFace

    persons = get_images()
    if not persons:
        print("[error] No images found in data/known/*")
        return

    db = {}  # name -> list of embeddings

    for name, img_paths in persons.items():
        embs = []
        print(f"[info] Processing {name} ({len(img_paths)} images)")
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"[warn] Could not read {p}")
                continue

            # detect face, use biggest box
            dets = detector.detect(img)
            if not dets:
                print(f"[warn] No face detected in {p}")
                continue

            det = max(dets, key=lambda d: (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]))
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            face_crop = img[y1:y2, x1:x2]

            try:
                emb = embedder.get_embedding(face_crop)
                embs.append(emb)
            except Exception as e:
                print(f"[warn] Failed on {p}: {e}")

        if embs:
            db[name] = np.stack(embs, axis=0)
            print(f"[info] {name}: {db[name].shape}")
        else:
            print(f"[warn] No usable embeddings for {name}")

    if not db:
        print("[error] No embeddings created, aborting save")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(db, f)

    print(f"[info] Saved embeddings DB to {OUT_PATH}")


if __name__ == "__main__":
    main()
