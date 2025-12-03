# Evalution of cropped ytf dataset
# 1 - Build embeddings for a YouTube Faces (YTF)-style dataset
# 2 - Evaluate embedding space by computing cosine similarities for same identity and different identity pairs and sweeping thresholds to find good similarity cutoff

# Run using this
# python -m src.ytf_eval --ytf_root data/ytf/frame_images_DB --max_per_identity 3

import argparse
from pathlib import Path
import pickle
import cv2
import numpy as np
from src.detector.yolo_face import YOLOFaceDetector
from src.recognition.arcface_embedder import ArcFaceEmbedder
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a = a.astype("float32")
    b = b.astype("float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def build_ytf_embeddings(
    ytf_root: Path,
    out_path: Path,
    max_per_identity: int = 20,
    frame_stride: int = 10,
) -> dict:
    """
    Go through portion of YTF dataset structure and build embeddings
    Assume:
    - Each of has one identity
    - Each may contain images and/or videos
    - For images: run YOLO, crop main face, ArcFace embedding
    - For videos: sample frames with `frame_stride` and do the same

    Returns:
        A dict with:
            "embeddings": np.ndarray of shape (N, D)
            "labels": np.ndarray of shape (N,)
    """
    detector = YOLOFaceDetector()
    embedder = ArcFaceEmbedder()

    all_embs = []
    all_labels = []

    identities = sorted(p for p in ytf_root.iterdir() if p.is_dir())
    print(f"Found {len(identities)} identities under {ytf_root}")

    for person_dir in identities:
        label = person_dir.name
        count_for_person = 0
        print(f"Processing identity: {label}")

        # Sort files
        for f in sorted(person_dir.rglob("*")):
            if count_for_person >= max_per_identity:
                break
            suffix = f.suffix.lower()

            # Case 1: image files
            if suffix in IMAGE_EXTS:
                img = cv2.imread(str(f))
                if img is None:
                    continue

                dets = detector.detect(img)
                if not dets:
                    continue

                # Take the highest-score detection
                det = max(dets, key=lambda d: d["score"])
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                face = img[y1:y2, x1:x2]

                try:
                    emb = embedder.get_embedding(face)
                except Exception:
                    continue

                all_embs.append(emb)
                all_labels.append(label)
                count_for_person += 1

            # Case 2: video files
            elif suffix in VIDEO_EXTS:
                cap = cv2.VideoCapture(str(f))
                if not cap.isOpened():
                    continue

                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_stride != 0:
                        frame_idx += 1
                        continue

                    dets = detector.detect(frame)
                    if dets:
                        det = max(dets, key=lambda d: d["score"])
                        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                        face = frame[y1:y2, x1:x2]

                        try:
                            emb = embedder.get_embedding(face)
                        except Exception:
                            frame_idx += 1
                            continue

                        all_embs.append(emb)
                        all_labels.append(label)
                        count_for_person += 1

                        if count_for_person >= max_per_identity:
                            break
                    frame_idx += 1
                cap.release()

        print(f"Collected {count_for_person} samples for {label}")

    if not all_embs:
        raise RuntimeError("[error] No embeddings were created from YTF dataset")

    X = np.stack(all_embs).astype("float32")
    y = np.array(all_labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"embeddings": X, "labels": y}, f)

    print(f"Saved YTF embeddings to {out_path} with {len(y)} samples")
    return {"embeddings": X, "labels": y}


def evaluate_embeddings(embeddings: np.ndarray, labels: np.ndarray, plot_dir: Path, ) -> float:
    """""
    Compute same vs different identity
    cosine similarities, sweep a threshold to find the best value,
    and generate the plots
    Returns:
        best_threshold
    """
    N = len(labels)
    same_sims = []
    diff_sims = []

    print(f"Evaluating {N} embeddings... this may take a bit for large N")

    # Compute pairwise cosine similarities

    for i in range(N):
        for j in range(i + 1, N):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if labels[i] == labels[j]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

    same_sims = np.array(same_sims, dtype="float32")
    diff_sims = np.array(diff_sims, dtype="float32")

    print("Same-identity similarities:")
    print(f"       mean = {same_sims.mean():.4f}, std = {same_sims.std():.4f}")
    print("Different-identity similarities:")
    print(f"       mean = {diff_sims.mean():.4f}, std = {diff_sims.std():.4f}")

    # Sweep thresholds and compute accuracy, precision, recall
    thresholds = np.linspace(0.1, 0.9, 17)  # 0.10, 0.15, to 0.90
    accs = []
    precisions = []
    recalls = []

    total_same = len(same_sims) + 1e-8
    total_diff = len(diff_sims) + 1e-8
    total_pairs = total_same + total_diff

    print("Threshold sweep:")
    best_thr = None
    best_acc = -1.0

    for T in thresholds:
        # For a given threshold T
        #   sim >= T -> "match"
        #   sim <  T -> "non-match"

        # Same-identity pairs
        TP = np.sum(same_sims >= T)  # correctly accept same
        FN = np.sum(same_sims < T)   # miss same

        # Different-identity pairs
        FP = np.sum(diff_sims >= T)  # incorrectly accept different
        TN = np.sum(diff_sims < T)   # correctly reject different

        acc = (TP + TN) / total_pairs
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)

        print(
            f"    T={T:.2f}  acc={acc:.4f}  "
            f"prec={precision:.4f}  rec={recall:.4f}  "
            f"(TP={TP}, FP={FP}, TN={TN}, FN={FN})"
        )

        if acc > best_acc:
            best_acc = acc
            best_thr = T

    print(
        f"Best threshold = {best_thr:.3f} with "
        f"approximate pairwise accuracy = {best_acc:.4f}"
    )

    # Create ytf output directory for plots
    ytfplot_dir = plot_dir
    ytfplot_dir.mkdir(parents=True, exist_ok=True)

    # Histogram plot same vs different similarities
    plt.figure(figsize=(6, 5))
    plt.hist(same_sims, bins=30, alpha=0.5, label="Same identity")
    plt.hist(diff_sims, bins=30, alpha=0.5, label="Different identity")
    plt.axvline(best_thr, linestyle="--", label=f"Threshold = {best_thr:.2f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("YTF Cosine Similarity Distributions")
    plt.legend()
    plt.grid(True)
    hist_path = ytfplot_dir / "ytf_histogram.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram plot to {hist_path}")

    # Threshold vs Accuracy plot
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, accs, marker="o")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Pairwise accuracy")
    plt.title("Accuracy vs Threshold on YTF")
    plt.grid(True)
    acc_path = ytfplot_dir / "ytf_threshold_accuracy.png"
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold-accuracy plot to {acc_path}")

    # Precision–Recall curve

    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve on YTF")
    plt.grid(True)
    pr_path = ytfplot_dir / "ytf_precision_recall.png"
    plt.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved precision–recall plot to {pr_path}")

    return float(best_thr)


def main():
    parser = argparse.ArgumentParser(description="YTF embedding builder and evaluator")
    parser.add_argument(
        "--ytf_root",
        type=str,
        required=True,
        help="Root directory of YTF-style data (subdirs per identity).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/ytf_embeddings.pkl",
        help="Output pickle file for embeddings and labels.",
    )
    parser.add_argument(
        "--max_per_identity",
        type=int,
        default=20,
        help="Maximum number of samples to collect per identity.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=10,
        help="For video files, sample every N-th frame.",
    )

    args = parser.parse_args()
    ytf_root = Path(args.ytf_root)

    # create subfolder in data that includes max_per_identity
    eval_dir = Path("data") / f"ytf_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "ytf_embeddings.pkl"

    # Step 1 build or load embeddings
    if out_path.exists():
        print(f"Loading existing embeddings from {out_path}")
        with open(out_path, "rb") as f:
            data = pickle.load(f)
        X = data["embeddings"]
        y = data["labels"]
    else:
        data = build_ytf_embeddings(
            ytf_root=ytf_root,
            out_path=out_path,
            max_per_identity=args.max_per_identity,
            frame_stride=args.frame_stride,
        )
        X = data["embeddings"]
        y = data["labels"]

    # Step 2 evaluate and get best threshold
    best_thr = evaluate_embeddings(X, y, eval_dir) 

    # save threshold as txt
    thr_path = eval_dir / "ytf_embeddings.threshold.txt" 
    with open(thr_path, "w") as f:
        f.write(f"{best_thr:.4f}\n")
    print(f"Saved best threshold to {thr_path}")


if __name__ == "__main__":
    main()
