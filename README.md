# Real-Time Face Recognition & Tracking

This project implements a local, real-time face recognition system.
It detects faces, tracks them across frames, and identifies known individuals using facial embeddings.
Everything runs on-device — no cloud processing.

The pipeline includes:

* YOLOv8-Face for detection
* DeepSORT for tracking
* InsightFace for embeddings
* Cosine similarity for recognition

---

## Installation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## Preparing Known Faces

Add images of each person under:

data/known/<person_name>/*.jpg

Example:

data/known/victoria/
data/known/christina/

Each image should contain one clear face.

---

## Build the Embedding Database

Run this after adding images:

python -m src.build_known_db

This creates:

data/known_embeddings.pkl

You only need to recreate it if you add or change images.

---

## Running the System
### Webcam:

python -m src.main --source 0

### Video file:

python -m src.main --source path/to/video.mp4

The output window will display:

* Face bounding boxes
* Stable DeepSORT track IDs
* Recognized names or “Unknown”
* FPS

---

## Project Structure

src/
  main.py                    # Main entry point
  detector/
    yolo_face.py             # YOLOv8-face detection
  recognition/
    arcface_embedder.py      # Embedding extraction
    matcher.py               # Cosine similarity matcher
  tracker/
    face_tracker.py          # DeepSORT + periodic re-recognition
  utils/
    video.py                 # Device selection, resizing, CLI args

data/
  known/                     # Images for each person
  known_embeddings.pkl       # Generated embeddings DB

---

## Notes

* The system automatically selects the best device (CUDA → MPS → CPU).
* Recognition is re-run periodically to improve stability.
* Tracking ensures labels stay consistent over time even if faces move.

---

## YTF Embedding Evaluation (Optional)

This project also includes a module for evaluating the quality of the facial embedding space using a YouTube Faces (YTF)-style dataset.
It builds embeddings for each identity, computes cosine similarities for same vs. different identities, and determines an optimal threshold for face matching.

---

### Preparing the YTF Dataset

Organize your dataset so that each person has a separate identity folder:

data/ytf/frame_images_DB/<identity_name>/

Example:

data/ytf/frame_images_DB/person_0001/
data/ytf/frame_images_DB/person_0002/

Each folder may contain:

* Images (.jpg, .png, etc.)
* Videos (.mp4, .avi, .mov, etc.)

The system will automatically detect faces, crop them, and generate embeddings.

---

### Building & Evaluating YTF Embeddings

Run the evaluation with:

python -m src.ytf_eval --ytf_root data/ytf/frame_images_DB --max_per_identity 3

This performs:

* YOLOv8-Face detection
* ArcFace embedding extraction
* Cosine similarity computation
* Threshold sweep to find best matching cutoff
* Histogram, accuracy, and precision–recall plot generation

---

### YTF Outputs

Running the script creates:

data/ytf_eval/
  ytf_embeddings.pkl              # Embeddings + labels
  ytf_embeddings.threshold.txt    # Best cosine similarity threshold
  ytf_histogram.png               # Same vs different similarity distributions
  ytf_threshold_accuracy.png      # Accuracy vs threshold curve
  ytf_precision_recall.png        # Precision–recall curve

These files help validate embedding separation quality and estimate the best threshold for recognition decisions.

---

### Project Structure (Extended with YTF)

src/
  ytf_eval.py                 # YTF embedding builder & evaluator

data/
  ytf/
    frame_images_DB/          # YTF dataset root: one folder per identity

  ytf_eval/
    ytf_embeddings.pkl
    ytf_embeddings.threshold.txt
    ytf_histogram.png
    ytf_threshold_accuracy.png
    ytf_precision_recall.png

---
