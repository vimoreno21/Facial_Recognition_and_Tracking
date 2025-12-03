# Real-Time Face Recognition & Tracking

This project implements a **local, real-time face recognition system**.
It detects faces, tracks them across frames, and identifies known individuals using facial embeddings.
Everything runs on-device — no cloud processing.

The pipeline includes:

* **YOLOv8-Face** for detection
* **DeepSORT** for tracking
* **InsightFace** for embeddings
* **Cosine similarity** for recognition

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate    
pip install -r requirements.txt
```

---

## Preparing Known Faces

Add images of each person under:

```
data/known/<person_name>/*.jpg
```

Example:

```
data/known/victoria/
data/known/christina/
```

Each image should contain **one clear face**.

---

## Build the Embedding Database

Run this after adding images:

```bash
python -m src.build_known_db
```

This creates:

```
data/known_embeddings.pkl
```

You only need to recreate it if you add or change images.

---

## Running the System

### Webcam:

```bash
python -m src.main --source 0
```

### Video file:

```bash
python -m src.main --source path/to/video.mp4
```

The output window will display:

* Face bounding boxes
* Stable DeepSORT track IDs
* Recognized names or “Unknown”
* FPS

---

## Project Structure

```
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
```

---

## Notes

* The system automatically selects the best device (CUDA → MPS → CPU).
* Recognition is re-run periodically to improve stability.
* Tracking ensures labels stay consistent over time even if faces move.
