# Real-Time Face Recognition, Tracking, and YTF Embedding Evaluation

This project implements a local, real-time face recognition system.
It detects faces, tracks them across frames, and identifies known individuals
using facial embeddings. Everything runs on-device and there is no cloud processing.

In addition to real-time recognition, the system includes full support for
evaluating the embedding space using a subset of the YouTube Faces (YTF) dataset.
YTF evaluation is used to measure embedding quality, compare identities,
and determine an optimal cosine-similarity threshold that improves
real-time recognition decisions.

The pipeline includes:

* YOLOv8-Face for detection
* DeepSORT for tracking
* InsightFace ArcFace for embeddings
* Cosine similarity for recognition
* YTF evaluation pipeline for computing similarity distributions
  and choosing a robust cosine threshold

------------------------------------------------------------

## Installation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

------------------------------------------------------------

## Preparing Known Faces

Add images of each person under:

data/known/<person_name>/*.jpg

Example:

data/known/victoria/
data/known/christina/

Each image should contain one clear face.

------------------------------------------------------------

## Build the Embedding Database

Run this after adding images:

python -m src.build_known_db

This creates:

data/known_embeddings.pkl

You only need to recreate it if you add or change images.

------------------------------------------------------------

## Running the System (Live Detection)

### Webcam:

python -m src.main --source 0

### Video file:

python -m src.main --source path/to/video.mp4

The output window displays:

* Face bounding boxes  
* Stable DeepSORT track IDs  
* Recognized names or “Unknown”  
* FPS  

### YTF Additions to Live Recognition

The real-time recognition system now incorporates YTF-derived data:

* The cosine-similarity threshold used for matching can be tuned using
  the YTF evaluation script.
* YTF analysis provides statistics on same-identity and different-identity
  similarity distributions, helping choose thresholds that reduce false matches.
* The threshold value saved in data/ytf_eval/ytf_embeddings.threshold.txt
  can be used to update matcher.py for more stable recognition behavior.

This allows the real-time system to operate with thresholds validated
on large-scale identity comparisons rather than arbitrary values.

------------------------------------------------------------

## YTF Embedding Evaluation

The project includes a module for evaluating the quality of the facial
embedding space using a YTF-style dataset. This component is used to:

* Build embeddings for many identities
* Compute same vs. different identity cosine similarities
* Sweep similarity thresholds to compute accuracy, precision, recall
* Save similarity distribution plots
* Produce an optimal cosine-similarity threshold for live recognition

------------------------------------------------------------
### Preparing the YTF Dataset
------------------------------------------------------------

Organize your dataset so each person has a separate folder:

data/ytf/frame_images_DB/<identity_name>/

Example:

data/ytf/frame_images_DB/person_0001/
data/ytf/frame_images_DB/person_0002/

Each folder may contain:

* Images (.jpg, .png, etc.)
* Videos (.mp4, .avi, .mov, etc.)

The system automatically detects faces and computes embeddings.

------------------------------------------------------------
### Running YTF Evaluation
------------------------------------------------------------

python -m src.ytf_eval --ytf_root data/ytf/frame_images_DB --max_per_identity 3

This will:

* Build or load YTF embeddings
* Compute similarity distributions
* Sweep threshold values
* Report best accuracy threshold
* Save plots and numerical results

Outputs are written to:

data/ytf_eval/
    ytf_embeddings.pkl
    ytf_embeddings.threshold.txt
    ytf_histogram.png
    ytf_threshold_accuracy.png
    ytf_precision_recall.png

The saved threshold can be used to improve real-time matching.

------------------------------------------------------------

## Project Structure

src/
  main.py                        # Real-time recognition
  build_known_db.py              # Known embeddings builder
  ytf_eval.py                    # YTF embedding builder + evaluator

  detector/
    yolo_face.py                 # YOLOv8-face detection

  recognition/
    arcface_embedder.py          # ArcFace embedding extraction
    matcher.py                   # Cosine similarity matcher (can use YTF threshold)

  tracker/
    face_tracker.py              # DeepSORT + periodic re-recognition

  utils/
    video.py                     # Device selection, resizing, CLI args

data/
  known/
  known_embeddings.pkl

  ytf/
    frame_images_DB/             # YTF subsection of dataset root

  ytf_eval/
    ytf_embeddings.pkl
    ytf_embeddings.threshold.txt
    ytf_histogram.png
    ytf_threshold_accuracy.png
    ytf_precision_recall.png

------------------------------------------------------------

## Notes

* The system automatically selects the best device (CUDA to MPS to CPU).
* YTF evaluation provides a statistically supported cosine threshold
  that can replace hard-coded match values.
* Recognition is re-run periodically to improve stability.
* Tracking ensures labels stay consistent even when faces move or briefly disappear.
* The YTF integration strengthens identity separation and reduces false positives.

