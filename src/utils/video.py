"""Small video utilities used by demo scripts.
"""

import argparse
import cv2
import torch


def get_best_device() -> str:
    """Pick the best available device: cuda to mps to cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    """Minimal CLI: `--source` (webcam or file) and `--img-size`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Webcam '0' or path to video")
    parser.add_argument("--img-size", type=int, default=640, help="Max side length of resized frame")
    return parser.parse_args()


def resize_keep_aspect(frame, max_size: int):
    """Resize `frame` so the longest side is `max_size`; preserve aspect."""
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return frame
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
