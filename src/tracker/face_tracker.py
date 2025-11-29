"""Tracker that combines DeepSort tracking with face recognition."""

from typing import List, Dict, Any

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.recognition.arcface_embedder import ArcFaceEmbedder
from src.recognition.matcher import KnownFaceMatcher


class FaceTracker:
    """Simple wrapper: DeepSort for tracking, matcher+embedder for ID."""

    def __init__(
        self,
        matcher: KnownFaceMatcher,
        embedder: ArcFaceEmbedder,
        max_age: int = 15,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
    ):
        self.matcher = matcher
        self.embedder = embedder

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
        )

        # track_id -> (name, score)
        self.track_id_to_identity: Dict[int, Any] = {}

    def update(self, detections, frame):
        """Update tracker with detections and return confirmed tracks.

        Returns a list of dicts: track_id, x1,y1,x2,y2, name, score.
        """
        h, w = frame.shape[:2]

        # Convert detections to DeepSort format: [x, y, w, h]
        ds_dets = []
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["score"]
            box_w = x2 - x1
            box_h = y2 - y1
            ds_dets.append(([x1, y1, box_w, box_h], conf, "face"))

        # Update DeepSort
        tracks = self.tracker.update_tracks(ds_dets, frame=frame)

        results: List[Dict[str, Any]] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # clamp to frame bounds
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            # assign identity once per track
            if track_id not in self.track_id_to_identity:
                name, score = self._assign_identity(frame, x1, y1, x2, y2)
                self.track_id_to_identity[track_id] = (name, score)

            name, score = self.track_id_to_identity[track_id]

            results.append(
                {
                    "track_id": track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "name": name,
                    "score": score,
                }
            )

        return results

    def _assign_identity(self, frame, x1, y1, x2, y2):
        """Crop face, get embedding, and match to known identities."""
        face_crop = frame[y1:y2, x1:x2]
        if face_crop is None or face_crop.size == 0:
            return "Unknown", -1.0

        try:
            emb = self.embedder.get_embedding(face_crop)
        except Exception:
            return "Unknown", -1.0

        try:
            name, score = self.matcher.match(emb)
        except Exception:
            name, score = "Unknown", -1.0

        return name, score
