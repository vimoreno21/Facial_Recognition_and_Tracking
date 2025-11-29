"""ArcFace embeddings via InsightFace's FaceAnalysis (buffalo_l)."""

import numpy as np
import cv2
from insightface.app import FaceAnalysis


class ArcFaceEmbedder:
    """Wrap InsightFace FaceAnalysis to produce L2-normalized embeddings.

    `ctx_id=-1` uses CPU; `0` would use the first GPU.
    """

    def __init__(self, ctx_id: int = -1):
        self.app = FaceAnalysis(name="buffalo_l")
        # prepare models (detector + recognizer)
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return a float32, L2-normalized embedding for a cropped face.

        Expects a BGR uint8 crop (as from OpenCV). Raises on empty input
        or when no face is found inside the crop.
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face crop")

        if face_bgr.dtype != np.uint8:
            face_bgr = face_bgr.astype(np.uint8)

        # FaceAnalysis will detect & embed
        faces = self.app.get(face_bgr)
        if not faces:
            raise RuntimeError("FaceAnalysis could not find a face in the crop")

        emb = faces[0].normed_embedding  # already normalized
        return np.asarray(emb, dtype="float32").flatten()
