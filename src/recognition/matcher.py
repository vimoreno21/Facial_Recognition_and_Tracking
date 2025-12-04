"""Simple face matcher using cosine similarity against a saved DB."""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


class KnownFaceMatcher:
    """Load known embeddings and match an embedding to a name.

    The matcher returns (name, score). If the top score is below
    `threshold` return Unknown.
    """

    def __init__(self, db_path: str = "data/known_embeddings.pkl", threshold: float = 0.45):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Known embeddings DB not found at {self.db_path}")

        with open(self.db_path, "rb") as f:
            self.known_db = pickle.load(f)

        if not self.known_db:
            raise RuntimeError("Known embeddings DB is empty")

        self.threshold = threshold

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Return cosine similarity between two 1-D vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def match(self, emb: np.ndarray) -> Tuple[str, float]:
        """Find the best matching person for `emb`.

        Returns `(name, score)`. Unknown is returned when the best
        score is below `threshold`.
        """
        best_name = "Unknown"
        best_score = -1.0

        for name, emb_array in self.known_db.items():
            for ref_emb in emb_array:
                sim = self.cosine_similarity(emb, ref_emb)
                if sim > best_score:
                    best_score = sim
                    best_name = name

        if best_score < self.threshold:
            best_name = "Unknown"

        return best_name, best_score
