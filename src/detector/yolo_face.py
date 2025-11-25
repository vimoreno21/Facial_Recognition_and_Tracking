from ultralytics import YOLO
import numpy as np
from huggingface_hub import hf_hub_download


class YOLOFaceDetector:
    def __init__(self, device="cpu", conf_threshold=0.5):
        # Auto-download the face model from HuggingFace
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt"
        )

        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray):
        """
        frame: BGR uint8 (OpenCV)
        returns: list of dicts [{x1, y1, x2, y2, score}, ...]
        """
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0].item())
                detections.append(
                    {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "score": score,
                    }
                )

        return detections
