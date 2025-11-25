import cv2
import time
import argparse
import torch

from detector.yolo_face import YOLOFaceDetector


def get_best_device():
    """Pick best device available: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                        help="Webcam '0' or path to video")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Max side length of resized frame")
    return parser.parse_args()


def resize_keep_aspect(frame, max_size):
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return frame
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def main():
    args = parse_args()

    # Select device automatically
    device = get_best_device()
    print(f"[info] Using device: {device}")

    # Open source
    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[error] Could not open video source: {args.source}")
        return

    # Init YOLOv8-face
    detector = YOLOFaceDetector(
        device=device,
        conf_threshold=0.5,
    )


    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize
        frame = resize_keep_aspect(frame, args.img_size)

        # Detect
        detections = detector.detect(frame)

        # Draw boxes
        for det in detections:
            x1, y1, x2, y2, score = det["x1"], det["y1"], det["x2"], det["y2"], det["score"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

        # FPS
        frame_count += 1
        if frame_count >= 10:
            now = time.time()
            fps = frame_count / (now - prev_time)
            prev_time = now
            frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv8-Face Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
