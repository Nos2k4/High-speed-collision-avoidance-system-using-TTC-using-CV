"""
Detection & Tracking Pipeline
================================
YOLOv8 detection + DeepSORT tracking.
Outputs confirmed tracks with bounding boxes and class IDs.

Optimized for high-speed: low latency model inference with
half-precision (FP16) when CUDA is available.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class Detection:
    track_id: int
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2


class DetectionTracker:
    """
    YOLOv8 + DeepSORT pipeline.
    Outputs stable per-frame track list.
    """

    def __init__(self, config: dict):
        det_cfg   = config.get("detection", {})
        track_cfg = config.get("tracking", {})

        self.model_path   = det_cfg.get("model", "yolov8s.pt")
        self.confidence   = det_cfg.get("confidence", 0.40)
        self.iou          = det_cfg.get("iou_threshold", 0.45)
        self.device       = det_cfg.get("device", "cuda")
        self.img_size     = det_cfg.get("img_size", 640)
        self.classes      = det_cfg.get("classes", [0, 1, 2, 3, 5, 7])

        self.max_age      = track_cfg.get("max_age", 30)
        self.n_init       = track_cfg.get("n_init", 2)
        self.max_cos_dist = track_cfg.get("max_cosine_distance", 0.4)
        self.nn_budget    = track_cfg.get("nn_budget", 100)

        self._model  = None
        self._tracker = None
        self._loaded  = False

    def load(self):
        """Lazy-load models (call once before first frame)."""
        if self._loaded:
            return

        print("[Detector] Loading YOLOv8...")
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, device=self.device, verbose=False)
        print(f"[Detector] YOLOv8 ready on {self.device}")

        print("[Tracker] Loading DeepSORT...")
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_cosine_distance=self.max_cos_dist,
            nn_budget=self.nn_budget,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
        )
        print("[Tracker] DeepSORT ready")
        self._loaded = True

    def run(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection + tracking on a single BGR frame.
        Returns list of Detection objects with stable track IDs.
        """
        if not self._loaded:
            self.load()

        # ── YOLOv8 Inference ──
        results = self._model.predict(
            frame,
            device=self.device,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.img_size,
            classes=self.classes,
            verbose=False,
            half=True,          # FP16 for speed on CUDA
        )

        # ── Format detections for DeepSORT ──
        # DeepSORT expects: list of ([x, y, w, h], confidence, class_id)
        raw_detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf  = float(box.conf[0].cpu().numpy())
                cls   = int(box.cls[0].cpu().numpy())
                w = x2 - x1
                h = y2 - y1
                raw_detections.append(([x1, y1, w, h], conf, cls))

        if not raw_detections:
            return []

        # ── DeepSORT Update ──
        tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        detections: List[Detection] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = ltrb
            # Clamp to frame bounds
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_frame, x2); y2 = min(h_frame, y2)

            det_cls = track.det_class if track.det_class is not None else -1
            det_conf = track.det_conf if track.det_conf is not None else 0.0

            detections.append(Detection(
                track_id=int(track.track_id),
                class_id=int(det_cls),
                confidence=float(det_conf) if det_conf else 0.5,
                bbox=(x1, y1, x2, y2),
            ))

        return detections
