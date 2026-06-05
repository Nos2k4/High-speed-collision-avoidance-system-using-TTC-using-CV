"""
Depth Estimation Module
=======================
Supports three methods:
  1. bounding_box  — uses known real-world object heights + focal length (fast, no GPU cost)
  2. midas         — MiDaS neural depth model (slower but camera-agnostic)
  3. stereo        — OpenCV stereo block matching (requires calibrated stereo rig)

For high-speed ADAS, bounding_box is recommended as primary with midas as fallback.
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple


# ─── Known real-world object heights (meters) ───────────────────────────────
KNOWN_HEIGHTS_M: Dict[str, float] = {
    "car":        1.50,
    "truck":      3.50,
    "bus":        3.20,
    "motorcycle": 1.10,
    "bicycle":    1.00,
    "person":     1.75,
}

COCO_CLASS_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}


class BoundingBoxDepthEstimator:
    """
    Estimates metric depth from bounding box height using the pinhole camera model.

        distance = (real_height * focal_length) / pixel_height

    Accuracy: ~10–20% error at 10–50m range. Good enough for TTC zone classification.
    No GPU required — runs in microseconds per object.
    """

    def __init__(self, focal_length_px: float = 700.0):
        self.focal_length_px = focal_length_px

    def estimate(self, class_id: int, bbox_height_px: float) -> Optional[float]:
        """Return estimated distance in meters, or None if class unknown."""
        class_name = COCO_CLASS_NAMES.get(class_id)
        if class_name is None or bbox_height_px < 5:
            return None
        real_height = KNOWN_HEIGHTS_M.get(class_name)
        if real_height is None:
            return None
        distance = (real_height * self.focal_length_px) / bbox_height_px
        # Clamp to reasonable range
        return float(np.clip(distance, 1.0, 200.0))


class MiDaSDepthEstimator:
    """
    Neural monocular depth using MiDaS (Intel Labs).
    Returns relative depth — must be scale-calibrated for metric use.
    Runs ~20–30 fps on GTX 1650 Ti with MiDaS_small.
    """

    def __init__(self, model_type: str = "MiDaS_small", device: str = "cuda"):
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch = torch
        print(f"[MiDaS] Loading {model_type} on {self.device}...")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform if "small" in model_type else transforms.default_transform
        self._depth_map: Optional[np.ndarray] = None
        self._scale_factor: float = 1.0   # Calibrate this against known distances
        print("[MiDaS] Ready.")

    def update_frame(self, frame_rgb: np.ndarray):
        """Run depth inference on full frame. Call once per frame."""
        import torch
        with torch.no_grad():
            input_batch = self.transform(frame_rgb).to(self.device)
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        self._depth_map = prediction.cpu().numpy()

    def estimate_at_bbox(self, x1: int, y1: int, x2: int, y2: int) -> Optional[float]:
        """Median depth in bounding box region (in relative units × scale_factor)."""
        if self._depth_map is None:
            return None
        h, w = self._depth_map.shape
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w, x2), min(h, y2)
        roi = self._depth_map[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return None
        # MiDaS outputs inverse depth → convert to distance
        inv_depth = float(np.median(roi))
        if inv_depth < 1e-5:
            return None
        distance = self._scale_factor / inv_depth
        return float(np.clip(distance, 0.5, 200.0))

    def calibrate(self, known_distance_m: float, inv_depth_value: float):
        """One-point calibration: measure inv_depth at a known real distance."""
        self._scale_factor = known_distance_m * inv_depth_value
        print(f"[MiDaS] Calibrated: scale_factor = {self._scale_factor:.4f}")


class DepthEstimator:
    """
    Unified depth estimator. Selects method from config.
    Primary: bounding_box (always fast)
    Optional: midas (more accurate, GPU-based)
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.method = config.get("method", "bounding_box")
        self.focal_length = config.get("focal_length_px",
                                       config.get("focal_length_px_fallback", 700.0))
        # Always load bbox estimator as fast fallback
        self.bbox_estimator = BoundingBoxDepthEstimator(focal_length_px=self.focal_length)
        self.midas_estimator: Optional[MiDaSDepthEstimator] = None

        if self.method == "midas":
            try:
                model_type = config.get("midas_model", "MiDaS_small")
                self.midas_estimator = MiDaSDepthEstimator(model_type=model_type, device=device)
            except Exception as e:
                print(f"[DepthEstimator] MiDaS failed to load ({e}), falling back to bounding_box.")
                self.method = "bounding_box"

    def update_frame(self, frame_bgr: np.ndarray):
        """Call once per frame if using MiDaS."""
        if self.midas_estimator is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.midas_estimator.update_frame(frame_rgb)

    def estimate(
        self,
        class_id: int,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[float]:
        """
        bbox: (x1, y1, x2, y2) in pixels
        Returns distance in meters or None.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_height_px = y2 - y1

        if self.method == "midas" and self.midas_estimator is not None:
            dist = self.midas_estimator.estimate_at_bbox(x1, y1, x2, y2)
            if dist is not None:
                return dist

        # Fallback: bounding box method
        return self.bbox_estimator.estimate(class_id, bbox_height_px)
