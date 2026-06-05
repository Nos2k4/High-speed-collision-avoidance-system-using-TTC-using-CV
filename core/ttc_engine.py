"""
TTC (Time-To-Collision) Computation
=====================================
Per-object TTC with Kalman-filtered velocity and smoothed estimates.

TTC = distance / relative_approach_velocity

For high-speed: we track rolling velocity over N frames and apply
a 1D Kalman filter on TTC output to reduce jitter from depth noise.

Risk zones (tunable via config):
  SAFE       TTC > 4.0s
  WARNING    TTC 2.0–4.0s
  DANGER     TTC 1.5–2.0s
  EMERGENCY  TTC < 0.8s
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import numpy as np
import time


class RiskLevel(Enum):
    UNKNOWN   = "UNKNOWN"
    SAFE      = "SAFE"
    WARNING   = "WARNING"
    DANGER    = "DANGER"
    EMERGENCY = "EMERGENCY"


RISK_COLORS_BGR = {
    RiskLevel.UNKNOWN:   (120, 120, 120),
    RiskLevel.SAFE:      (50,  220,  80),
    RiskLevel.WARNING:   (30,  180, 255),
    RiskLevel.DANGER:    (30,   80, 255),
    RiskLevel.EMERGENCY: (255, 255, 255),
}

RISK_COLORS_HEX = {
    RiskLevel.UNKNOWN:   "#787878",
    RiskLevel.SAFE:      "#32DC50",
    RiskLevel.WARNING:   "#FFB41E",
    RiskLevel.DANGER:    "#FF5020",
    RiskLevel.EMERGENCY: "#FFFFFF",
}


@dataclass
class TTCResult:
    track_id: int
    class_id: int
    class_name: str
    distance_m: float
    relative_velocity_ms: float   # positive = approaching
    ttc_raw: float                # seconds (raw)
    ttc_filtered: float           # seconds (Kalman-smoothed)
    risk: RiskLevel
    bbox: tuple                   # (x1, y1, x2, y2)
    timestamp: float = field(default_factory=time.time)

    @property
    def ttc_display(self) -> str:
        if self.ttc_filtered >= 99:
            return "∞"
        return f"{self.ttc_filtered:.1f}s"

    @property
    def speed_kmh(self) -> float:
        return self.relative_velocity_ms * 3.6


class KalmanTTC:
    """
    Simple 1D Kalman filter for scalar TTC value.
    Handles depth noise that causes TTC to spike / dip between frames.
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.Q = process_noise       # process noise covariance
        self.R = measurement_noise   # measurement noise covariance
        self.x = None                # state estimate
        self.P = 1.0                 # error covariance

    def update(self, measurement: float) -> float:
        if self.x is None:
            self.x = measurement
            return self.x

        # Predict
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)   # Kalman gain
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred

        return float(self.x)

    def reset(self):
        self.x = None
        self.P = 1.0


class ObjectTTCTracker:
    """
    Tracks TTC state for a single object across frames.
    Maintains a rolling window of (timestamp, distance) pairs
    to compute smooth relative velocity.
    """

    def __init__(self, track_id: int, window: int = 5,
                 kalman_Q: float = 0.01, kalman_R: float = 0.1):
        self.track_id = track_id
        self.window = window
        self.history: deque = deque(maxlen=window)   # (time, distance_m)
        self.kalman = KalmanTTC(process_noise=kalman_Q, measurement_noise=kalman_R)
        self.last_ttc: float = 99.0

    def update(self, distance_m: float, timestamp: float) -> tuple:
        """
        Returns (relative_velocity_ms, ttc_raw, ttc_filtered).
        positive velocity = approaching ego vehicle.
        """
        self.history.append((timestamp, distance_m))

        if len(self.history) < 2:
            return 0.0, 99.0, 99.0

        # Least-squares fit over history window for smooth velocity
        times  = np.array([h[0] for h in self.history])
        dists  = np.array([h[1] for h in self.history])
        times -= times[0]   # normalise to t=0

        if times[-1] < 1e-4:
            return 0.0, 99.0, 99.0

        # Linear regression: distance = a + b*t  →  b = velocity of object
        # If b < 0 → object getting closer → positive approach speed
        coeffs = np.polyfit(times, dists, 1)
        velocity_ms = -coeffs[0]   # approach velocity (positive = closing)

        # Current distance = latest measurement
        current_dist = distance_m

        # Raw TTC
        if velocity_ms > 0.5:   # only compute if meaningfully closing
            ttc_raw = current_dist / velocity_ms
            ttc_raw = float(np.clip(ttc_raw, 0.0, 99.0))
        else:
            ttc_raw = 99.0

        ttc_filtered = self.kalman.update(ttc_raw)
        ttc_filtered = float(np.clip(ttc_filtered, 0.0, 99.0))
        self.last_ttc = ttc_filtered

        return float(velocity_ms), ttc_raw, ttc_filtered


class TTCEngine:
    """
    Central TTC computation engine.
    Manages per-object trackers and computes risk levels.
    """

    COCO_CLASS_NAMES = {
        0: "person", 1: "bicycle", 2: "car",
        3: "motorcycle", 5: "bus", 7: "truck",
    }

    def __init__(self, config: dict):
        ttc_cfg = config.get("ttc", {})
        risk_cfg = config.get("risk", {})

        self.velocity_window   = ttc_cfg.get("velocity_window", 5)
        self.kalman_Q          = ttc_cfg.get("kalman_process_noise", 0.01)
        self.kalman_R          = ttc_cfg.get("kalman_measurement_noise", 0.1)
        self.min_approach_spd  = ttc_cfg.get("min_approach_speed", 0.5)

        self.t_safe      = risk_cfg.get("safe_threshold", 4.0)
        self.t_warning   = risk_cfg.get("warning_threshold", 2.0)
        self.t_danger    = risk_cfg.get("critical_threshold", 1.5)
        self.t_emergency = risk_cfg.get("emergency_threshold", 0.8)
        self.hs_mult     = risk_cfg.get("high_speed_multiplier", 1.4)

        self._trackers: Dict[int, ObjectTTCTracker] = {}

    def _get_tracker(self, track_id: int) -> ObjectTTCTracker:
        if track_id not in self._trackers:
            self._trackers[track_id] = ObjectTTCTracker(
                track_id=track_id,
                window=self.velocity_window,
                kalman_Q=self.kalman_Q,
                kalman_R=self.kalman_R,
            )
        return self._trackers[track_id]

    def _classify_risk(self, ttc: float, ego_speed_ms: float = 0.0) -> RiskLevel:
        # Expand thresholds at high speed (>33 m/s ≈ 120 km/h)
        mult = self.hs_mult if ego_speed_ms > 33.0 else 1.0
        if ttc >= self.t_safe * mult:
            return RiskLevel.SAFE
        elif ttc >= self.t_warning * mult:
            return RiskLevel.WARNING
        elif ttc >= self.t_emergency * mult:
            return RiskLevel.DANGER
        else:
            return RiskLevel.EMERGENCY

    def compute(
        self,
        track_id: int,
        class_id: int,
        bbox: tuple,
        distance_m: float,
        timestamp: float,
        ego_speed_ms: float = 0.0,
    ) -> TTCResult:
        tracker = self._get_tracker(track_id)
        vel_ms, ttc_raw, ttc_filt = tracker.update(distance_m, timestamp)
        risk = self._classify_risk(ttc_filt, ego_speed_ms)
        class_name = self.COCO_CLASS_NAMES.get(class_id, f"cls{class_id}")

        return TTCResult(
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            distance_m=distance_m,
            relative_velocity_ms=vel_ms,
            ttc_raw=ttc_raw,
            ttc_filtered=ttc_filt,
            risk=risk,
            bbox=bbox,
            timestamp=timestamp,
        )

    def purge_stale(self, active_ids: set):
        """Remove trackers for objects no longer in scene."""
        stale = [tid for tid in self._trackers if tid not in active_ids]
        for tid in stale:
            del self._trackers[tid]

    @property
    def most_critical(self) -> Optional[TTCResult]:
        """Return the TTCResult with the lowest TTC (most dangerous)."""
        if not self._trackers:
            return None
        # Find min TTC across all active trackers
        min_ttc = 99.0
        min_id = None
        for tid, tracker in self._trackers.items():
            if tracker.last_ttc < min_ttc:
                min_ttc = tracker.last_ttc
                min_id = tid
        return min_id
