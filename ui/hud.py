"""
HUD Overlay Renderer
=====================
Racing-inspired high-contrast HUD for high-speed ADAS.

Design language:
  - Black background panels with neon accent colors
  - Per-object colored bounding boxes (green / amber / red / white)
  - TTC bar on the right: stacked colored segments per object
  - Top-left: FPS, frame time, object count
  - Bottom center: most-critical alert banner with flash
  - Track trails (last N positions)
  - Corner speed indicators
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

from core.ttc_engine import TTCResult, RiskLevel, RISK_COLORS_BGR


# ── Color palette (BGR) ──────────────────────────────────────────────────────
C_BG         = (15, 15, 20)
C_PANEL      = (25, 28, 35)
C_BORDER     = (60, 65, 80)
C_TEXT_DIM   = (130, 130, 145)
C_TEXT_HI    = (220, 225, 235)
C_ACCENT     = (0, 200, 255)     # cyan
C_SAFE       = (50, 220, 80)
C_WARNING    = (30, 180, 255)
C_DANGER     = (30, 80, 255)
C_EMERGENCY  = (255, 255, 255)

RISK_BGR = {
    RiskLevel.UNKNOWN:   C_TEXT_DIM,
    RiskLevel.SAFE:      C_SAFE,
    RiskLevel.WARNING:   C_WARNING,
    RiskLevel.DANGER:    C_DANGER,
    RiskLevel.EMERGENCY: C_EMERGENCY,
}

RISK_LABEL = {
    RiskLevel.UNKNOWN:   "---",
    RiskLevel.SAFE:      "SAFE",
    RiskLevel.WARNING:   "WARNING",
    RiskLevel.DANGER:    "DANGER",
    RiskLevel.EMERGENCY: "EMERGENCY",
}


def _alpha_blend(
    frame: np.ndarray,
    overlay: np.ndarray,
    alpha: float,
    x: int, y: int
) -> np.ndarray:
    """Blend overlay patch onto frame at (x, y)."""
    h, w = overlay.shape[:2]
    fh, fw = frame.shape[:2]
    x2 = min(x + w, fw); y2 = min(y + h, fh)
    ow = x2 - x; oh = y2 - y
    if ow <= 0 or oh <= 0:
        return frame
    roi = frame[y:y2, x:x2].astype(np.float32)
    ovr = overlay[:oh, :ow].astype(np.float32)
    blended = cv2.addWeighted(roi, 1 - alpha, ovr, alpha, 0)
    frame[y:y2, x:x2] = blended.astype(np.uint8)
    return frame


def _panel(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    color: tuple = C_PANEL,
    alpha: float = 0.75,
    border: bool = True,
    border_color: tuple = C_BORDER,
):
    """Draw a semi-transparent dark panel."""
    patch = np.full((h, w, 3), color, dtype=np.uint8)
    _alpha_blend(frame, patch, alpha, x, y)
    if border:
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 1)


def _text(
    frame: np.ndarray,
    text: str,
    x: int, y: int,
    color: tuple = C_TEXT_HI,
    scale: float = 0.55,
    thickness: int = 1,
    font = cv2.FONT_HERSHEY_SIMPLEX,
):
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


class TrailRenderer:
    """Renders track history trails behind each object."""

    def __init__(self, max_len: int = 25):
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, track_id: int, center: Tuple[int, int]):
        self.trails[track_id].append(center)

    def render(self, frame: np.ndarray, ttc_results: Dict[int, TTCResult]):
        for tid, pts in self.trails.items():
            if len(pts) < 2:
                continue
            color = RISK_BGR.get(
                ttc_results[tid].risk if tid in ttc_results else RiskLevel.UNKNOWN,
                C_TEXT_DIM,
            )
            pts_list = list(pts)
            for i in range(1, len(pts_list)):
                alpha = i / len(pts_list)
                faded = tuple(int(c * alpha * 0.7) for c in color)
                cv2.line(frame, pts_list[i - 1], pts_list[i], faded, 1, cv2.LINE_AA)

    def purge(self, active_ids: set):
        stale = [tid for tid in self.trails if tid not in active_ids]
        for tid in stale:
            del self.trails[tid]


class HUDRenderer:
    """
    Full HUD compositor.
    Call render() each frame with the list of TTC results.
    """

    def __init__(self, config: dict):
        display_cfg = config.get("display", {})
        self.show_fps    = display_cfg.get("show_fps", True)
        self.show_tracks = display_cfg.get("show_tracks", True)
        self.show_ttc_bar = display_cfg.get("show_ttc_bar", True)
        self.trail_len   = display_cfg.get("trail_length", 20)

        self.trail_renderer = TrailRenderer(max_len=self.trail_len)

        # FPS tracking
        self._frame_times: deque = deque(maxlen=60)
        self._last_t = time.perf_counter()

        # Emergency flash state
        self._flash_phase = 0.0

    def _update_fps(self) -> float:
        now = time.perf_counter()
        self._frame_times.append(now - self._last_t)
        self._last_t = now
        if len(self._frame_times) < 2:
            return 0.0
        return 1.0 / (sum(self._frame_times) / len(self._frame_times))

    def render(
        self,
        frame: np.ndarray,
        ttc_results: List[TTCResult],
        ego_speed_ms: float = 0.0,
    ) -> np.ndarray:
        fps = self._update_fps()
        h, w = frame.shape[:2]

        results_by_id = {r.track_id: r for r in ttc_results}
        active_ids    = set(results_by_id.keys())

        # ── Emergency flash ──────────────────────────────────────────────────
        has_emergency = any(r.risk == RiskLevel.EMERGENCY for r in ttc_results)
        if has_emergency:
            self._flash_phase += 0.35
            flash_alpha = abs(np.sin(self._flash_phase)) * 0.25
            overlay = np.full_like(frame, (0, 0, 220))
            frame = cv2.addWeighted(frame, 1 - flash_alpha, overlay, flash_alpha, 0)
        else:
            self._flash_phase = 0.0

        # ── Track trails ─────────────────────────────────────────────────────
        for r in ttc_results:
            x1, y1, x2, y2 = [int(v) for v in r.bbox]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.trail_renderer.update(r.track_id, (cx, cy))
        self.trail_renderer.render(frame, results_by_id)
        self.trail_renderer.purge(active_ids)

        # ── Bounding boxes + per-object labels ───────────────────────────────
        for r in ttc_results:
            x1, y1, x2, y2 = [int(v) for v in r.bbox]
            color = RISK_BGR[r.risk]

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Corner ticks (racing style)
            tick = 12
            for (px, py), (dx, dy) in [
                ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))
            ]:
                cv2.line(frame, (px, py), (px + dx * tick, py), color, 2, cv2.LINE_AA)
                cv2.line(frame, (px, py), (px, py + dy * tick), color, 2, cv2.LINE_AA)

            # Label panel
            label_h = 52 if r.risk != RiskLevel.SAFE else 40
            lx = x1; ly = max(0, y1 - label_h - 2)
            lw = max(140, x2 - x1)
            _panel(frame, lx, ly, lw, label_h, alpha=0.82)

            # Class + track ID
            _text(frame, f"#{r.track_id} {r.class_name.upper()}",
                  lx + 6, ly + 14, color=C_TEXT_HI, scale=0.48)
            # Distance
            _text(frame, f"{r.distance_m:.1f}m",
                  lx + 6, ly + 28, color=C_ACCENT, scale=0.52)
            # TTC
            if r.risk != RiskLevel.SAFE:
                ttc_color = RISK_BGR[r.risk]
                _text(frame, f"TTC {r.ttc_display}",
                      lx + 6, ly + 43, color=ttc_color, scale=0.55, thickness=1)

        # ── Top-left info panel ───────────────────────────────────────────────
        _panel(frame, 10, 10, 200, 72, alpha=0.80)
        _text(frame, f"FPS  {fps:5.1f}", 18, 30, color=C_ACCENT, scale=0.55)
        _text(frame, f"OBJS {len(ttc_results):3d}",  18, 48, color=C_TEXT_HI, scale=0.50)
        ego_kmh = ego_speed_ms * 3.6
        _text(frame, f"EGO  {ego_kmh:5.1f} km/h", 18, 66, color=C_TEXT_DIM, scale=0.46)

        # ── TTC bar (right side) ──────────────────────────────────────────────
        if self.show_ttc_bar and ttc_results:
            bar_x = w - 130
            bar_y = 10
            bar_w = 118
            bar_h = min(h - 20, 30 + len(ttc_results) * 50)
            _panel(frame, bar_x - 4, bar_y, bar_w + 4, bar_h, alpha=0.80)
            _text(frame, "TTC", bar_x + 36, bar_y + 14, color=C_ACCENT, scale=0.50)

            sorted_results = sorted(ttc_results, key=lambda r: r.ttc_filtered)
            for i, r in enumerate(sorted_results[:6]):
                oy = bar_y + 22 + i * 44
                color = RISK_BGR[r.risk]
                # Mini progress bar
                max_ttc = 6.0
                fill_pct = 1.0 - min(1.0, r.ttc_filtered / max_ttc)
                fill_w = int((bar_w - 12) * fill_pct)
                cv2.rectangle(frame, (bar_x, oy + 18), (bar_x + bar_w - 12, oy + 26), C_BORDER, -1)
                if fill_w > 0:
                    cv2.rectangle(frame, (bar_x, oy + 18), (bar_x + fill_w, oy + 26), color, -1)
                _text(frame, f"#{r.track_id}", bar_x, oy + 12, color=C_TEXT_DIM, scale=0.40)
                _text(frame, r.ttc_display, bar_x + 30, oy + 12, color=color, scale=0.46)
                _text(frame, f"{r.distance_m:.0f}m", bar_x + 72, oy + 12, color=C_TEXT_DIM, scale=0.38)

        # ── Critical alert banner (bottom center) ────────────────────────────
        worst = min(ttc_results, key=lambda r: r.ttc_filtered) if ttc_results else None
        if worst and worst.risk in (RiskLevel.WARNING, RiskLevel.DANGER, RiskLevel.EMERGENCY):
            bw = 420; bh = 56
            bx = (w - bw) // 2; by = h - bh - 14
            border_c = RISK_BGR[worst.risk]
            _panel(frame, bx, by, bw, bh, alpha=0.88)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), border_c, 2)
            cv2.line(frame, (bx, by), (bx + bw, by), border_c, 3)

            risk_label = RISK_LABEL[worst.risk]
            _text(frame, risk_label, bx + 16, by + 22,
                  color=RISK_BGR[worst.risk], scale=0.70, thickness=2)
            _text(frame,
                  f"#{worst.track_id} {worst.class_name}  "
                  f"{worst.distance_m:.1f}m  TTC {worst.ttc_display}  "
                  f"{worst.speed_kmh:.0f} km/h",
                  bx + 16, by + 44, color=C_TEXT_HI, scale=0.46)

        return frame
