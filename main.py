"""
TTC-ADAS Main Pipeline
========================
High-speed Time-To-Collision collision avoidance system.

Entry point — ties together:
  DetectionTracker → DepthEstimator → TTCEngine → HUDRenderer

Run:
    python main.py                          # webcam
    python main.py --source highway.mp4     # video file
    python main.py --source 0 --debug       # verbose debug mode
"""

import argparse
import sys
import time
import yaml
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.detector    import DetectionTracker
from core.depth       import DepthEstimator
from core.ttc_engine  import TTCEngine, TTCResult
from ui.hud           import HUDRenderer


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def open_capture(source, config: dict) -> cv2.VideoCapture:
    cam_cfg = config.get("camera", {})
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))
        cap.set(cv2.CAP_PROP_FPS,          cam_cfg.get("fps_target", 30))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimize latency

    return cap


class ADASystem:
    """Full ADAS pipeline orchestrator."""

    def __init__(self, config: dict, source, debug: bool = False, ego_speed_kmh: float = 0.0):
        self.config       = config
        self.source       = source
        self.debug        = debug
        self.ego_speed_ms = ego_speed_kmh / 3.6

        device = config.get("detection", {}).get("device", "cuda")

        self.detector  = DetectionTracker(config)
        self.depth_est = DepthEstimator(config.get("depth", {}), device=device)
        self.ttc_engine = TTCEngine(config)
        self.hud        = HUDRenderer(config)

        alerts_cfg = config.get("alerts", {})
        self.record = alerts_cfg.get("record_output", False)
        self.out_path = alerts_cfg.get("output_path", "output_ttc.mp4")
        self._writer: cv2.VideoWriter | None = None

        self._frame_count = 0
        self._t_start     = time.perf_counter()

    def _init_writer(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.out_path, fourcc, 25, (w, h))
        print(f"[Record] Writing to {self.out_path}")

    def run(self):
        cap = open_capture(self.source, self.config)
        print("\n╔══════════════════════════════════╗")
        print("║   TTC-ADAS  High-Speed System    ║")
        print("║   Press Q to quit  |  S = snap   ║")
        print("╚══════════════════════════════════╝\n")

        # Load models
        self.detector.load()

        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            t_frame = time.perf_counter()

            # ── Depth update (MiDaS, if enabled) ──────────────────────────
            self.depth_est.update_frame(frame)

            # ── Detect + Track ────────────────────────────────────────────
            detections = self.detector.run(frame)

            # ── TTC Computation ───────────────────────────────────────────
            ttc_results: list[TTCResult] = []
            active_ids = set()

            for det in detections:
                dist = self.depth_est.estimate(det.class_id, det.bbox)
                if dist is None:
                    continue

                result = self.ttc_engine.compute(
                    track_id=det.track_id,
                    class_id=det.class_id,
                    bbox=det.bbox,
                    distance_m=dist,
                    timestamp=t_frame,
                    ego_speed_ms=self.ego_speed_ms,
                )
                ttc_results.append(result)
                active_ids.add(det.track_id)

            self.ttc_engine.purge_stale(active_ids)

            # ── HUD Render ────────────────────────────────────────────────
            frame = self.hud.render(frame, ttc_results, ego_speed_ms=self.ego_speed_ms)

            # ── Debug print ───────────────────────────────────────────────
            if self.debug and ttc_results:
                for r in sorted(ttc_results, key=lambda x: x.ttc_filtered):
                    print(
                        f"  ID#{r.track_id:3d} {r.class_name:<12s} "
                        f"dist={r.distance_m:6.1f}m  "
                        f"vel={r.relative_velocity_ms:+5.1f}m/s  "
                        f"TTC={r.ttc_filtered:5.1f}s  "
                        f"[{r.risk.value}]"
                    )

            # ── Record ────────────────────────────────────────────────────
            if self.record:
                if self._writer is None:
                    self._init_writer(frame)
                self._writer.write(frame)

            # ── Display ───────────────────────────────────────────────────
            cv2.imshow("TTC-ADAS | High Speed Collision Avoidance", frame)

            self._frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snap_path = f"snapshot_{self._frame_count:05d}.jpg"
                cv2.imwrite(snap_path, frame)
                print(f"[Snap] Saved {snap_path}")
            elif key == ord("+") or key == ord("="):
                self.ego_speed_ms = min(self.ego_speed_ms + 5.556, 83.3)  # +20 km/h
                print(f"[Ego speed] {self.ego_speed_ms * 3.6:.0f} km/h")
            elif key == ord("-"):
                self.ego_speed_ms = max(0.0, self.ego_speed_ms - 5.556)
                print(f"[Ego speed] {self.ego_speed_ms * 3.6:.0f} km/h")

        cap.release()
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()

        elapsed = time.perf_counter() - self._t_start
        avg_fps = self._frame_count / elapsed if elapsed > 0 else 0
        print(f"\n[Done] {self._frame_count} frames in {elapsed:.1f}s  ({avg_fps:.1f} fps avg)")


def main():
    parser = argparse.ArgumentParser(
        description="TTC-ADAS High-Speed Collision Avoidance System"
    )
    parser.add_argument(
        "--source", default=None,
        help="Video source: camera index (0,1) or path to video file"
    )
    parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to settings YAML"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print per-frame TTC values to console"
    )
    parser.add_argument(
        "--speed", type=float, default=0.0,
        help="Ego vehicle speed in km/h (affects risk thresholds)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Override YOLO model: yolov8n/s/m/l"
    )
    parser.add_argument(
        "--depth", default=None,
        choices=["bounding_box", "midas"],
        help="Override depth method"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.source is not None:
        config["camera"]["source"] = args.source
    if args.model is not None:
        config["detection"]["model"] = args.model
    if args.depth is not None:
        config["depth"]["method"] = args.depth

    source = config["camera"].get("source", 0)

    system = ADASystem(
        config=config,
        source=source,
        debug=args.debug,
        ego_speed_kmh=args.speed,
    )
    system.run()


if __name__ == "__main__":
    main()
