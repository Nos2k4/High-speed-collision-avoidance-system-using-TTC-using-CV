# TTC-ADAS — High-Speed Collision Avoidance System

Real-time **Time-To-Collision (TTC)** computation pipeline for ADAS research.
Designed for **high-speed scenarios** (80–200+ km/h). Runs on **GTX 1650 Ti**.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Camera /    │───▶│  YOLOv8 +   │───▶│   Depth      │───▶│  TTC Engine  │
│  Video File  │    │  DeepSORT   │    │  Estimation  │    │  (Kalman)    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                    │
                         ┌──────────────────────────────────────────┘
                         ▼
                 ┌──────────────┐
                 │  Racing HUD  │  SAFE / WARNING / DANGER / EMERGENCY
                 │  Overlay     │  + Track trails + TTC bar
                 └──────────────┘
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> CUDA PyTorch (install this FIRST if not already installed):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Calibrate your camera (important for accurate TTC!)

```bash
# Quick method — park a car 10 meters away, run and press SPACE
python utils/calibrate.py --method distance --known_dist 10.0 --known_class car

# Best method — print a checkerboard and run
python utils/calibrate.py --method checkerboard --cols 9 --rows 6
```

Copy the `focal_length_px` value into `config/settings.yaml`.

### 3. Run the system

```bash
# Webcam
python main.py

# Video file
python main.py --source path/to/highway_footage.mp4

# Set your driving speed (adjusts risk thresholds)
python main.py --source highway.mp4 --speed 120

# Debug mode — prints TTC values per frame to console
python main.py --source highway.mp4 --debug

# Use MiDaS neural depth (more accurate, uses GPU)
python main.py --depth midas

# Fastest mode (smaller model)
python main.py --model yolov8n.pt
```

### 4. Keyboard controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save snapshot |
| `+` | Increase ego speed (+20 km/h) |
| `-` | Decrease ego speed (−20 km/h) |

---

## Project Structure

```
ttc_adas/
├── main.py                  ← Entry point
├── requirements.txt
├── config/
│   └── settings.yaml        ← All tunable parameters
├── core/
│   ├── detector.py          ← YOLOv8 + DeepSORT tracker
│   ├── depth.py             ← Depth estimation (bbox / MiDaS)
│   └── ttc_engine.py        ← TTC computation + Kalman filter + risk zones
├── ui/
│   └── hud.py               ← Racing HUD overlay renderer
└── utils/
    ├── calibrate.py         ← Camera focal length calibration
    └── test_ttc.py          ← TTC unit tests + benchmarks
```

---

## How TTC works

```
TTC = distance / relative_approach_velocity
```

1. **Distance** is estimated per-object via bounding box height + focal length:
   ```
   distance = (real_object_height × focal_length_px) / bounding_box_height_px
   ```

2. **Relative velocity** is computed by fitting a line through the last 5 frames
   of distance measurements (least-squares regression → smooth, noise-robust).

3. **Kalman filter** smooths the final TTC value to remove depth noise jitter.

4. **Risk zones** (auto-scales at high speed):

   | Zone       | TTC         | Color  |
   |------------|-------------|--------|
   | SAFE       | > 4.0s      | Green  |
   | WARNING    | 2.0 – 4.0s  | Amber  |
   | DANGER     | 0.8 – 2.0s  | Red    |
   | EMERGENCY  | < 0.8s      | White + flash |

---

## Performance on GTX 1650 Ti

| Configuration                | Approx FPS |
|------------------------------|-----------|
| YOLOv8n + bbox depth         | 45–60 fps |
| YOLOv8s + bbox depth         | 30–45 fps |
| YOLOv8s + MiDaS small depth  | 18–25 fps |
| YOLOv8m + bbox depth         | 20–30 fps |

Recommended: `yolov8s + bounding_box` for best speed/accuracy tradeoff.

---

## Tuning for high-speed scenarios

In `config/settings.yaml`:

```yaml
risk:
  safe_threshold: 5.0        # Increase for highway speeds
  warning_threshold: 3.0
  critical_threshold: 2.0
  emergency_threshold: 1.2
  high_speed_multiplier: 1.5  # Auto-expands thresholds above 120 km/h

ttc:
  velocity_window: 8          # More frames for smoother velocity at high speed
```

---

## Extending the system

**Add radar/LiDAR depth**: Implement a new class in `core/depth.py` following
the `DepthEstimator` interface — `update_frame()` + `estimate()`.

**Add CAN bus output**: In `main.py`, hook into the risk level output from
`TTCEngine` and send brake commands via `python-can`.

**Add stereo depth**: Set `depth.method: stereo` in config and implement
`StereoDepthEstimator` using OpenCV's `StereoBM` or `StereoSGBM`.

**Train on custom data**: Replace `yolov8s.pt` with your own YOLOv8 model
trained on KITTI or nuScenes for better performance on highway vehicles.

---

## Datasets for testing

- **KITTI** — https://www.cvlibs.net/datasets/kitti/  (stereo + LiDAR + labels)
- **nuScenes** — https://nuscenes.org/  (full sensor suite)
- **BDD100K** — https://bdd-data.berkeley.edu/  (100k driving videos)
- **UA-DETRAC** — vehicle detection & tracking benchmark
