"""
Camera Calibration Utility
============================
Run this ONCE to calibrate your camera's focal length.
This is critical for accurate metric depth (and thus accurate TTC).

Method A — Checkerboard (most accurate):
    python utils/calibrate.py --method checkerboard --cols 9 --rows 6

Method B — Single known distance (quick):
    python utils/calibrate.py --method distance --known_dist 10.0 --known_class car

Usage:
    - For checkerboard: print a 9×6 checkerboard, hold in front of camera
    - For distance: park a car 10m from camera, press SPACE to capture
    - Results are saved to config/camera_calibration.yaml
"""

import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path

KNOWN_HEIGHTS = {
    "car": 1.50, "truck": 3.50, "bus": 3.20,
    "motorcycle": 1.10, "person": 1.75,
}


def calibrate_checkerboard(source=0, cols=9, rows=6, square_size_m=0.025):
    """Full intrinsic calibration using checkerboard pattern."""
    cap = cv2.VideoCapture(source)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m

    obj_points = []
    img_points = []
    captured   = 0
    target     = 20

    print(f"[Calibrate] Checkerboard {cols}×{rows}")
    print(f"  Move the board around. Press SPACE to capture. Need {target} samples.")
    print("  Press Q to finish early (min 10 samples).\n")

    while captured < target:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        disp = frame.copy()
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(disp, (cols, rows), corners2, found)
            cv2.putText(disp, f"Board found! SPACE to capture ({captured}/{target})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "Looking for checkerboard...",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        cv2.imshow("Calibration", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and found:
            obj_points.append(objp)
            img_points.append(corners2)
            captured += 1
            print(f"  Captured {captured}/{target}")
        elif key == ord("q") and captured >= 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured < 5:
        print("[Error] Not enough samples. Aborting.")
        return

    print("\n[Calibrate] Computing intrinsics...")
    h, w = frame.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])
    error = ret

    print(f"  Reprojection error: {error:.4f} px")
    print(f"  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

    out = {
        "focal_length_px": fx,
        "fy": fy, "cx": cx, "cy": cy,
        "reprojection_error": float(error),
        "distortion": dist.tolist(),
        "method": "checkerboard",
    }
    Path("config").mkdir(exist_ok=True)
    with open("config/camera_calibration.yaml", "w") as f:
        yaml.dump(out, f)
    print("\n[Saved] config/camera_calibration.yaml")
    print(f"  → Set focal_length_px: {fx:.1f} in config/settings.yaml")


def calibrate_distance(source=0, known_dist_m=10.0, known_class="car"):
    """Quick single-distance calibration."""
    real_height = KNOWN_HEIGHTS.get(known_class, 1.5)
    cap = cv2.VideoCapture(source)
    print(f"\n[Calibrate] Distance method")
    print(f"  Place a {known_class} at exactly {known_dist_m}m from camera.")
    print(f"  Press SPACE when the object fills the bounding box well. Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        h_frame = frame.shape[0]
        cv2.putText(disp,
                    f"Place {known_class} at {known_dist_m}m. SPACE to capture.",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp,
                    "Draw a box: click top-left then bottom-right of the object.",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Calibration", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            # Let user select ROI
            roi = cv2.selectROI("Select the object tightly", frame, False, False)
            cv2.destroyWindow("Select the object tightly")
            if roi[3] > 0:
                pixel_height = roi[3]
                focal_length = (real_height * h_frame) / (pixel_height * (known_dist_m / h_frame))
                focal_length = (real_height * h_frame * known_dist_m) / (pixel_height * known_dist_m)
                focal_length = (real_height * known_dist_m) / (pixel_height / h_frame * known_dist_m)
                # Correct formula:
                focal_length = (real_height * known_dist_m * h_frame) / (pixel_height * known_dist_m)
                # Simplified:
                focal_length = (real_height / pixel_height) * h_frame / (1 / known_dist_m)
                focal_length = (real_height * known_dist_m * h_frame) / pixel_height
                # Pinhole: f = (H_real * f) / H_pixel  → f = (H_real * d) / H_pixel * (1 / d) * d
                focal_length = (real_height * known_dist_m) / (pixel_height / h_frame) / known_dist_m
                # Direct pinhole camera formula:
                # pixel_height = (real_height * focal_px) / distance_m
                # → focal_px = pixel_height * distance_m / real_height
                focal_px = pixel_height * known_dist_m / real_height
                print(f"\n  Object pixel height: {pixel_height}px")
                print(f"  Estimated focal_length_px: {focal_px:.1f}")
                out = {
                    "focal_length_px": float(focal_px),
                    "method": "single_distance",
                    "known_dist_m": known_dist_m,
                    "known_class": known_class,
                }
                Path("config").mkdir(exist_ok=True)
                with open("config/camera_calibration.yaml", "w") as f:
                    yaml.dump(out, f)
                print("[Saved] config/camera_calibration.yaml")
                print(f"  → Set focal_length_px: {focal_px:.1f} in config/settings.yaml")
                break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["checkerboard", "distance"], default="distance")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--cols",  type=int, default=9)
    parser.add_argument("--rows",  type=int, default=6)
    parser.add_argument("--square_size", type=float, default=0.025, help="Checkerboard square size in meters")
    parser.add_argument("--known_dist",  type=float, default=10.0,  help="Known object distance in meters")
    parser.add_argument("--known_class", default="car", choices=list(KNOWN_HEIGHTS.keys()))
    args = parser.parse_args()

    if args.method == "checkerboard":
        calibrate_checkerboard(args.source, args.cols, args.rows, args.square_size)
    else:
        calibrate_distance(args.source, args.known_dist, args.known_class)
