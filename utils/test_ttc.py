"""
TTC Engine Test & Benchmark
==============================
Tests the TTC math, Kalman filter, and risk zones WITHOUT a camera.
Simulates a car approaching at various speeds.

Run: python utils/test_ttc.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.table   import Table
from rich.live    import Live

from core.ttc_engine import TTCEngine, RiskLevel

console = Console()


def simulate_scenario(name: str, initial_dist: float, approach_speed_ms: float,
                       ego_speed_ms: float = 0.0, noise_std: float = 0.5):
    """
    Simulate an object approaching at constant speed with Gaussian depth noise.
    Returns list of (frame, ttc_true, ttc_filtered, risk) tuples.
    """
    config = {
        "ttc": {
            "velocity_window": 5,
            "kalman_process_noise": 0.01,
            "kalman_measurement_noise": 0.1,
            "min_approach_speed": 0.5,
        },
        "risk": {
            "safe_threshold": 4.0,
            "warning_threshold": 2.0,
            "critical_threshold": 1.5,
            "emergency_threshold": 0.8,
            "high_speed_multiplier": 1.4,
        }
    }
    engine  = TTCEngine(config)
    results = []
    dt      = 1.0 / 30.0   # 30 fps
    dist    = initial_dist

    true_ttc = initial_dist / approach_speed_ms if approach_speed_ms > 0 else 99.0

    for frame in range(int(true_ttc / dt) + 10):
        t = frame * dt
        dist = max(0.1, initial_dist - approach_speed_ms * t)
        noisy_dist = dist + np.random.normal(0, noise_std)
        noisy_dist = max(0.5, noisy_dist)

        result = engine.compute(
            track_id=1,
            class_id=2,   # car
            bbox=(100, 100, 300, 300),
            distance_m=noisy_dist,
            timestamp=t,
            ego_speed_ms=ego_speed_ms,
        )
        ttc_true = dist / approach_speed_ms if approach_speed_ms > 0 else 99.0
        results.append((frame, ttc_true, result.ttc_filtered, result.risk))
        if dist <= 0.5:
            break

    return results


def test_scenarios():
    console.print("\n[bold cyan]TTC Engine — Scenario Tests[/bold cyan]\n")

    scenarios = [
        ("Highway approach (30 m/s = 108 km/h)",  50.0, 30.0,  30.0, 1.0),
        ("City approach  (14 m/s = 50 km/h)",     30.0, 14.0,  0.0,  0.5),
        ("Very fast approach (50 m/s = 180 km/h)", 80.0, 50.0, 50.0, 1.5),
        ("Slow approach (5 m/s = 18 km/h)",        15.0,  5.0,  0.0,  0.3),
        ("Static object (0 m/s relative)",         10.0,  0.0,  0.0,  0.2),
    ]

    for name, d0, v_approach, ego_v, noise in scenarios:
        results = simulate_scenario(name, d0, v_approach, ego_v, noise)
        console.print(f"[bold]{name}[/bold]")
        console.print(f"  d₀={d0}m  v_approach={v_approach} m/s  noise σ={noise}m")

        # Show key frames: entry into each risk zone
        shown_zones = set()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Frame", style="dim", width=6)
        table.add_column("TTC true (s)", width=12)
        table.add_column("TTC filtered (s)", width=16)
        table.add_column("Risk", width=10)

        for (frame, ttc_true, ttc_filt, risk) in results:
            if risk not in shown_zones or frame % 30 == 0:
                color = {
                    RiskLevel.SAFE:      "green",
                    RiskLevel.WARNING:   "yellow",
                    RiskLevel.DANGER:    "red",
                    RiskLevel.EMERGENCY: "bold white on red",
                    RiskLevel.UNKNOWN:   "dim",
                }[risk]
                table.add_row(
                    str(frame),
                    f"{ttc_true:.2f}" if ttc_true < 90 else "∞",
                    f"{ttc_filt:.2f}" if ttc_filt < 90 else "∞",
                    f"[{color}]{risk.value}[/{color}]",
                )
                shown_zones.add(risk)
                if len(shown_zones) == 5:
                    break

        console.print(table)
        console.print()


def benchmark_engine(n_objects: int = 20, n_frames: int = 500):
    """Benchmark TTC engine throughput (excludes CV models)."""
    console.print(f"[bold cyan]TTC Engine Benchmark[/bold cyan]  "
                  f"({n_objects} objects × {n_frames} frames)\n")

    config = {
        "ttc": {"velocity_window": 5, "kalman_process_noise": 0.01,
                "kalman_measurement_noise": 0.1, "min_approach_speed": 0.5},
        "risk": {"safe_threshold": 4.0, "warning_threshold": 2.0,
                 "critical_threshold": 1.5, "emergency_threshold": 0.8,
                 "high_speed_multiplier": 1.4},
    }
    engine = TTCEngine(config)
    rng = np.random.default_rng(42)
    dists = rng.uniform(5, 80, n_objects)

    t0 = time.perf_counter()
    for frame in range(n_frames):
        t = frame / 30.0
        dists = np.maximum(1.0, dists - rng.uniform(0.1, 1.0, n_objects) / 30.0)
        for i in range(n_objects):
            noisy = dists[i] + rng.normal(0, 0.3)
            engine.compute(i, 2, (0, 0, 100, 100), float(noisy), t, 0.0)

    elapsed = time.perf_counter() - t0
    total   = n_frames * n_objects
    console.print(f"  {total} TTC computations in {elapsed*1000:.1f} ms")
    console.print(f"  {total/elapsed:,.0f} computations/sec")
    console.print(f"  {elapsed/n_frames*1000:.3f} ms per frame (engine only)\n")


if __name__ == "__main__":
    np.random.seed(42)
    test_scenarios()
    benchmark_engine()
    console.print("[bold green]All tests passed![/bold green]")
