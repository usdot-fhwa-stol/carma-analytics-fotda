#!/usr/bin/env python3
"""CP-01: FLIR camera detection drop rate, pooled across recording sessions.

The CP-01 test plan spans two sessions -- 5/10/15-second dwells recorded on
2026-09-14 and 20-second dwells on 2026-09-15 -- and asks for a single pooled
figure, so this takes one or more session directories::

    python run_cp01_analysis.py \\
        --data-root .../20260914_verification_test \\
        --data-root .../20260915_verification_test \\
        --output-dir out/cp01

Writes ``cp01_detection_drops.json``, ``cp01_detection_drops.csv`` (one row per
run) and a per-condition plot. See ``cp01_detection_drops`` for how each run's
dwell window is located and why it is measured on the Kafka topic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cp01_detection_drops as cp01  # noqa: E402
import dt_wz_dataset as dataset  # noqa: E402
from dt_wz_cascade.plots import stage_colors  # noqa: E402


def analyse(data_roots: List[Path], rate_hz: float) -> List[cp01.RunDrops]:
    results: List[cp01.RunDrops] = []
    for root in data_roots:
        runs = dataset.load_runs_csv(root / "runs.csv", root)
        session = dataset.discover_session(root)
        if session.pc2_v2xhub is None:
            raise FileNotFoundError(
                f"{root}: no pc2 V2XHub log; CP-01 counts the camera's websocket stream"
            )
        print(f"{root.name}: {len(runs)} runs, camera frames from "
              f"{session.pc2_v2xhub.name}")
        results.extend(cp01.analyse_session(runs, session, rate_hz))
    return results


def plot_drops(results: List[cp01.RunDrops], summary, output_path):
    """Per-run drop counts, grouped by dwell condition."""
    if not results:
        return None
    conditions = list(summary["by_condition"])
    colors = dict(zip(conditions, stage_colors(range(max(len(conditions), 2)))))

    figure, axis = plt.subplots(figsize=(11, 4.6))
    labels = [item.run for item in results]
    values = [item.dropped_frames for item in results]
    axis.bar(
        range(len(results)), values,
        color=[colors[item.condition] for item in results],
        edgecolor="white", linewidth=0.8,
    )
    for index, item in enumerate(results):
        if item.dropped_frames:
            axis.text(index, item.dropped_frames, f"{item.dropped_frames}",
                      ha="center", va="bottom", fontsize=7, color="#333333")

    axis.set_xticks(range(len(results)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("Camera frames dropped")
    axis.set_title(
        f"CP-01: FLIR camera detection drops — {summary['headline']} "
        f"({summary['total_drop_pct']:.2f}%)",
        loc="left", fontsize=11,
    )
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=7, color=colors[condition],
                   label=f"{condition} ({summary['by_condition'][condition]['runs']} runs)")
        for condition in conditions
    ]
    axis.legend(handles=handles, frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session directory with runs.csv; repeat to pool sessions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=cp01.DETECTION_RATE_HZ,
                        help=f"Nominal camera frame rate (default {cp01.DETECTION_RATE_HZ})")
    args = parser.parse_args(argv)

    try:
        results = analyse(args.data_root, args.rate_hz)
        summary = cp01.summarise(results, args.rate_hz)
        summary["sessions"] = [str(root) for root in args.data_root]

        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "cp01_detection_drops.json"
        json_path.write_text(json.dumps(summary, indent=2))

        csv_path = args.output_dir / "cp01_detection_drops.csv"
        rows = summary["runs_detail"]
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        plot_path = args.output_dir / "cp01_detection_drops.png"
        plot_drops(results, summary, plot_path)

        print()
        for condition, bucket in summary["by_condition"].items():
            print(f"  {condition:<8} {bucket['runs']:>2} runs  "
                  f"{bucket['dropped_frames']:>4}/{bucket['expected_frames']:<5} dropped  "
                  f"({bucket['drop_pct']:.2f}%)")
        print()
        print(f"CP-01: {summary['headline']} ({summary['total_drop_pct']:.2f}%)")
        print(f"       measured at the {summary['measured_at']}")
        print(f"       camera frames with no detection (counter gaps): "
              f"{summary['camera_counter_gaps']}")
        print(f"       additionally lost after the camera, inside the plugin: "
              f"{summary['lost_after_camera']}")
        print(f"       across {summary['runs']} runs / {summary['total_measured_burst_sec']} s of measured dwell")
        print(f"  -> {json_path}\n  -> {csv_path}\n  -> {plot_path}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
