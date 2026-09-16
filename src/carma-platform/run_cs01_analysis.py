#!/usr/bin/env python3
"""CS-01: SDSM location spoofing verification, per recording session.

Checks that each SDSM places the spoofed pedestrian at the reference location
FLIRCameraDriver was configured with, and that the reported heading matches the
detection velocity::

    python run_cs01_analysis.py \\
        --data-root .../20260915_verification_test \\
        --output-dir out/cs01

Give ``--data-root`` more than once to check several sessions in one command.
Each session is verified on its own, because the configured reference can change
between sessions; the reference is read from the data, not supplied.

Writes ``cs01_location_spoofing.json``, ``cs01_location_spoofing.csv`` (one row
per verified object) and the verification plot, in a sub-directory per session.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cs01_location_spoofing as cs01
import dt_wz_dataset as dataset


def analyse(root: Path, output_dir: Path, args) -> dict:
    runs = dataset.load_runs_csv(root / "runs.csv", root)
    session = dataset.discover_session(root)
    session_dir = output_dir / root.name
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {root.name}: {len(runs)} runs ===")
    result = cs01.analyse_session(
        runs, session, plots_dir=session_dir,
        max_mean_position_error_m=args.max_position_error_m,
        max_mean_heading_error_deg=args.max_heading_error_deg,
    )

    rows = result.pop("rows")
    (session_dir / "cs01_location_spoofing.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    if rows:
        with open(session_dir / "cs01_location_spoofing.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"  reference      : {result['reference_lat']:.7f}, {result['reference_lon']:.7f}")
    for origin in result["reference_detail"]["origins_in_window"]:
        print(f"      origin seen in window: {origin['lat']:.7f}, {origin['lon']:.7f}"
              f"  ({origin['detections']} detections)")
    print(f"  driver rotation: {result['driver_rotation_deg']:.2f} deg clockwise")
    print(f"  verified       : {result.get('verified_objects', 0)} objects "
          f"({result.get('unmatched_objects', 0)} unmatched)")
    print(f"  position error : mean {result.get('mean_position_error_m', float('nan')):.4f} m  "
          f"p95 {result.get('p95_position_error_m', float('nan')):.4f} m  "
          f"max {result.get('max_position_error_m', float('nan')):.4f} m   "
          f"-> {'PASS' if result.get('position_pass') else 'FAIL'} "
          f"(< {result['max_mean_position_error_m']} m)")
    print(f"  heading error  : mean {result.get('mean_heading_error_deg', float('nan')):.4f} deg "
          f"over {result.get('heading_objects', 0)} moving objects   "
          f"-> {'PASS' if result.get('heading_pass') else 'FAIL'} "
          f"(< {result['max_mean_heading_error_deg']} deg)")
    print(f"  CS-01          : {'PASS' if result['pass'] else 'FAIL'}")
    print(f"  -> {session_dir}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session directory with runs.csv; repeat for several sessions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-position-error-m", type=float,
                        default=cs01.DEFAULT_MAX_MEAN_POSITION_ERROR_M)
    parser.add_argument("--max-heading-error-deg", type=float,
                        default=cs01.DEFAULT_MAX_MEAN_HEADING_ERROR_DEG)
    args = parser.parse_args(argv)

    results = {}
    for root in args.data_root:
        results[root.name] = analyse(root, args.output_dir, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        name: {key: value for key, value in result.items() if key != "reference_detail"}
        for name, result in results.items()
    }
    (args.output_dir / "cs01_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print()
    for name, result in results.items():
        print(f"CS-01 {name}: {'PASS' if result['pass'] else 'FAIL'}  "
              f"({result.get('verified_objects', 0)} objects, "
              f"mean position error {result.get('mean_position_error_m', float('nan')):.4f} m)")
    return 0 if all(result["pass"] for result in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
