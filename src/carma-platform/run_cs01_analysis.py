#!/usr/bin/env python3
"""CS-01: SDSM location spoofing verification, pooled across recording sessions.

Checks that each SDSM places the spoofed pedestrian at the reference location
FLIRCameraDriver was configured with, and that the reported heading matches the
detection velocity::

    python run_cs01_analysis.py \\
        --data-root .../20260914_verification_test \\
        --data-root .../20260915_verification_test \\
        --output-dir out/cs01

Every session given is windowed to its own runs, then all of them are verified
together, so the result is one plot and one set of statistics covering every run.
Pooling requires the sessions to share a configured reference; the reference is
read from the data and a mismatch stops the run rather than averaging two
different geometries.

Writes ``cs01_location_spoofing.json``, ``cs01_location_spoofing.csv`` (one row
per verified object) and the verification plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dt_wz_analysis_util import cs01
from dt_wz_analysis_util import dataset


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session directory with runs.csv; repeat to pool sessions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-position-error-m", type=float,
                        default=cs01.DEFAULT_MAX_MEAN_POSITION_ERROR_M)
    parser.add_argument("--max-heading-error-deg", type=float,
                        default=cs01.DEFAULT_MAX_MEAN_HEADING_ERROR_DEG)
    args = parser.parse_args(argv)

    specs = []
    for root in args.data_root:
        runs = dataset.load_runs_csv(root / "runs.csv", root)
        specs.append((root.name, runs, dataset.discover_session(root)))
        print(f"{root.name}: {len(runs)} runs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = cs01.analyse_sessions(
        specs, plots_dir=args.output_dir,
        max_mean_position_error_m=args.max_position_error_m,
        max_mean_heading_error_deg=args.max_heading_error_deg,
    )

    rows = result.pop("rows")
    (args.output_dir / "cs01_location_spoofing.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    if rows:
        with open(args.output_dir / "cs01_location_spoofing.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    nan = float("nan")
    print()
    print(f"  sessions pooled : {', '.join(result['sessions'])}")
    print(f"  runs            : {result['total_runs']}")
    for item in result["per_session"]:
        print(f"      {item['session']}: {item['runs']} runs, "
              f"{item['detection_records_in_window']} detections, "
              f"{item['sdsm_records_in_window']} SDSMs")
    print(f"  reference       : {result['reference_lat']:.7f}, {result['reference_lon']:.7f}")
    print(f"  driver rotation : {result['driver_rotation_deg']:.2f} deg clockwise")
    print(f"  verified        : {result.get('verified_objects', 0)} objects "
          f"({result.get('unmatched_objects', 0)} unmatched)")
    print(f"  position error  : mean {result.get('mean_position_error_m', nan):.4f} m  "
          f"p95 {result.get('p95_position_error_m', nan):.4f} m  "
          f"max {result.get('max_position_error_m', nan):.4f} m   "
          f"-> {'PASS' if result.get('position_pass') else 'FAIL'} "
          f"(< {result['max_mean_position_error_m']} m)")
    print(f"  heading error   : mean {result.get('mean_heading_error_deg', nan):.4f} deg "
          f"over {result.get('heading_objects', 0)} moving objects   "
          f"-> {'PASS' if result.get('heading_pass') else 'FAIL'} "
          f"(< {result['max_mean_heading_error_deg']} deg)")
    print()
    print(f"CS-01: {'PASS' if result['pass'] else 'FAIL'}")
    print(f"  -> {args.output_dir}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
