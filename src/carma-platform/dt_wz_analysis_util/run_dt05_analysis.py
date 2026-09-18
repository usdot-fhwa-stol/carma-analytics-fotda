#!/usr/bin/env python3
"""DT-05: camera detection to SDSM received at the vehicle, pooled across sessions.

The end-to-end latency the vehicle actually experiences::

    python run_dt05_analysis.py \
        --data-root .../20260914_verification_test \
        --data-root .../20260915_verification_test \
        --output-dir out/dt05

Where CP-02 asks whether a detection arrived, DT-05 asks how long it took. The
two are independent: a run can deliver every detection and still be slow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dt_wz_analysis_util import metrics, report

METRIC = "DT-05"
PREFIX = "dt05_detection_to_sdsm_receipt"
TITLE = "DT-05: detection to SDSM received at the vehicle"


def measure(run, window, _context):
    passed, stats = metrics.detection_to_sdsm_receipt_latency(run.mcap, window)
    latency = stats.get("latency_s") or {}
    return {
        "samples": stats.get("sample_count"),
        "median_s": (
            round(stats["median_latency_s"], 4)
            if stats.get("median_latency_s") is not None else None
        ),
        "p95_s": round(float(latency["maximum"]), 4) if latency.get("maximum") else None,
        "passed": passed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = report.analyse_runs(args.data_root, measure)
    summary = report.summarise_latency(
        rows, METRIC, metrics.DETECTION_TO_SDSM_RECEIPT_LATENCY_THRESHOLD_SEC, "s")
    summary["sessions"] = [str(root) for root in args.data_root]
    report.write_outputs(summary, args.output_dir, PREFIX, TITLE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
