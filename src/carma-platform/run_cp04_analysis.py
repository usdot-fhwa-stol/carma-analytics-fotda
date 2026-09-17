#!/usr/bin/env python3
"""CP-04: camera detection to Kafka, pooled across sessions.

Measures the delay between the camera's own detection timestamp and the Kafka
broker accepting the record::

    python run_cp04_analysis.py \
        --data-root .../20260914_verification_test \
        --data-root .../20260915_verification_test \
        --output-dir out/cp04

This covers only the first hop of the chain, so it isolates the V2XHub FLIR
plugin. A camera websocket stall shows up here directly and nowhere earlier,
which makes CP-04 the most sensitive indicator of that fault.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dt_wz_analysis_util import metrics, report
from dt_wz_analysis_util.portable import kafka_log

METRIC = "CP-04"
PREFIX = "cp04_detection_to_kafka"
TITLE = "CP-04: camera detection to Kafka"


def prepare(session):
    if session.kafka_detected_object is None:
        raise FileNotFoundError("CP-04 needs the v2xhub_sim_sensor_detected_object log")
    print(f"  parsing {session.kafka_detected_object.name} ...", flush=True)
    return kafka_log.parse_kafka_log_records(session.kafka_detected_object)


def measure(run, window, detection_records):
    passed, stats = metrics.detection_to_kafka_latency(detection_records, window)
    return {
        "samples": stats.get("total_detections"),
        "median_s": (
            round(stats["latency_s"]["median"], 4)
            if stats.get("latency_s") else None
        ),
        "mean_s": round(stats["mean_latency_s"], 4) if stats.get("mean_latency_s") else None,
        "late_detections": stats.get("late_detections"),
        "late_pct": round(stats["late_pct"], 3) if stats.get("late_pct") is not None else None,
        "passed": passed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = report.analyse_runs(args.data_root, measure, prepare)
    summary = report.summarise_latency(
        rows, METRIC, metrics.DETECTION_TO_KAFKA_MEAN_LATENCY_THRESHOLD_SEC, "s")
    summary["sessions"] = [str(root) for root in args.data_root]
    report.write_outputs(summary, args.output_dir, PREFIX, TITLE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
