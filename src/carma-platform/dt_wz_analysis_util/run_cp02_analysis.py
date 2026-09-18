#!/usr/bin/env python3
"""CP-02: raw detection to SDSM received at the vehicle, pooled across sessions.

Counts camera detections that never reached CARMA Platform inside an SDSM::

    python run_cp02_analysis.py \\
        --data-root .../20260914_verification_test \\
        --data-root .../20260915_verification_test \\
        --output-dir out/cp02

This spans the whole chain -- Kafka, the sensor_data_sharing_service, V2XHub,
TENA, the RSU and the radio -- so a CP-02 failure alone does not say where the
loss happened. Compare it against CP-03: where CP-02 fails and CP-03 passes, the
loss is upstream of the RSU.

Writes ``cp02_detection_to_sdsm.json``, a per-run CSV and a per-run plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dt_wz_analysis_util import report
from dt_wz_analysis_util import metrics
from dt_wz_analysis_util.readers import kafka_log

METRIC = "CP-02"
PREFIX = "cp02_detection_to_sdsm"
TITLE = "CP-02: detection to SDSM received at the vehicle"


def prepare(session):
    """Parse the session-wide detection log once, not once per run."""
    if session.kafka_detected_object is None:
        raise FileNotFoundError("CP-02 needs the v2xhub_sim_sensor_detected_object log")
    print(f"  parsing {session.kafka_detected_object.name} ...", flush=True)
    return kafka_log.parse_kafka_log_records(session.kafka_detected_object)


def measure(run, window, detection_records):
    passed, stats = metrics.detection_to_sdsm_drop_rate(
        run.mcap, detection_records, window
    )
    return {
        "checked": stats.get("total_raw_detections"),
        "matched": stats.get("total_matched"),
        "dropped": stats.get("total_dropped"),
        "drop_rate_pct": (
            round(stats["drop_rate_pct"], 3) if stats.get("drop_rate_pct") is not None else None
        ),
        "sdsm_objects_at_vehicle": stats.get("sdsm_object_detections"),
        "passed": passed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session directory with runs.csv; repeat to pool sessions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-drop-rate-pct", type=float,
                        default=metrics.SDSM_DROP_RATE_THRESHOLD_PCT)
    args = parser.parse_args(argv)

    rows = report.analyse_runs(args.data_root, measure, prepare)
    summary = report.summarise(rows, METRIC, args.max_drop_rate_pct)
    summary["sessions"] = [str(root) for root in args.data_root]
    report.write_outputs(summary, args.output_dir, PREFIX, TITLE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
