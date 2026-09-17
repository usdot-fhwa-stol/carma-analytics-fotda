#!/usr/bin/env python3
"""PL-01: message communication rates and OBU radio activity, pooled across sessions.

Checks that each message CARMA Platform sends or receives holds its expected
rate, and counts what the OBU radio actually carried::

    python run_pl01_analysis.py \\
        --data-root .../20260914_verification_test \\
        --data-root .../20260915_verification_test \\
        --output-dir out/pl01

A topic absent from every recording is reported as **not applicable**, not as a
failure. That distinction matters: MAP and SPAT were never recorded in these
sessions, so there is nothing to regress, which is a different statement from
"the rate was wrong".

SDSM is event-driven -- it only flows while an object is detected -- so its rate
is averaged over the detection periods rather than the whole engaged window.
Averaging it over the window reports about 1.8 Hz for a stream running at its
nominal 10 Hz whenever it runs at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dt_wz_analysis_util import dataset, metrics, report
from dt_wz_analysis_util.portable import kafka_log

PREFIX = "pl01_message_communication"

# topic -> (label, expected Hz, event-driven)
EXPECTED_RATES_HZ = {
    "/message/incoming_map": ("MAP", 1.0, False),
    "/message/incoming_spat": ("SPAT", 10.0, False),
    "/message/incoming_sdsm": ("SDSM", 10.0, True),
    "/message/incoming_mobility_operation": ("MOM", 1.0, False),
    "/message/bsm_outbound": ("BSM", 10.0, False),
}
RATE_TOLERANCE_PCT = 0.2


def prepare(session):
    if session.kafka_detected_object is None:
        return []
    print(f"  parsing {session.kafka_detected_object.name} ...", flush=True)
    return kafka_log.parse_kafka_log_records(session.kafka_detected_object)


def measure(run, window, detection_records):
    active = metrics.detection_intervals(detection_records or [], window)
    row = {"detection_active_s": round(sum(end - start for start, end in active), 3)}
    outcomes = []

    for topic, (label, expected_hz, event_driven) in EXPECTED_RATES_HZ.items():
        passed, stats = metrics.message_rate(
            run.mcap, topic, expected_hz, window, None, RATE_TOLERANCE_PCT,
            label.lower(), active if event_driven else None,
        )
        rate = stats.get("average_rate_hz")
        row[f"{label.lower()}_rate_hz"] = round(rate, 3) if rate is not None else None
        row[f"{label.lower()}_passed"] = passed
        outcomes.append(passed)

    try:
        obu = metrics.obu_radio_activity(
            run.obu_capture, run.start_time.date(), run.mcap, window, rsu_pcap=run.rsu_pcap)
        row["bsm_sent_by_platform"] = obu["total_bsms_sent_by_platform"]
        row["bsm_on_radio"] = obu["total_bsms_on_radio"]
        row["bsm_shortfall"] = obu["bsm_shortfall"]
        row["sdsm_on_radio"] = obu["total_sdsms_on_radio"]
    except Exception as error:
        row["obu_error"] = str(error)

    # The run passes when every applicable rate check passed. Topics that were
    # never recorded return None and are excluded rather than counted against it.
    applicable = [outcome for outcome in outcomes if outcome is not None]
    row["passed"] = all(applicable) if applicable else None
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = report.analyse_runs(args.data_root, measure, prepare)

    per_topic = {}
    for topic, (label, expected_hz, _event) in EXPECTED_RATES_HZ.items():
        key = label.lower()
        outcomes = [row.get(f"{key}_passed") for row in rows]
        applicable = [outcome for outcome in outcomes if outcome is not None]
        rates = [row[f"{key}_rate_hz"] for row in rows if row.get(f"{key}_rate_hz") is not None]
        per_topic[label] = {
            "topic": topic,
            "expected_rate_hz": expected_hz,
            "runs_evaluated": len(applicable),
            "runs_passed": sum(1 for outcome in applicable if outcome),
            "not_applicable": len(outcomes) - len(applicable),
            "median_rate_hz": round(sorted(rates)[len(rates) // 2], 3) if rates else None,
        }

    evaluated = [row for row in rows if row.get("passed") is not None]
    passed = sum(1 for row in evaluated if row["passed"])
    summary = {
        "metric": "PL-01",
        "rate_tolerance_pct": RATE_TOLERANCE_PCT,
        "runs": len(rows),
        "runs_evaluated": len(evaluated),
        "runs_passed": passed,
        "pass_rate": f"{passed / len(evaluated) * 100:.1f}%" if evaluated else "n/a",
        "failed_runs": [row["run"] for row in evaluated if not row["passed"]],
        "per_topic": per_topic,
        "sessions": [str(root) for root in args.data_root],
        "runs_detail": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{PREFIX}.json").write_text(json.dumps(summary, indent=2, default=float))
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(args.output_dir / f"{PREFIX}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    for label, stats in per_topic.items():
        if stats["runs_evaluated"] == 0:
            print(f"  {label:<6} not applicable — topic absent from all {len(rows)} recordings")
        else:
            print(f"  {label:<6} {stats['runs_passed']}/{stats['runs_evaluated']} runs passed, "
                  f"median {stats['median_rate_hz']} Hz (expected {stats['expected_rate_hz']:g} Hz)")
    print()
    print(f"PL-01: {passed}/{len(evaluated)} runs passed every applicable rate check "
          f"({summary['pass_rate']})")
    if summary["failed_runs"]:
        print(f"       failed: {', '.join(summary['failed_runs'])}")
    print(f"  -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
