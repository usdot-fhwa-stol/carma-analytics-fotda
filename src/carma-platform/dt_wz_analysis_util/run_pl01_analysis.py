#!/usr/bin/env python3
"""PL-01: message communication rates and OBU radio activity, pooled across sessions.

Checks that each message CARMA Platform sends or receives holds its expected
rate::

    python run_pl01_analysis.py \\
        --data-root .../20260917 \\
        --sdsm-data-root .../20260914_verification_test \\
        --sdsm-data-root .../20260915_verification_test \\
        --output-dir out/pl01

**Two groups of sessions, because no single session carries every message.**
MAP and SPAT were not broadcast during the verification runs, and the session
recorded to exercise them carries no SDSM. So each topic is measured on a
session that actually contains it, and the results are combined:

``--data-root``
    MAP, SPAT, MOM and BSM. The session dedicated to message rates.

``--sdsm-data-root``
    SDSM only. The verification sessions, which are the ones with pedestrian
    detections and therefore the only ones where SDSMs flow.

Give only ``--data-root`` and every topic is measured there, which is the
behaviour when one session carries them all.

**Two topics do not run continuously, and are averaged over the periods when
they were present rather than over the whole engaged window.** SDSM only flows
while an object is detected; averaged over the window it reports about 1.8 Hz
for a stream running at its nominal 10 Hz whenever it runs at all. MOM stops
when the vehicle drives out of range of the source, which is expected, so its
own in-range periods are used. MAP, SPAT and BSM are continuous and are averaged
over the whole window.

A topic absent from every recording in its group is reported as **not
applicable**, not as a failure. That is a different statement from "the rate was
wrong".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dt_wz_analysis_util import metrics, report
from dt_wz_analysis_util.readers import kafka_log

PREFIX = "pl01_message_communication"

# topic -> (label, expected Hz, gating)
#
# Gating says what the rate should be averaged over. Two topics here do not run
# continuously, and averaging them over the whole engaged window would measure
# how long they were absent rather than how fast they ran when present:
#
#   "detections"  SDSM only flows while an object is detected, so its periods
#                 come from the raw detection log.
#   "self"        MOM stops when the vehicle drives out of range of the source,
#                 which is expected; its periods come from its own arrivals.
#
# MAP, SPAT and BSM are continuous and are averaged over the whole window.
ALWAYS_ON_TOPICS = {
    "/message/incoming_map": ("MAP", 1.0, None),
    "/message/incoming_spat": ("SPAT", 10.0, None),
    "/message/incoming_mobility_operation": ("MOM", 1.0, "self"),
    "/message/bsm_outbound": ("BSM", 10.0, None),
}
SDSM_TOPICS = {
    "/message/incoming_sdsm": ("SDSM", 10.0, "detections"),
}

# A MOM gap longer than this means the vehicle left the source's range, rather
# than a message being missed. Nominal interval is 1 s.
OUT_OF_RANGE_GAP_SEC = 2.0
ALL_TOPICS = {**ALWAYS_ON_TOPICS, **SDSM_TOPICS}
RATE_TOLERANCE_PCT = 0.2


def prepare(session):
    """Detection records, needed only to window the event-driven SDSM rate."""
    if session.kafka_detected_object is None:
        return []
    print(f"  parsing {session.kafka_detected_object.name} ...", flush=True)
    return kafka_log.parse_kafka_log_records(session.kafka_detected_object)


def measure_topics(topics, with_obu: bool):
    """Build a measurement function for one group of topics."""

    def measure(run, window, detection_records):
        active = metrics.detection_intervals(detection_records or [], window)
        row = {"topics": ",".join(label for label, _hz, _g in topics.values())}
        outcomes = []

        for topic, (label, expected_hz, gating) in topics.items():
            if gating == "detections":
                intervals = active
            elif gating == "self":
                intervals = metrics.topic_active_intervals(
                    run.mcap, topic, window, OUT_OF_RANGE_GAP_SEC)
            else:
                intervals = None
            passed, stats = metrics.message_rate(
                run.mcap, topic, expected_hz, window, None, RATE_TOLERANCE_PCT,
                label.lower(), intervals,
            )
            row[f"{label.lower()}_active_periods"] = len(intervals) if intervals else 0
            rate = stats.get("average_rate_hz")
            row[f"{label.lower()}_rate_hz"] = round(rate, 3) if rate is not None else None
            row[f"{label.lower()}_messages"] = stats.get("total_messages")
            row[f"{label.lower()}_passed"] = passed
            outcomes.append(passed)

        if with_obu:
            try:
                capture_date = run.start_time.date() if run.start_time else None
                obu = metrics.obu_radio_activity(
                    run.obu_capture, capture_date, run.mcap, window, rsu_pcap=run.rsu_pcap)
                row["bsm_sent_by_platform"] = obu["total_bsms_sent_by_platform"]
                row["bsm_on_radio"] = obu["total_bsms_on_radio"]
                row["bsm_shortfall"] = obu["bsm_shortfall"]
                row["sdsm_on_radio"] = obu["total_sdsms_on_radio"]
            except Exception as error:
                row["obu_error"] = str(error)

        # The run passes when every applicable rate check passed. A topic that
        # was never recorded returns None and is excluded rather than counted
        # against the run.
        applicable = [outcome for outcome in outcomes if outcome is not None]
        row["passed"] = all(applicable) if applicable else None
        row["summary"] = "  ".join(
            f"{label}={row[f'{label.lower()}_rate_hz']}Hz"
            for label, _hz, _g in topics.values()
            if row.get(f"{label.lower()}_rate_hz") is not None
        )
        return row

    return measure


def summarise_topics(rows, topics):
    """Per-topic rollup over the rows that measured those topics."""
    summary = {}
    for topic, (label, expected_hz, _gating) in topics.items():
        key = label.lower()
        relevant = [row for row in rows if f"{key}_passed" in row]
        outcomes = [row[f"{key}_passed"] for row in relevant]
        applicable = [outcome for outcome in outcomes if outcome is not None]
        rates = [row[f"{key}_rate_hz"] for row in relevant
                 if row.get(f"{key}_rate_hz") is not None]
        summary[label] = {
            "topic": topic,
            "expected_rate_hz": expected_hz,
            "sessions": sorted({row["session"] for row in relevant}),
            "runs": len(relevant),
            "runs_evaluated": len(applicable),
            "runs_passed": sum(1 for outcome in applicable if outcome),
            "not_applicable": len(outcomes) - len(applicable),
            "median_rate_hz": round(sorted(rates)[len(rates) // 2], 3) if rates else None,
            "min_rate_hz": round(min(rates), 3) if rates else None,
            "max_rate_hz": round(max(rates), 3) if rates else None,
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session(s) carrying MAP, SPAT, MOM and BSM")
    parser.add_argument("--sdsm-data-root", type=Path, action="append",
                        help="Session(s) carrying SDSM; defaults to --data-root")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    sdsm_roots = args.sdsm_data_root or args.data_root
    split = args.sdsm_data_root is not None

    print("=== MAP, SPAT, MOM, BSM ===")
    rows = report.analyse_runs(
        args.data_root, measure_topics(ALWAYS_ON_TOPICS, with_obu=True), prepare)
    for row in rows:
        row["measured"] = "always-on"

    if split:
        print("\n=== SDSM ===")
        sdsm_rows = report.analyse_runs(
            sdsm_roots, measure_topics(SDSM_TOPICS, with_obu=False), prepare)
        for row in sdsm_rows:
            row["measured"] = "sdsm"
        rows += sdsm_rows

    per_topic = summarise_topics(rows, ALL_TOPICS)
    evaluated = [row for row in rows if row.get("passed") is not None]
    passed = sum(1 for row in evaluated if row["passed"])

    summary = {
        "metric": "PL-01",
        "rate_tolerance_pct": RATE_TOLERANCE_PCT,
        "rate_sessions": [str(root) for root in args.data_root],
        "sdsm_sessions": [str(root) for root in sdsm_roots],
        "runs": len(rows),
        "runs_evaluated": len(evaluated),
        "runs_passed": passed,
        "pass_rate": f"{passed / len(evaluated) * 100:.1f}%" if evaluated else "n/a",
        "topics_passed": sum(
            1 for stats in per_topic.values()
            if stats["runs_evaluated"] and stats["runs_passed"] == stats["runs_evaluated"]
        ),
        "topics_evaluated": sum(1 for stats in per_topic.values() if stats["runs_evaluated"]),
        "failed_runs": [row["run"] for row in evaluated if not row["passed"]],
        "per_topic": per_topic,
        "runs_detail": rows,
    }
    summary["headline"] = (
        f"{summary['topics_passed']}/{summary['topics_evaluated']} topics held their rate"
    )

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
    print(f"{'topic':<6}{'expected':>10}{'median':>10}{'range':>18}{'runs':>8}  source")
    for label, stats in per_topic.items():
        if not stats["runs_evaluated"]:
            print(f"{label:<6}{stats['expected_rate_hz']:>9g}Hz"
                  f"{'not applicable — topic absent from all recordings':>46}")
            continue
        span = f"{stats['min_rate_hz']:.2f}–{stats['max_rate_hz']:.2f}"
        source = ", ".join(Path(name).name for name in stats["sessions"])
        print(f"{label:<6}{stats['expected_rate_hz']:>9g}Hz"
              f"{stats['median_rate_hz']:>9.3f}Hz{span:>18}"
              f"{stats['runs_passed']:>4}/{stats['runs_evaluated']:<3}  {source}")
    print()
    print(f"PL-01: {summary['headline']}, {passed}/{len(evaluated)} runs passed "
          f"every applicable check ({summary['pass_rate']})")
    if summary["failed_runs"]:
        print(f"       failed runs: {', '.join(summary['failed_runs'])}")
    print(f"  -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
