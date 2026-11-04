"""Message communication rates, pooled across sessions.

Checks that each message CARMA Platform sends or receives holds the rate it is
expected to.

**Two groups of sessions, because no single session carries every message.**
MAP and SPAT were not broadcast during the verification runs, and the session
recorded to exercise them carries no SDSM. So each topic is measured on a
session that actually contains it, and the results are combined. Which topic
belongs to which group is in ``config.Pl01Config``, not here.

**Two topics do not run continuously**, and are averaged over the periods when
they were present rather than over the whole engaged window. SDSM only flows
while an object is detected; averaged over the window it reports about 1.8 Hz
for a stream running at its nominal 10 Hz whenever it runs at all. MOM stops
when the vehicle drives out of range of the source, which is expected. MAP,
SPAT and BSM are continuous. The rule per topic is the ``gating`` field of its
``config.TopicRate``.

A topic absent from every recording in its group is reported as **not
applicable**, not as a failure. That is a different statement from "the rate
was wrong".
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from . import config as cfg
from . import metrics, report
from .readers import kafka_log

def prepare(session):
    """Detection records, needed only to window the event-driven SDSM rate."""
    if session.kafka_detected_object is None:
        return []
    print(f"  parsing {session.kafka_detected_object.name} ...", flush=True)
    return kafka_log.parse_kafka_log_records(session.kafka_detected_object)


def measure_topics(topics: Dict[str, cfg.TopicRate], with_obu: bool,
                   config: cfg.MessageRateConfig = cfg.MESSAGE_RATES):
    """Build a measurement function for one group of topics.

    Returned as a closure because ``report.analyse_runs`` calls one function per
    run; the group and the configuration are fixed for the whole pass.
    """

    def measure(run, window, detection_records):
        active = metrics.detection_intervals(
            detection_records or [], window, config.detection_gap_sec)
        row = {"topics": ",".join(rate.label for rate in topics.values())}
        outcomes = []

        for topic, rate in topics.items():
            if rate.gating == cfg.GATING_DETECTIONS:
                intervals = active
            elif rate.gating == cfg.GATING_SELF:
                intervals = metrics.topic_active_intervals(
                    run.mcap, topic, window, config.out_of_range_gap_sec)
            else:
                intervals = None
            passed, stats = metrics.message_rate(
                run.mcap, topic, rate.expected_rate_hz, window, None,
                config.rate_tolerance_pct, rate.label.lower(), intervals,
            )
            key = rate.label.lower()
            row[f"{key}_active_periods"] = len(intervals) if intervals else 0
            measured = stats.get("average_rate_hz")
            row[f"{key}_rate_hz"] = round(measured, 3) if measured is not None else None
            row[f"{key}_messages"] = stats.get("total_messages")
            row[f"{key}_passed"] = passed
            outcomes.append(passed)

        if with_obu:
            try:
                capture_date = run.start_time.date() if run.start_time else None
                obu = metrics.obu_radio_activity(
                    run.obu_capture, capture_date, run.mcap, window,
                    rsu_pcap=run.rsu_pcap)
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
            f"{rate.label}={row[f'{rate.label.lower()}_rate_hz']}Hz"
            for rate in topics.values()
            if row.get(f"{rate.label.lower()}_rate_hz") is not None
        )
        return row

    return measure


def summarise_topics(rows: List[Dict], topics: Dict[str, cfg.TopicRate]) -> Dict:
    """Per-topic rollup over the rows that measured those topics."""
    summary = {}
    for topic, rate in topics.items():
        key = rate.label.lower()
        relevant = [row for row in rows if f"{key}_passed" in row]
        outcomes = [row[f"{key}_passed"] for row in relevant]
        applicable = [outcome for outcome in outcomes if outcome is not None]
        rates = [row[f"{key}_rate_hz"] for row in relevant
                 if row.get(f"{key}_rate_hz") is not None]
        summary[rate.label] = {
            "topic": topic,
            "expected_rate_hz": rate.expected_rate_hz,
            "gating": rate.gating or "continuous",
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


def analyse(data_roots, secondary_roots=None,
            config: cfg.MessageRateConfig = cfg.MESSAGE_RATES,
            metric: str = "message rates",
            layout: cfg.SessionLayout = cfg.LAYOUT) -> Dict:
    """Measure every topic on a session that carries it, then combine.

    ``secondary_roots`` names the sessions the secondary topic group is
    measured on. Leave it out and every topic is measured on ``data_roots``,
    which is the behaviour when one session carries them all.
    """
    secondary_roots = secondary_roots or data_roots
    split = secondary_roots is not data_roots

    print("=== " + ", ".join(r.label for r in config.primary_topics.values()) + " ===")
    rows = report.analyse_runs(
        data_roots, measure_topics(config.primary_topics, True, config),
        prepare, layout=layout)
    for row in rows:
        row["group"] = "primary"

    if split:
        print("\n=== " + ", ".join(r.label for r in config.secondary_topics.values()) + " ===")
        secondary_rows = report.analyse_runs(
            secondary_roots, measure_topics(config.secondary_topics, False, config),
            prepare, layout=layout)
        for row in secondary_rows:
            row["group"] = "secondary"
        rows += secondary_rows

    per_topic = summarise_topics(rows, config.all_topics)
    evaluated = [row for row in rows if row.get("passed") is not None]
    passed = sum(1 for row in evaluated if row["passed"])

    summary = {
        "metric": metric,
        "rate_tolerance_pct": config.rate_tolerance_pct,
        "primary_sessions": [str(root) for root in data_roots],
        "secondary_sessions": [str(root) for root in secondary_roots],
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
    summary["is_passed"] = (
        summary["topics_evaluated"] > 0
        and summary["topics_passed"] == summary["topics_evaluated"]
    )
    return summary


def write_outputs(summary: Dict, output_dir: Path, case: cfg.TestCase) -> None:
    """The JSON, the per-run CSV and the console table.

    ``case`` supplies the file stem and the label printed with the verdict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summary["runs_detail"]
    (output_dir / f"{case.prefix}.json").write_text(
        json.dumps(summary, indent=2, default=float))

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_dir / f"{case.prefix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"{'topic':<6}{'expected':>10}{'median':>10}{'range':>18}{'runs':>8}  source")
    for label, stats in summary["per_topic"].items():
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
    print(f"{case.metric}: {summary['headline']}, "
          f"{summary['runs_passed']}/{summary['runs_evaluated']} runs passed "
          f"every applicable check ({summary['pass_rate']})")
    if summary["failed_runs"]:
        print(f"       failed runs: {', '.join(summary['failed_runs'])}")
    print(f"  -> {output_dir}")
