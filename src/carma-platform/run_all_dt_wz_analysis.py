#!/usr/bin/env python3
"""Run every DT-WZ (pedestrian detection to SDSM) metric over a verification session.

Point it at the session directory and it does the rest::

    python run_all_dt_wz_analysis.py \
        --data-root /path/to/20260914_verification_test \
        --output-dir out/verification_20260914

``runs.csv`` in the session directory is the authority on which files constitute
a run; see ``dt_wz_dataset``. Everything is written under ``--output-dir``:
per-run stats JSON and a detection-level CSV, plus session-level summary tables
and plots grouped by pedestrian dwell condition.

Metrics produced per run
------------------------
======  ======================================================================
CP-02   raw detection -> SDSM drop rate               (threshold 2%)
CP-03   RSU broadcast -> CARMA Platform receipt       (threshold 2%)
CP-04   detection -> Kafka latency                    (mean < 0.5 s)
DT-05   detection -> SDSM receipt at the vehicle      (median < 0.3 s)
PL-01   per-topic message rates, and OBU radio counts
======  ======================================================================

Plus the end-to-end cascade: one row per camera detection with a timestamp for
each of the ~19 stages between the FLIR camera and the vehicle's fused output.

A metric reports one of four outcomes, and they are not interchangeable: passed,
failed, **not applicable** (the data needed was never recorded -- MAP and SPAT
are in this position for the 2026-09-14 session), or errored.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import dt_wz_dataset as dataset
import dt_wz_metrics as metrics
from dt_wz_cascade import build as cascade_build
from dt_wz_cascade import cascade_config
from dt_wz_cascade import plots as cascade_plots
from dt_wz_cascade import qa as cascade_qa
from guidance_scripts import get_engage_time
from portable import kafka_log, obu_capture

# Topics checked for rate regression, with the rate each is expected to hold.
# SDSM is event-driven -- it only flows while an object is detected -- so its
# rate is averaged over the detection periods rather than the whole window.
EXPECTED_RATES_HZ = {
    "/message/incoming_map": ("MAP", 1.0, False),
    "/message/incoming_spat": ("SPAT", 10.0, False),
    "/message/incoming_sdsm": ("SDSM", 10.0, True),
    "/message/incoming_mobility_operation": ("MOM", 1.0, False),
    "/message/bsm_outbound": ("BSM", 10.0, False),
}
RATE_TOLERANCE_PCT = 0.2


def analyse_run(run, session_logs, detection_records, output_dir, verbose=True):
    """Run every metric for one run. Returns (results, summary_row, cascade_table)."""
    run_dir = output_dir / run.name
    stats_dir = run_dir / "stats"
    for directory in (run_dir, stats_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Optional[bool]] = {}
    row: Dict[str, object] = {
        "run": run.name,
        "condition": run.condition,
        "run_id": run.run_id,
        "dwell_sec": run.dwell_sec,
        "start_time": run.start_time.isoformat(),
    }

    try:
        engage_time, disengage_time = get_engage_time(run.mcap)
        window = metrics.engaged_window_epoch(run.mcap, engage_time, disengage_time)
    except Exception as error:
        print(f"  ERROR: cannot determine engaged window: {error}")
        row["error"] = f"engaged window: {error}"
        return {"engaged_window": None}, row, pd.DataFrame()

    row["engaged_duration_s"] = round(window[1] - window[0], 3)
    active = metrics.detection_intervals(detection_records, window)
    row["detection_active_s"] = round(sum(end - start for start, end in active), 3)

    def record(name, function, *args, **kwargs):
        try:
            passed, stats = function(*args, **kwargs)
            results[name] = passed
            return stats
        except Exception as error:
            print(f"  ERROR in {name}: {error}")
            if verbose:
                traceback.print_exc(limit=2)
            results[name] = None
            row[f"{name}_error"] = str(error)
            return {}

    cp02 = record("CP02_sdsm_detection_drop_rate", metrics.detection_to_sdsm_drop_rate,
                  run.mcap, detection_records, window, stats_dir)
    row["cp02_raw_detections"] = cp02.get("total_raw_detections")
    row["cp02_dropped"] = cp02.get("total_dropped")
    row["cp02_drop_rate_pct"] = cp02.get("drop_rate_pct")

    cp03 = record("CP03_rsu_sdsm_transmission_drop_rate", metrics.rsu_transmission_drop_rate,
                  run.mcap, run.rsu_pcap, window, stats_dir)
    row["cp03_broadcasts"] = cp03.get("total_broadcasts_checked")
    row["cp03_dropped"] = cp03.get("total_dropped")
    row["cp03_drop_rate_pct"] = cp03.get("drop_rate_pct")
    latency = cp03.get("receive_latency_ms") or {}
    row["cp03_receive_latency_median_ms"] = latency.get("median")

    cp04 = record("CP04_detection_to_kafka_latency", metrics.detection_to_kafka_latency,
                  detection_records, window, stats_dir)
    row["cp04_mean_latency_s"] = cp04.get("mean_latency_s")
    row["cp04_late_pct"] = cp04.get("late_pct")

    dt05 = record("DT05_detection_to_sdsm_receipt_latency",
                  metrics.detection_to_sdsm_receipt_latency, run.mcap, window, stats_dir)
    row["dt05_samples"] = dt05.get("sample_count")
    row["dt05_median_latency_s"] = dt05.get("median_latency_s")

    for topic, (label, expected_hz, event_driven) in EXPECTED_RATES_HZ.items():
        stats = record(
            f"PL01_{label}_rate", metrics.message_rate, run.mcap, topic, expected_hz, window,
            stats_dir, RATE_TOLERANCE_PCT, label.lower(),
            active if event_driven else None,
        )
        row[f"pl01_{label.lower()}_rate_hz"] = stats.get("average_rate_hz")

    try:
        obu = metrics.obu_radio_activity(
            run.obu_capture, run.start_time.date(), run.mcap, window, stats_dir,
            rsu_pcap=run.rsu_pcap,
        )
        row["obu_bsm_on_radio"] = obu["total_bsms_on_radio"]
        row["obu_bsm_shortfall"] = obu["bsm_shortfall"]
        row["obu_sdsm_on_radio"] = obu["total_sdsms_on_radio"]
        if obu.get("ota_reception_rate_pct") is not None:
            row["obu_ota_reception_pct"] = obu["ota_reception_rate_pct"]
            row["obu_ota_missed"] = obu["ota_missed_by_radio"]
        results["PL01_obu_radio_activity"] = True
    except Exception as error:
        print(f"  ERROR in PL01_obu_radio_activity: {error}")
        results["PL01_obu_radio_activity"] = None
        obu = {}

    cascade = _build_cascade(run, session_logs, window, run_dir, row, verbose)
    return results, row, cascade


def _build_cascade(run, session_logs, window, run_dir, row, verbose):
    """Build and persist this run's per-detection stage table."""
    try:
        # The OBU capture is a binary pcap in some sessions and tcpdump text in
        # others; the reader dispatches on the file and only the binary form
        # carries payloads to match on.
        capture = obu_capture.read_obu_capture(run.obu_capture, run.start_time.date())
        radio = [
            {"timestamp": message["timestamp"] * 1e3, "payload_hex": message["payload_hex"]}
            for message in obu_capture.messages_of_type(capture, "SDSM", window[0], window[1])
        ]
        row["obu_payload_matched"] = capture["payloads_available"]

        table = cascade_build.build_run_table(run, session_logs, window)
        if table.empty:
            row["cascade_rows"] = 0
            return table
        table = cascade_build.attach_run_sources(table, run, window, radio)
        table = cascade_build.add_quality(table)
        table = cascade_build.add_deltas(table)
        table.insert(0, "condition", run.condition)
        table.insert(1, "run", run.name)
        table.to_csv(run_dir / "detections.csv", index=False, float_format="%.17g")
        (run_dir / "qa_report.txt").write_text(
            cascade_qa.report(table, f"Cascade QA — {run.name}")
        )

        row["cascade_rows"] = len(table)
        row["cascade_reached_vehicle"] = int(table["reached_vehicle"].sum())
        for stage in ("t_rsu_broadcast", "t_ros_inbound", "t_ros_fused"):
            column = f"lat_{stage[2:]}_ms"
            if column in table.columns and table[column].notna().any():
                row[f"cascade_{stage[2:]}_median_ms"] = float(table[column].median())
        return table
    except Exception as error:
        print(f"  ERROR building cascade: {error}")
        if verbose:
            traceback.print_exc(limit=2)
        row["cascade_error"] = str(error)
        return pd.DataFrame()


def write_session_outputs(runs, rows, cascades, results_by_run, output_dir):
    """Write the summary tables and the condition-grouped plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary_by_run.csv", index=False)
    print(f"\nPer-run summary  -> {output_dir / 'summary_by_run.csv'}")

    numeric = summary.select_dtypes("number").columns
    by_condition = (
        summary.groupby("condition", sort=False)[list(numeric)]
        .agg(["mean", "median"])
        .round(4)
    )
    by_condition.columns = [f"{column}_{stat}" for column, stat in by_condition.columns]
    by_condition = by_condition.reset_index()
    by_condition.insert(1, "runs", summary.groupby("condition", sort=False).size().values)
    by_condition.to_csv(output_dir / "summary_by_condition.csv", index=False)
    print(f"Condition summary -> {output_dir / 'summary_by_condition.csv'}")

    cascade_plots.plot_drop_rates(
        summary, "DT-WZ drop rates by run", plots_dir / "drop_rates_by_run.png"
    )

    grouped: Dict[str, pd.DataFrame] = {}
    for condition in dataset.condition_order(runs):
        frames = [table for name, table in cascades.items()
                  if not table.empty and name.startswith(f"{condition}_")]
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        grouped[condition] = combined
        cascade_plots.plot_latency_by_stage(
            combined, f"SDSM latency by stage — {condition} dwell",
            plots_dir / f"latency_by_stage_{condition}.png",
        )
        cascade_plots.plot_latency_by_stage(
            combined, f"Per-hop SDSM latency — {condition} dwell",
            plots_dir / f"latency_per_hop_{condition}.png", mode="delta",
        )
        cascade_plots.plot_cascade(
            combined, f"SDSM latency cascade — {condition} dwell",
            plots_dir / f"latency_cascade_{condition}.png",
        )

    if grouped:
        cascade_plots.plot_condition_comparison(
            grouped, "End-to-end SDSM latency by dwell condition",
            plots_dir / "latency_by_condition.png",
        )
        everything = pd.concat(grouped.values(), ignore_index=True)
        everything.to_csv(output_dir / "detections_all_runs.csv", index=False,
                          float_format="%.17g")
        (output_dir / "qa_report.txt").write_text(
            cascade_qa.report(everything, "Cascade QA — all runs")
        )
        cascade_qa.stage_summary(everything).to_csv(
            output_dir / "stage_summary.csv", index=False
        )
        print(f"QA report         -> {output_dir / 'qa_report.txt'}")
        cascade_plots.plot_latency_by_stage(
            everything, "SDSM latency by stage — all runs",
            plots_dir / "latency_by_stage_all.png",
        )
        print(f"Detection table   -> {output_dir / 'detections_all_runs.csv'} "
              f"({len(everything)} detections)")
    print(f"Plots             -> {plots_dir}")

    _write_summary_json(runs, results_by_run, summary, output_dir)
    _write_column_docs(output_dir)


def _write_summary_json(runs, results_by_run, summary, output_dir):
    """analysis_summary.json, with N/A counted separately from failures."""
    metrics_summary: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "not_applicable": 0, "errors": 0}
    )
    for results in results_by_run.values():
        for metric, outcome in results.items():
            bucket = metrics_summary[metric]
            if outcome is True:
                bucket["passed"] += 1
            elif outcome is False:
                bucket["failed"] += 1
            elif outcome is None:
                bucket["not_applicable"] += 1

    for metric, bucket in metrics_summary.items():
        evaluated = bucket["passed"] + bucket["failed"]
        bucket["total_runs"] = len(runs)
        bucket["pass_rate"] = f"{bucket['passed'] / evaluated * 100:.2f}%" if evaluated else "n/a"

    pooled = {}
    for label, checked, dropped in (
        ("CP02_detection_to_sdsm", "cp02_raw_detections", "cp02_dropped"),
        ("CP03_rsu_to_vehicle", "cp03_broadcasts", "cp03_dropped"),
    ):
        if checked in summary.columns and dropped in summary.columns:
            total = pd.to_numeric(summary[checked], errors="coerce").sum()
            lost = pd.to_numeric(summary[dropped], errors="coerce").sum()
            pooled[label] = {
                "checked": int(total),
                "dropped": int(lost),
                "drop_rate": f"{lost / total * 100:.3f}%" if total else "n/a",
            }

    payload = {
        "analysis_time": datetime.now().isoformat(),
        "analysis_type": "dt_wz_verification",
        "total_runs_analyzed": len(runs),
        "conditions": {
            condition: len(group)
            for condition, group in dataset.group_by_condition(runs).items()
        },
        "metrics_summary": dict(metrics_summary),
        "session_totals": pooled,
        "notes": [
            "not_applicable means the data needed was never recorded, not that the "
            "check failed: these MCAPs contain no /message/incoming_map or "
            "/message/incoming_spat, so PL01_MAP_rate and PL01_SPAT_rate cannot be "
            "evaluated for this session.",
            "t_rsu_broadcast is the RSU's transmit instant, taken from the RSU's own "
            "pcap. Earlier sessions captured the OBU instead, so their t_ota_capture "
            "stage sits one propagation hop later and end-to-end totals are not "
            "directly comparable.",
            "t_obu_radio_rx is paired by ordinal position within the run, not by "
            "payload: the OBU capture is tcpdump text and carries no payload bytes. "
            "Trust it in aggregate, not per row.",
        ],
    }
    path = output_dir / "analysis_summary.json"
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"Summary JSON      -> {path}")


def _write_column_docs(output_dir):
    """Document the cascade table's stage columns alongside the data."""
    lines = [
        "# `detections.csv` columns",
        "",
        "One row per FLIR camera detection. Every `t_*` column is an absolute epoch",
        "timestamp in **milliseconds UTC**; `lat_*_ms` is that stage measured from the",
        "camera detection; `d_a__b_ms` is the cost of the single hop from `a` to `b`.",
        "",
        "| Column | Host | Meaning |",
        "| --- | --- | --- |",
    ]
    for column, host, label in cascade_config.STAGES:
        note = ""
        if column in cascade_config.COUNT_PAIRED_STAGES:
            note = " **Paired by ordinal position, not payload.**"
        elif column == "t_rsu_broadcast":
            note = " **RSU transmit instant** (earlier sessions captured the OBU instead)."
        lines.append(f"| `{column}` | {host} | {label}.{note} |")

    lines += [
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        "| `condition` | Pedestrian dwell condition (`5sec`/`10sec`/`15sec`) |",
        "| `run` | Run identifier, `<condition>_run<N>` |",
        "| `object_id` | FLIR track id; recycled across the session, unique within a run |",
        "| `sdsm_uper_hex` | The SDSM's ASN.1-UPER bytes, the join key from encode to vehicle |",
        "| `stages_reached` | How many stages carry a timestamp for this detection |",
        "| `reached_vehicle` | Whether it arrived on the vehicle's inbound topic |",
        "| `last_stage_reached` | The furthest stage reached, i.e. where it was lost |",
        "",
        "## Joins",
        "",
        "Stages up to the SDSM being encoded join on detection identity",
        "`(object_id, t_flir_detect)`, with the SDSM-side identity recovered as",
        "`sdsm_time_stamp - measurement_time`. From `t_streets_encode` to",
        "`t_ros_inbound` the join is the UPER payload bytes, which is an exact",
        "identity. `t_obu_radio_rx` alone is ordinal, and is left empty for the whole",
        "run when the radio and broadcast counts disagree rather than risking a",
        "systematic misalignment.",
    ]
    path = output_dir / "detections_columns.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Column docs       -> {path}")


def run_session(data_root, runs_csv=None, output_dir=None, only_condition=None, limit=None):
    """Analyse every run named in ``runs.csv`` and write the session outputs."""
    data_root = Path(data_root)
    runs_csv = Path(runs_csv) if runs_csv else data_root / "runs.csv"
    runs = dataset.load_runs_csv(runs_csv, data_root)
    if only_condition:
        runs = [run for run in runs if run.condition == only_condition]
        if not runs:
            raise ValueError(f"No runs with condition {only_condition!r}")
    if limit:
        runs = runs[:limit]

    session = dataset.discover_session(data_root)
    output_dir = Path(output_dir) if output_dir else data_root / (
        f"dt_wz_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Session: {data_root}")
    print(session.describe())
    print(f"Runs: {len(runs)} across conditions "
          f"{', '.join(dataset.condition_order(runs))}")
    print(f"Output: {output_dir}\n")

    print("Parsing session-wide logs (once for all runs):")
    session_logs = cascade_build.SessionLogs.load(session)
    detection_records = kafka_log.parse_kafka_log_records(session.kafka_detected_object)
    print(f"  {len(detection_records)} detection records in the Kafka dump\n")

    rows, cascades, results_by_run = [], {}, {}
    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {run.name} ({run.mcap.name})")
        results, row, cascade = analyse_run(run, session_logs, detection_records, output_dir)
        rows.append(row)
        cascades[run.name] = cascade
        results_by_run[run.name] = results
        print(f"      detections={row.get('cp02_raw_detections')} "
              f"cp02_drop={row.get('cp02_drop_rate_pct')} "
              f"cp03_drop={row.get('cp03_drop_rate_pct')} "
              f"dt05_median={row.get('dt05_median_latency_s')} "
              f"cascade_rows={row.get('cascade_rows')}")

    write_session_outputs(runs, rows, cascades, results_by_run, output_dir)
    print("\nAnalysis complete.")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Session directory holding runs.csv and the log folders")
    parser.add_argument("--runs-csv", type=Path, default=None,
                        help="Run manifest (default: <data-root>/runs.csv)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write results (default: a timestamped dir under --data-root)")
    parser.add_argument("--condition", default=None,
                        help="Only analyse this dwell condition, e.g. 5sec")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only analyse the first N runs (for a quick check)")
    args = parser.parse_args(argv)

    try:
        run_session(args.data_root, args.runs_csv, args.output_dir,
                    args.condition, args.limit)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
