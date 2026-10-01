"""Build the end-to-end SDSM latency cascade and write its outputs.

One row per FLIR camera detection, carrying a timestamp for each of the ~19
stages between the camera and the vehicle's fused output.

This is not a pass/fail metric. It is the latency breakdown that says *where*
the time goes and where detections are lost, which is what explains the
detection-delivery result rather than merely reporting it.

Session-wide logs (pc1/pc2 V2XHub, SDSS, Kafka) are parsed once per session --
pc2 alone is over a million lines -- and then windowed per run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .. import config as cfg
from .. import dataset, report
from ..readers import obu_capture
from . import build as cascade_build
from . import cascade_config
from . import plots as cascade_plots
from . import qa as cascade_qa

# Stages whose median latency is worth a line in the console rollup.
_HEADLINE_STAGES = (("ros_inbound", "to the vehicle"), ("ros_fused", "to fused output"))


def build_run(run, session_logs, window) -> Tuple[pd.DataFrame, bool]:
    """The per-detection stage table for one run."""
    capture = obu_capture.read_obu_capture(run.obu_capture, run.start_time.date())
    radio = [
        {"timestamp": message["timestamp"] * 1e3, "payload_hex": message["payload_hex"]}
        for message in obu_capture.messages_of_type(
            capture, "SDSM", window[0], window[1])
    ]
    table = cascade_build.build_run_table(run, session_logs, window)
    if table.empty:
        return table, capture["payloads_available"]
    table = cascade_build.attach_run_sources(table, run, window, radio)
    table = cascade_build.add_quality(table)
    table = cascade_build.add_deltas(table)
    table.insert(0, "condition", run.condition)
    table.insert(1, "run", run.name)
    return table, capture["payloads_available"]


def collect(data_roots, layout: cfg.SessionLayout = cfg.LAYOUT):
    """Every run's cascade table, plus a per-run summary row."""
    tables, rows = [], []
    for root in data_roots:
        runs, session = dataset.load_session(root, layout)
        print(f"{Path(root).name}: {len(runs)} runs")
        print("  parsing session-wide logs (once for all runs) ...", flush=True)
        session_logs = cascade_build.SessionLogs.load(session)

        for run in runs:
            row = {"session": Path(root).name, "run": run.name,
                   "condition": run.condition, "dwell_sec": run.dwell_sec}
            try:
                window = report.engaged_window(run)
                table, payload_matched = build_run(run, session_logs, window)
                row["detections"] = len(table)
                row["obu_payload_matched"] = payload_matched
                if not table.empty:
                    row["reached_vehicle"] = int(table["reached_vehicle"].sum())
                    for stage in ("t_rsu_broadcast", "t_ros_inbound", "t_ros_fused"):
                        column = f"lat_{stage[2:]}_ms"
                        if column in table.columns and table[column].notna().any():
                            row[f"median_to_{stage[2:]}_ms"] = round(
                                float(table[column].median()), 1)
                    tables.append(table)
            except Exception as error:
                print(f"  ERROR on {run.name}: {error}")
                row["error"] = str(error)
            rows.append(row)
            print(f"  {run.name:<12} detections={row.get('detections')} "
                  f"to_vehicle={row.get('median_to_ros_inbound_ms')}ms "
                  f"to_fused={row.get('median_to_ros_fused_ms')}ms")
    return tables, rows


def write_outputs(tables: List[pd.DataFrame], rows: List[Dict],
                  output_dir: Path) -> None:
    """The detection table, the QA report, the column docs and the figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_dir / "summary_by_run.csv", index=False)
    if not tables:
        print("No cascade tables were built; nothing further to write.")
        return

    everything = pd.concat(tables, ignore_index=True)
    everything.to_csv(output_dir / "detections.csv", index=False, float_format="%.17g")
    (output_dir / "qa_report.txt").write_text(
        cascade_qa.report(everything, "Cascade QA — all runs"))
    cascade_qa.stage_summary(everything).to_csv(
        output_dir / "stage_summary.csv", index=False)
    write_column_docs(output_dir)

    grouped = {}
    for condition, group in everything.groupby("condition", sort=False):
        grouped[condition] = group
        cascade_plots.plot_latency_by_stage(
            group, f"SDSM latency by stage — {condition} dwell",
            plots_dir / f"latency_by_stage_{condition}.png")
        cascade_plots.plot_latency_by_stage(
            group, f"Per-hop SDSM latency — {condition} dwell",
            plots_dir / f"latency_per_hop_{condition}.png", mode="delta")
        cascade_plots.plot_cascade(
            group, f"SDSM latency cascade — {condition} dwell",
            plots_dir / f"latency_cascade_{condition}.png")

    cascade_plots.plot_latency_by_stage(
        everything, "SDSM latency by stage — all runs",
        plots_dir / "latency_by_stage_all.png")
    if len(grouped) > 1:
        cascade_plots.plot_condition_comparison(
            grouped, "End-to-end SDSM latency by dwell condition",
            plots_dir / "latency_by_condition.png")

    print()
    print(f"Cascade: {len(everything)} detections across {len(rows)} runs")
    for stage, label in _HEADLINE_STAGES:
        column = f"lat_{stage}_ms"
        if column in everything.columns and everything[column].notna().any():
            print(f"       median {label:<16} {everything[column].median():.0f} ms")
    print(f"  -> {output_dir}")


def write_column_docs(output_dir: Path) -> None:
    """A data dictionary for ``detections.csv``, generated from the stage table.

    Generated rather than written by hand so a stage added to
    ``cascade_config.STAGES`` documents itself.
    """
    lines = [
        "# `detections.csv` columns", "",
        "One row per FLIR camera detection. Every `t_*` column is an absolute epoch",
        "timestamp in **milliseconds UTC**; `lat_*_ms` is that stage measured from the",
        "camera detection; `d_a__b_ms` is the cost of the single hop from `a` to `b`.",
        "", "| Column | Host | Meaning |", "| --- | --- | --- |",
    ]
    for column, host, label in cascade_config.STAGES:
        note = ""
        if column in cascade_config.COUNT_PAIRED_STAGES:
            note = " **Paired by time, not payload, when the OBU capture carries none.**"
        elif column == "t_rsu_broadcast":
            note = " **RSU transmit instant** (earlier sessions captured the OBU instead)."
        lines.append(f"| `{column}` | {host} | {label}.{note} |")
    lines += [
        "", "| Column | Meaning |", "| --- | --- |",
        "| `condition` | Pedestrian dwell condition |",
        "| `run` | Run identifier, `<condition>_run<N>` |",
        "| `object_id` | FLIR track id; unique within a run |",
        "| `sdsm_uper_hex` | The SDSM's ASN.1-UPER bytes, the join key from encode to vehicle |",
        "| `stages_reached` | How many stages carry a timestamp for this detection |",
        "| `reached_vehicle` | Whether it arrived on any vehicle-side topic |",
        "| `last_stage_reached` | The furthest stage reached, i.e. where it was lost |",
    ]
    (output_dir / "detections_columns.md").write_text("\n".join(lines) + "\n")
