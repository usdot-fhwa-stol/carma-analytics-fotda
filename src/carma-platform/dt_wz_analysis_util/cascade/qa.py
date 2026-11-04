"""Quality report for an assembled cascade table.

Two things this is for. First, per-hop coverage and timing, so it is visible
where detections were lost and which hop costs the most. Second, and more
important, **negative hops**: a stage that appears to happen before the stage
that caused it. That is never physically real, so a persistent negative hop
measures the clock offset between the two hosts involved and nothing else.

Those offsets are reported, not silently corrected. A cross-host hop of "-4 ms"
is a statement about clock sync, and quietly clamping it to zero would hide a
real finding while making the neighbouring hops wrong by the same amount.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from . import cascade_config as config


def stage_summary(table: pd.DataFrame, stages=None) -> pd.DataFrame:
    """Per-hop coverage and latency statistics, one row per consecutive stage pair."""
    stages = [
        stage for stage in (stages or config.STAGE_COLUMNS) if stage in table.columns
    ]
    rows = []
    total = len(table)
    for earlier, later in zip(stages, stages[1:]):
        both = table[earlier].notna() & table[later].notna()
        delta = (table.loc[both, later] - table.loc[both, earlier]).astype(float)
        rows.append(
            {
                "from_stage": earlier,
                "to_stage": later,
                "from_host": config.STAGE_HOST.get(earlier, "?"),
                "to_host": config.STAGE_HOST.get(later, "?"),
                "cross_host": config.STAGE_HOST.get(earlier) != config.STAGE_HOST.get(later),
                "n": int(both.sum()),
                "coverage_pct": round(both.sum() / total * 100, 2) if total else 0.0,
                "min_ms": round(float(delta.min()), 3) if len(delta) else None,
                "median_ms": round(float(delta.median()), 3) if len(delta) else None,
                "p95_ms": round(float(delta.quantile(0.95)), 3) if len(delta) else None,
                "max_ms": round(float(delta.max()), 3) if len(delta) else None,
                "negative_count": int((delta < 0).sum()) if len(delta) else 0,
            }
        )
    return pd.DataFrame(rows)


def clock_offsets(table: pd.DataFrame) -> pd.DataFrame:
    """Estimated clock offset for each cross-host hop.

    A message cannot arrive before it was sent, so the *minimum* observed
    transit time over many samples is the best available estimate of the
    offset between the two hosts' clocks: real transit adds a non-negative
    amount on top of it, and the minimum is where that addition is smallest.
    """
    summary = stage_summary(table)
    cross = summary[summary["cross_host"] & summary["n"].gt(0)].copy()
    cross["offset_estimate_ms"] = cross["min_ms"].apply(
        lambda value: round(-value, 3) if value is not None and value < 0 else 0.0
    )
    cross["clock_skew_suspected"] = cross["negative_count"] > 0
    return cross[
        [
            "from_stage", "to_stage", "from_host", "to_host", "n",
            "min_ms", "median_ms", "negative_count",
            "offset_estimate_ms", "clock_skew_suspected",
        ]
    ]


def report(table: pd.DataFrame, title: str) -> str:
    """Human-readable QA report for one cascade table."""
    lines: List[str] = [title, "=" * len(title), ""]
    lines.append(f"detections: {len(table)}")
    if "reached_vehicle" in table.columns:
        reached = int(table["reached_vehicle"].sum())
        lines.append(
            f"reached the vehicle: {reached}/{len(table)} "
            f"({reached / len(table) * 100:.2f}%)" if len(table) else "reached the vehicle: 0/0"
        )
    if "last_stage_reached" in table.columns:
        lost = table.loc[~table["reached_vehicle"].astype(bool), "last_stage_reached"]
        if len(lost):
            lines.append("")
            lines.append("where detections were lost:")
            for stage, count in lost.value_counts().items():
                lines.append(f"  {count:>5}  last seen at {stage}")

    lines += ["", "per-hop latency", "-" * 15]
    summary = stage_summary(table)
    if not summary.empty:
        lines.append(summary.to_string(index=False))

    offsets = clock_offsets(table)
    suspected = offsets[offsets["clock_skew_suspected"]]
    lines += ["", "cross-host clock offsets", "-" * 24]
    if suspected.empty:
        lines.append("No negative cross-host hops: every host's clock is consistent")
        lines.append("with the causal order of the pipeline.")
    else:
        lines.append("A hop cannot complete before it starts, so a negative transit time")
        lines.append("is a clock offset between the two hosts, not a measurement.")
        lines.append("These offsets are reported, not applied:")
        lines.append("")
        lines.append(suspected.to_string(index=False))

    lines += ["", "notes", "-" * 5]
    lines.append(
        "t_rsu_broadcast is the RSU's transmit instant (this session captured the\n"
        "RSU). Earlier sessions captured the OBU, so their t_ota_capture sits one\n"
        "propagation hop later and totals are not directly comparable."
    )
    lines.append(
        "t_obu_radio_rx is matched by nearest time after the broadcast, not by\n"
        "payload: the OBU capture is tcpdump text and carries no payload bytes.\n"
        "Reliable in aggregate, not per row."
    )
    return "\n".join(lines) + "\n"
