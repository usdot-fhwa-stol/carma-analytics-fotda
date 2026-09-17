"""Shared reporting for the per-run CP metrics pooled across sessions.

CP-02 and CP-03 are both "of N things checked, how many were lost", evaluated
once per run against a 2% limit. They differ only in what they count, so the
windowing, the pooling, the output files and the plot live here and each metric
supplies its own measurement function.

Pooling across sessions is the point: a per-run pass rate says how often the
system met the limit, and the pooled rate says how much was actually lost. Both
are reported, because they answer different questions and can disagree -- a
single bad run can fail on its own and still barely move the pooled figure.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import dataset  # noqa: E402
from . import metrics  # noqa: E402
from .cascade.plots import stage_colors  # noqa: E402
from guidance_scripts import get_engage_time  # noqa: E402

SESSION_TZ = timezone(timedelta(hours=-4))


def _iso(epoch_sec):
    if epoch_sec is None:
        return None
    return datetime.fromtimestamp(epoch_sec, SESSION_TZ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def engaged_window(run):
    """The run's engaged interval in epoch seconds, from its own recording.

    Taken from ``/guidance/state`` rather than from runs.csv, because it marks
    when the vehicle was actually driving the scenario. Every count is confined
    to it so that traffic recorded before or after the run is not scored.
    """
    engage, disengage = get_engage_time(run.mcap)
    return metrics.engaged_window_epoch(run.mcap, engage, disengage)


def analyse_runs(data_roots: List[Path], measure: Callable, prepare: Callable = None,
                 verbose: bool = True) -> List[Dict]:
    """Run ``measure`` over every run of every session.

    ``prepare(session)`` may return a value passed to each ``measure`` call --
    used by CP-02 to parse the session-wide detection log once rather than once
    per run.
    """
    rows: List[Dict] = []
    for root in data_roots:
        runs = dataset.load_runs_csv(root / "runs.csv", root)
        session = dataset.discover_session(root)
        if verbose:
            print(f"{root.name}: {len(runs)} runs")
        context = prepare(session) if prepare else None

        for run in runs:
            row = {
                "session": root.name,
                "run": run.name,
                "condition": run.condition,
                "dwell_sec": run.dwell_sec,
            }
            try:
                window = engaged_window(run)
                row["window_start"] = _iso(window[0])
                row["window_end"] = _iso(window[1])
                row["window_duration_sec"] = round(window[1] - window[0], 3)
                row.update(measure(run, window, context))
            except Exception as error:
                print(f"  ERROR on {run.name}: {error}")
                row["error"] = str(error)
            rows.append(row)
            if verbose:
                print(f"  {run.name:<12} checked={row.get('checked')} "
                      f"dropped={row.get('dropped')} "
                      f"rate={row.get('drop_rate_pct')} "
                      f"{'PASS' if row.get('passed') else 'FAIL' if row.get('passed') is False else '-'}")
    return rows


def summarise(rows: List[Dict], metric: str, threshold_pct: float) -> Dict:
    """Pooled totals and the per-run pass rate."""
    evaluated = [row for row in rows if row.get("passed") is not None]
    passed = sum(1 for row in evaluated if row["passed"])
    checked = sum(int(row.get("checked") or 0) for row in rows)
    dropped = sum(int(row.get("dropped") or 0) for row in rows)

    by_condition: Dict[str, Dict] = {}
    for row in rows:
        bucket = by_condition.setdefault(
            row["condition"], {"runs": 0, "checked": 0, "dropped": 0, "passed": 0}
        )
        bucket["runs"] += 1
        bucket["checked"] += int(row.get("checked") or 0)
        bucket["dropped"] += int(row.get("dropped") or 0)
        bucket["passed"] += 1 if row.get("passed") else 0
    for bucket in by_condition.values():
        bucket["drop_rate_pct"] = (
            round(bucket["dropped"] / bucket["checked"] * 100.0, 3) if bucket["checked"] else None
        )

    return {
        "metric": metric,
        "threshold_pct": threshold_pct,
        "runs": len(rows),
        "runs_evaluated": len(evaluated),
        "runs_passed": passed,
        "runs_failed": len(evaluated) - passed,
        "pass_rate": f"{passed / len(evaluated) * 100:.1f}%" if evaluated else "n/a",
        "total_checked": checked,
        "total_dropped": dropped,
        "pooled_drop_rate_pct": round(dropped / checked * 100.0, 3) if checked else None,
        "headline": f"{dropped} of {checked} lost ({dropped / checked * 100:.2f}%)" if checked else "n/a",
        "failed_runs": [row["run"] for row in evaluated if not row["passed"]],
        "by_condition": dict(
            sorted(by_condition.items(), key=lambda item: dataset.condition_dwell_sec(item[0]))
        ),
        "runs_detail": rows,
    }


def summarise_latency(rows: List[Dict], metric: str, threshold: float, unit: str,
                      value_key: str = "median_s") -> Dict:
    """Pooled summary for a latency metric rather than a drop-rate one.

    The pooled figure is the median of the per-run medians, not a mean of means:
    latency distributions here have long right tails from stalls, and a mean
    would follow the tail rather than the typical message.
    """
    evaluated = [row for row in rows if row.get("passed") is not None]
    passed = sum(1 for row in evaluated if row["passed"])
    values = [row[value_key] for row in rows if row.get(value_key) is not None]
    samples = sum(int(row.get("samples") or 0) for row in rows)

    by_condition: Dict[str, Dict] = {}
    for row in rows:
        bucket = by_condition.setdefault(row["condition"], {"runs": 0, "passed": 0, "values": []})
        bucket["runs"] += 1
        bucket["passed"] += 1 if row.get("passed") else 0
        if row.get(value_key) is not None:
            bucket["values"].append(row[value_key])
    for bucket in by_condition.values():
        collected = bucket.pop("values")
        bucket[f"median_{unit}"] = (
            round(float(np.median(collected)), 4) if collected else None
        )

    return {
        "metric": metric,
        "threshold": threshold,
        "unit": unit,
        "runs": len(rows),
        "runs_evaluated": len(evaluated),
        "runs_passed": passed,
        "runs_failed": len(evaluated) - passed,
        "pass_rate": f"{passed / len(evaluated) * 100:.1f}%" if evaluated else "n/a",
        "total_samples": samples,
        "pooled_median": round(float(np.median(values)), 4) if values else None,
        "worst_run_value": round(float(np.max(values)), 4) if values else None,
        "headline": (
            f"median {np.median(values):.3f} {unit} over {samples} samples" if values else "n/a"
        ),
        "failed_runs": [row["run"] for row in evaluated if not row["passed"]],
        "by_condition": dict(
            sorted(by_condition.items(), key=lambda item: dataset.condition_dwell_sec(item[0]))
        ),
        "runs_detail": rows,
    }


def plot_latency(rows: List[Dict], summary: Dict, title: str, output_path,
                 value_key: str = "median_s"):
    """Per-run median latency against the limit."""
    usable = [row for row in rows if row.get(value_key) is not None]
    if not usable:
        return None
    conditions = list(summary["by_condition"])
    colors = dict(zip(conditions, stage_colors(range(max(len(conditions), 2)))))

    figure, axis = plt.subplots(figsize=(11, 4.6))
    values = [row[value_key] for row in usable]
    axis.bar(range(len(usable)), values,
             color=[colors[row["condition"]] for row in usable],
             edgecolor="white", linewidth=0.8)
    axis.axhline(summary["threshold"], color="#b00020", linestyle="--", linewidth=1.0)
    axis.text(len(usable) - 0.4, summary["threshold"],
              f" {summary['threshold']:g} {summary['unit']} limit", color="#b00020",
              fontsize=8, va="bottom", ha="right")

    axis.set_xticks(range(len(usable)))
    axis.set_xticklabels([row["run"] for row in usable], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel(f"Median latency ({summary['unit']})")
    axis.set_title(
        f"{title} — {summary['headline']}, "
        f"{summary['runs_passed']}/{summary['runs_evaluated']} runs passed",
        loc="left", fontsize=11,
    )
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=7, color=colors[condition],
                   label=f"{condition} ({summary['by_condition'][condition]['runs']} runs)")
        for condition in conditions
    ]
    axis.legend(handles=handles, frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_rates(rows: List[Dict], summary: Dict, title: str, output_path):
    """Per-run drop rate, coloured by condition, against the limit."""
    usable = [row for row in rows if row.get("drop_rate_pct") is not None]
    if not usable:
        return None
    conditions = list(summary["by_condition"])
    colors = dict(zip(conditions, stage_colors(range(max(len(conditions), 2)))))

    figure, axis = plt.subplots(figsize=(11, 4.6))
    values = [row["drop_rate_pct"] for row in usable]
    axis.bar(range(len(usable)), values,
             color=[colors[row["condition"]] for row in usable],
             edgecolor="white", linewidth=0.8)
    for index, row in enumerate(usable):
        if row["drop_rate_pct"]:
            axis.text(index, row["drop_rate_pct"], f"{row['dropped']}",
                      ha="center", va="bottom", fontsize=7, color="#333333")

    axis.axhline(summary["threshold_pct"], color="#b00020", linestyle="--", linewidth=1.0)
    axis.text(len(usable) - 0.4, summary["threshold_pct"],
              f" {summary['threshold_pct']:g}% limit", color="#b00020",
              fontsize=8, va="bottom", ha="right")

    axis.set_xticks(range(len(usable)))
    axis.set_xticklabels([row["run"] for row in usable], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("Drop rate (%)")
    axis.set_title(
        f"{title} — {summary['headline']}, "
        f"{summary['runs_passed']}/{summary['runs_evaluated']} runs passed",
        loc="left", fontsize=11,
    )
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=7, color=colors[condition],
                   label=f"{condition} ({summary['by_condition'][condition]['runs']} runs)")
        for condition in conditions
    ]
    axis.legend(handles=handles, frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def write_outputs(summary: Dict, output_dir: Path, prefix: str, title: str) -> None:
    """Write the JSON, the per-run CSV and the plot, mirroring CP-01's layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summary["runs_detail"]

    (output_dir / f"{prefix}.json").write_text(json.dumps(summary, indent=2, default=float))

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_dir / f"{prefix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if "unit" in summary:
        plot_latency(rows, summary, title, output_dir / f"{prefix}.png")
    else:
        plot_rates(rows, summary, title, output_dir / f"{prefix}.png")

    print()
    latency = "unit" in summary
    for condition, bucket in summary["by_condition"].items():
        if latency:
            value = bucket.get(f"median_{summary['unit']}")
            print(f"  {condition:<8} {bucket['runs']:>2} runs  "
                  f"median {value if value is not None else 'n/a'} {summary['unit']}  "
                  f"{bucket['passed']}/{bucket['runs']} passed")
            continue
        rate = bucket["drop_rate_pct"]
        print(f"  {condition:<8} {bucket['runs']:>2} runs  "
              f"{bucket['dropped']:>4}/{bucket['checked']:<6} lost  "
              f"({rate if rate is not None else 'n/a'}%)  "
              f"{bucket['passed']}/{bucket['runs']} passed")
    print()
    limit = (f"{summary['threshold']:g} {summary['unit']}" if latency
             else f"{summary['threshold_pct']:g}%")
    print(f"{summary['metric']}: {summary['headline']}")
    print(f"       per-run pass rate: {summary['runs_passed']}/{summary['runs_evaluated']} "
          f"({summary['pass_rate']}), limit {limit} per run")
    if summary["failed_runs"]:
        print(f"       failed: {', '.join(summary['failed_runs'])}")
    print(f"  -> {output_dir}")
