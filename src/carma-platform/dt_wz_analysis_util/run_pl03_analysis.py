#!/usr/bin/env python3
"""PL-03: CARMA Platform vehicle yield performance, pooled across sessions.

Measures whether the vehicle stopped short of the pedestrian it was warned
about::

    python run_pl03_analysis.py \\
        --data-root .../20260914_verification_test \\
        --data-root .../20260915_verification_test \\
        --output-dir out/pl03

At least 90% of the **valid** runs must yield. A run is valid only if its camera
detections were consistent and its SDSMs were consistently received, so the
metric isolates vehicle behaviour from sensing faults. Invalid runs are still
scored and reported with the reason they were excluded.

Writes ``pl03_vehicle_yield.{json,csv,png}`` and ``pl03_trajectories.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from dt_wz_analysis_util import cp01, dataset, metrics, pl03, report  # noqa: E402
from dt_wz_analysis_util.cascade.plots import stage_colors  # noqa: E402
from dt_wz_analysis_util.portable import kafka_log  # noqa: E402

PREFIX = "pl03_vehicle_yield"
TRAJECTORY_NAME = "pl03_trajectories.png"

# Cool at rest through hot at speed. Sequential rather than a literal blue-to-red:
# plasma's lightness rises monotonically, so a mid-speed segment cannot read as a
# fast one. --speed-cmap switches it if a literal ramp is wanted.
DEFAULT_SPEED_CMAP = "plasma"

PEDESTRIAN_COLOR = "#6b6b6b"

# Teal for the stops: plasma runs purple through orange to yellow and contains no
# teal, so the marker cannot be mistaken for a point on the speed ramp.
STOP_COLOR = "#00707a"


def analyse(data_roots):
    """Score every run, then mark the ones whose detection data was unsound."""
    results = []
    for root in data_roots:
        runs = dataset.load_runs_csv(root / "runs.csv", root)
        session = dataset.discover_session(root)
        print(f"{root.name}: {len(runs)} runs")

        # Criterion [1]: camera detection consistency, from CP-01.
        try:
            drops = cp01.analyse_session(runs, session)
        except Exception as error:
            print(f"  WARNING: camera detection check unavailable: {error}")
            drops = []

        # Criterion [3]: SDSMs consistently received, from CP-02.
        records = (kafka_log.parse_kafka_log_records(session.kafka_detected_object)
                   if session.kafka_detected_object else [])

        session_results = []
        for run in runs:
            try:
                window = report.engaged_window(run)
                result = pl03.measure_run(run, window)
            except Exception as error:
                print(f"  ERROR on {run.name}: {error}")
                result = pl03.RunYield(run=run.name, condition=run.condition,
                                       dwell_sec=run.dwell_sec, note=str(error))
            session_results.append(result)
            verdict = ("YIELD" if result.yielded else
                       "NO YIELD" if result.yielded is False else "not scored")
            print(f"  {run.name:<12} stop={_fmt(result.distance_to_pedestrian_m)}m "
                  f"ahead={_fmt(result.pedestrian_ahead_m)}m "
                  f"approach={_fmt(result.approach_distance_m)}m  {verdict}")

        drop_pct = {}
        for run in runs:
            if not records:
                continue
            try:
                _passed, stats = metrics.detection_to_sdsm_drop_rate(
                    run.mcap, records, report.engaged_window(run))
                if stats.get("drop_rate_pct") is not None:
                    drop_pct[run.name] = stats["drop_rate_pct"]
            except Exception:
                pass

        pl03.apply_validity(session_results, drops, drop_pct)
        results.extend(session_results)
    return results


def _fmt(value):
    return "-" if value is None else f"{value:.1f}"


def plot_summary(results, summary, output_path):
    """Per-run distance from the stop to the pedestrian, by dwell condition."""
    scored = [item for item in results if item.distance_to_pedestrian_m is not None]
    if not scored:
        return None
    conditions = list(summary["by_condition"])
    colors = dict(zip(conditions, stage_colors(range(max(len(conditions), 2)))))

    figure, axis = plt.subplots(figsize=(11, 4.6))
    values = [item.pedestrian_ahead_m for item in scored]
    axis.bar(range(len(scored)), values,
             color=[colors[item.condition] for item in scored],
             edgecolor="white", linewidth=0.8,
             hatch=["" if item.valid else "//" for item in scored])

    axis.axhline(0, color="#444444", linewidth=1.0)
    axis.axhline(-pl03.YIELD_TOLERANCE_M, color="#b00020", linestyle="--", linewidth=1.0)
    axis.text(len(scored) - 0.4, -pl03.YIELD_TOLERANCE_M,
              f" {pl03.YIELD_TOLERANCE_M:g} m tolerance", color="#b00020",
              fontsize=8, va="top", ha="right")

    axis.set_xticks(range(len(scored)))
    axis.set_xticklabels([item.run for item in scored], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("Pedestrian ahead of the stopped vehicle (m)")
    axis.set_title(f"PL-03: vehicle yield — {summary['headline']}", loc="left", fontsize=11)

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=7,
                          color=colors[condition], label=condition)
               for condition in conditions]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#444444",
                                 hatch="//", label="invalid run"))
    axis.legend(handles=handles, frameon=False, fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_trajectories(results, output_path, cmap=DEFAULT_SPEED_CMAP):
    """Every run's path on one axes, coloured by speed, with the pedestrians.

    Overlaid rather than faceted: the runs share one corridor, so together they
    show where the fleet slowed. The yield reads as a cool band just short of the
    pedestrian scatter, which per-run panels would hide.

    The three phases are called out with arrows rather than plotted markers. A
    marker per stop would add thirty symbols that say only what the colour ramp
    already shows; one arrow names the place.
    """
    tracks = [item.track for item in results if item.track]
    if not tracks:
        return None

    peak = max(float(np.nanmax(t["speed"])) for t in tracks)
    norm = plt.Normalize(0.0, peak)
    # The corridor is roughly 140 m by 25 m. Equal aspect is required -- these are
    # ground positions -- so the figure is shaped to match, otherwise the axes
    # box inflates the short dimension into empty space.
    figure, axis = plt.subplots(figsize=(13, 4.8))

    for track in tracks:
        points = np.column_stack([track["east"], track["north"]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Thick and translucent: thirty paths lie on top of one another, so weight
        # carries the corridor while alpha keeps a single outlier visible through it.
        collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3.2, alpha=0.35)
        collection.set_array(track["speed"][:-1])
        axis.add_collection(collection)

    # Filled, edgeless, and very transparent. Roughly 6700 reports land in a few
    # square metres, so at this alpha they accumulate into a density: the darker
    # the patch, the more often the pedestrian stood there. An outline on each
    # marker would draw 6700 rings and lose that.
    axis.scatter(np.concatenate([t["ped_east"] for t in tracks]),
                 np.concatenate([t["ped_north"] for t in tracks]),
                 s=44, color=PEDESTRIAN_COLOR, edgecolors="none",
                 alpha=0.05, zorder=3)

    axis.legend(handles=[
        plt.Line2D([], [], color=plt.get_cmap(cmap)(0.6), linewidth=3.2, alpha=0.7,
                   label="vehicle trajectory"),
        # Drawn at a readable alpha: at the 0.05 used on the plot a single
        # legend marker would be invisible.
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, alpha=0.55,
                   color=PEDESTRIAN_COLOR, label="pedestrian location"),
    ], frameon=False, fontsize=9, loc="lower right")

    all_east = np.concatenate([t["east"] for t in tracks] + [t["ped_east"] for t in tracks])
    all_north = np.concatenate([t["north"] for t in tracks] + [t["ped_north"] for t in tracks])

    # Name the three phases of the trial where they happen.
    stops = [t["stop"] for t in tracks if t["stop"]]
    end_east, end_north = pl03._enu([pl03.END_LATLON[0]], [pl03.END_LATLON[1]])
    callouts = [("vehicle start", (0.0, 0.0), (-10, 42))]
    if stops:
        callouts.append(("vehicle yield",
                         (float(np.median([s[0] for s in stops])),
                          float(np.median([s[1] for s in stops]))), (30, -46)))
    callouts.append(("trial end", (float(end_east[0]), float(end_north[0])), (48, 34)))

    for label, (x, y), offset in callouts:
        axis.annotate(
            label, xy=(x, y), xycoords="data", xytext=offset, textcoords="offset points",
            fontsize=9, color="#222222", ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color="#222222", linewidth=1.2,
                            shrinkA=2, shrinkB=4),
        )

    bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis,
                          pad=0.02, shrink=0.42)
    bar.set_label("Vehicle speed (m/s)")
    bar.outline.set_visible(False)

    axis.set_xlabel("East of the start point (m)")
    axis.set_ylabel("North of the start point (m)")
    axis.set_title(f"PL-03: vehicle trajectories and pedestrian positions "
                   f"({len(tracks)} runs)", loc="left", fontsize=11)

    # Set the limits from the data, then let the axes *box* honour equal aspect.
    # With adjustable="datalim" matplotlib widens the short axis instead, which
    # here padded 25 m of data out to 80 m of mostly empty plot.
    margin = 9.0
    axis.set_xlim(all_east.min() - margin, all_east.max() + margin)
    axis.set_ylim(all_north.min() - margin, all_north.max() + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session directory with runs.csv; repeat to pool sessions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-pct", type=float, default=90.0,
                        help="Required success rate over valid runs (default 90)")
    parser.add_argument("--speed-cmap", default=DEFAULT_SPEED_CMAP,
                        help=f"Colour ramp for speed (default {DEFAULT_SPEED_CMAP}; "
                             f"'coolwarm' for a literal blue-to-red)")
    args = parser.parse_args(argv)

    results = analyse(args.data_root)
    summary = pl03.summarise(results, args.target_pct)
    summary["sessions"] = [str(root) for root in args.data_root]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{PREFIX}.json").write_text(json.dumps(summary, indent=2, default=float))
    rows = summary["runs_detail"]
    with open(args.output_dir / f"{PREFIX}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_summary(results, summary, args.output_dir / f"{PREFIX}.png")
    plot_trajectories(results, args.output_dir / TRAJECTORY_NAME, args.speed_cmap)

    print()
    for condition, bucket in summary["by_condition"].items():
        print(f"  {condition:<8} {bucket['runs']:>2} runs  "
              f"{bucket['valid']:>2} valid  {bucket['yielded']:>2} yielded  "
              f"({bucket['success_rate_pct']}%)")
    approach = summary["approach_distance_m"]
    if approach:
        print(f"\n  approach distance when the pedestrian was first reported: "
              f"mean {approach['mean']} m, median {approach['median']} m "
              f"(range {approach['min']}–{approach['max']} m)")
    if summary["invalid_detail"]:
        print(f"\n  invalid runs ({len(summary['invalid_detail'])}):")
        for item in summary["invalid_detail"]:
            print(f"    {item['run']:<12} {item['reason']}")
    if summary["failed_runs"]:
        print(f"\n  valid runs that did not yield: {', '.join(summary['failed_runs'])}")
    print()
    verdict = ("PASS" if summary["is_passed"] else
               "FAIL" if summary["is_passed"] is False else "not evaluated")
    print(f"PL-03: {summary['headline']} — {verdict} "
          f"(target {summary['target_success_pct']:g}%)")
    print(f"  -> {args.output_dir}")
    return 0 if summary["is_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
