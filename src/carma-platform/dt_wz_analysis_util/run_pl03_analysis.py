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

Writes ``pl03_vehicle_yield.{json,csv,png}``, ``pl03_trajectories.png`` and
``pl03_speed_profile.png``.

Each run also caches its trajectory and pedestrian points to
``pl03_tracks.npz``, so ``--plot-only`` redraws the figures without re-reading
the recordings -- minutes of MCAP parsing for points that cannot have changed.
If the cache is absent, ``--plot-only`` falls back to a full run and builds it.
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
from dt_wz_analysis_util.readers import kafka_log  # noqa: E402

PREFIX = "pl03_vehicle_yield"
TRAJECTORY_NAME = "pl03_trajectories.png"
SPEED_PROFILE_NAME = "pl03_speed_profile.png"

# Resolution of the common distance grid the runs are averaged on.
PROFILE_STEP_M = 0.5

# Cool at rest through hot at speed. Sequential rather than a literal blue-to-red:
# plasma's lightness rises monotonically, so a mid-speed segment cannot read as a
# fast one. --speed-cmap switches it if a literal ramp is wanted.
DEFAULT_SPEED_CMAP = "plasma"

PEDESTRIAN_COLOR = "#6b6b6b"

# Teal for the stops: plasma runs purple through orange to yellow and contains no
# teal, so the marker cannot be mistaken for a point on the speed ramp.
STOP_COLOR = "#00707a"

# Line width in points at the slowest and fastest speed observed. Width runs
# *inverse* to speed, so a halted vehicle draws a thick stroke and a fast one a
# thin thread. This encodes speed a second time, alongside the colour ramp: the
# redundancy is the point, since it keeps the yield legible where the dark end of
# the ramp is hard to separate by eye, and in print or greyscale.
LINEWIDTH_SLOW = 5.0
LINEWIDTH_FAST = 5.0


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
    axis.grid(axis="y", alpha=0.1, linewidth=1.0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def _draw_corridor(axis, tracks, cmap, norm, slowest, span, limits):
    """Draw one panel: the speed-coloured paths plus the pedestrian density."""
    for track in tracks:
        points = np.column_stack([track["east"], track["north"]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Width runs *inverse* to speed, alongside the colour. Two encodings of
        # one quantity, which is redundancy rather than clutter: a halted vehicle
        # draws a thick dark stroke and a fast one a thin bright thread, so the
        # yield is legible from the line's shape even where the ramp's dark end
        # is hard to separate by eye.
        speed = track["speed"][:-1]
        fraction = (speed - slowest) / span if span > 0 else np.zeros_like(speed)
        widths = LINEWIDTH_SLOW + fraction * (LINEWIDTH_FAST - LINEWIDTH_SLOW)

        collection = LineCollection(segments, cmap=cmap, norm=norm,
                                    linewidths=widths, alpha=0.2)
        collection.set_array(speed)
        axis.add_collection(collection)

    # Filled, edgeless, and very transparent. Thousands of reports land in a few
    # square metres, so at this alpha they accumulate into a density: the darker
    # the patch, the more often the pedestrian stood there. An outline on each
    # marker would draw thousands of rings and lose that.
    if tracks:
        axis.scatter(np.concatenate([t["ped_east"] for t in tracks]),
                     np.concatenate([t["ped_north"] for t in tracks]),
                     s=10, color=PEDESTRIAN_COLOR, edgecolors="none",
                     alpha=0.02, zorder=3)

    # Set the limits from the data, then let the axes *box* honour equal aspect.
    # With adjustable="datalim" matplotlib widens the short axis instead, which
    # here padded 25 m of data out to 80 m of mostly empty plot.
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_trajectories(results, output_path, cmap=DEFAULT_SPEED_CMAP):
    """Valid runs above, invalid runs below, on a shared scale.

    Overlaid rather than faceted per run: the runs share one corridor, so
    together they show where the fleet slowed. The yield reads as a cool band
    just short of the pedestrian scatter, which per-run panels would hide.

    Split valid from invalid so the top panel is the corridor the success rate
    was measured over, and the bottom is what was excluded and why it looked
    different. Both panels share the speed ramp, the width scale and the axis
    limits, so the two are directly comparable rather than each self-normalised.

    The phases are called out with arrows rather than plotted markers. A marker
    per stop would add thirty symbols that say only what the colour ramp already
    shows; one arrow names the place.
    """
    scored = [item for item in results if item.track]
    if not scored:
        return None
    groups = [("valid", [i.track for i in scored if i.valid]),
              ("invalid", [i.track for i in scored if not i.valid])]
    tracks = [i.track for i in scored]

    # One scale for both panels: speed, line width and extent all come from the
    # whole set, so a thick dark stroke means the same thing in either.
    peak = max(float(np.nanmax(t["speed"])) for t in tracks)
    slowest = min(float(np.nanmin(t["speed"])) for t in tracks)
    span = peak - slowest
    norm = plt.Normalize(0.0, peak)

    all_east = np.concatenate([t["east"] for t in tracks] + [t["ped_east"] for t in tracks])
    all_north = np.concatenate([t["north"] for t in tracks] + [t["ped_north"] for t in tracks])
    margin = 9.0
    limits = ((all_east.min() - margin, all_east.max() + margin),
              (all_north.min() - margin, all_north.max() + margin))

    # Equal aspect is required -- these are ground positions -- so each panel's
    # height is fixed by the corridor's own shape. Size the figure to match, or
    # the boxes shrink inside oversized subplot areas and leave a band of empty
    # page between them.
    width = 13.0
    plot_width = width - 2.6                      # axis labels and the colourbar
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    figure, axes = plt.subplots(
        2, 1, figsize=(width, 2 * plot_width * aspect + 1.6),
        dpi=300, sharex=True, sharey=True)
    figure.subplots_adjust(hspace=0.18)

    for axis, (label, group) in zip(axes, groups):
        _draw_corridor(axis, group, cmap, norm, slowest, span, limits)
        axis.set_ylabel("North of the start point (m)")
        axis.set_title(f"{label} runs ({len(group)})", loc="left", fontsize=10)
        if not group:
            axis.text(0.5, 0.5, f"no {label} runs", transform=axis.transAxes,
                      ha="center", va="center", fontsize=10, color="#666666")
    axes[-1].set_xlabel("East of the start point (m)")

    axes[0].legend(handles=[
        plt.Line2D([], [], color=plt.get_cmap(cmap)(0.6), linewidth=3.2, alpha=0.7,
                   label="vehicle trajectory"),
        # Drawn at a readable alpha: at the 0.02 used on the plot a single
        # legend marker would be invisible.
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, alpha=0.55,
                   color=PEDESTRIAN_COLOR, label="pedestrian location"),
    ], frameon=False, fontsize=9, loc="lower right")

    # Name the phases where they happen, on the valid panel only: repeating them
    # below would label the same places twice.
    stops = [t["stop"] for t in groups[0][1] if t["stop"]]
    end_east, end_north = pl03._enu([pl03.END_LATLON[0]], [pl03.END_LATLON[1]])
    callouts = [("vehicle start", (0.0, 0.0), (-10, 42))]
    if stops:
        callouts.append(("vehicle yield",
                         (float(np.median([s[0] for s in stops])),
                          float(np.median([s[1] for s in stops]))), (30, -46)))
    callouts.append(("trial end", (float(end_east[0]), float(end_north[0])), (48, 34)))
    callouts.append(("pedestrian", (-75., -7.), (5, 40)))

    for label, (x, y), offset in callouts:
        axes[0].annotate(
            label, xy=(x, y), xycoords="data", xytext=offset, textcoords="offset points",
            fontsize=9, color="#222222", ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color="#222222", linewidth=1.2,
                            shrinkA=2, shrinkB=4),
        )

    bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                          pad=0.02, shrink=0.42)
    bar.set_label("Vehicle speed (m/s)")
    bar.outline.set_visible(False)

    figure.suptitle(f"PL-03: vehicle trajectories and pedestrian positions "
                    f"({len(tracks)} runs)", x=0.01, ha="left", fontsize=11)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _stop_index(track):
    """Index of the slowest sample in the first genuine halt, or None.

    Same rule the scoring uses to find a halt: speed under the threshold, held
    for at least the minimum duration, and only after the vehicle has left its
    parked start. Within that halt the *slowest* sample is the anchor, not the
    first one to cross the threshold. Anchoring on the crossing leaves each run
    aligned a fraction of a metre before it actually came to rest, so the
    averaged curve bottoms out near 0.7 m/s instead of at zero.
    """
    speed = track["speed"]
    travelled = np.hypot(track["east"] - track["east"][0],
                         track["north"] - track["north"][0])
    halted = (speed < pl03.STOP_SPEED_MPS) & (travelled > pl03.MOVED_FROM_START_M)
    if not halted.any():
        return None
    times = track["times"]
    indices = np.flatnonzero(halted)
    for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if times[group[-1]] - times[group[0]] >= pl03.MIN_STOP_SEC:
            return int(group[int(np.argmin(speed[group]))])
    return None


def _distance_along(track, anchor):
    """Distance travelled, in metres, with zero at sample ``anchor``.

    Integrated from speed rather than summed from GPS positions. Summing
    position deltas looks like the obvious choice and is wrong here: the vehicle
    idles at the start, and accumulating noisy fixes there adds around 30 m of
    travel that never happened, so the same route measures 156-166 m instead of
    its true 132 m. Integrating speed gives 126-131 m and is immune to that.
    """
    times, speed = track["times"], track["speed"]
    distance = np.concatenate(
        [[0.0], np.cumsum(np.diff(times) * (speed[1:] + speed[:-1]) / 2.0)])
    return distance - distance[anchor]


def plot_speed_profile(results, output_path):
    """Speed against distance travelled: every run, plus the mean and spread.

    The trajectory plot shows *where* the vehicle slowed; this shows *how much*
    and over what distance, which is what says whether the yield was a controlled
    deceleration or a late stop.
    """
    # Align every run on its own stop, so distance 0 is where the vehicle halted
    # and the averaged curve genuinely reaches zero there. Anchoring at the trial
    # start instead smears the trough, because the runs stop at slightly
    # different distances and the mean never touches 0.
    aligned = [(item.track, _stop_index(item.track))
               for item in results if item.track]
    skipped = [item.run for item, (_t, index) in
               zip([i for i in results if i.track], aligned) if index is None]
    tracks = [(track, index) for track, index in aligned if index is not None]
    if not tracks:
        return None

    figure, axis = plt.subplots(figsize=(11, 4.8))

    profiles = []
    crossings = []
    for track, stop in tracks:
        distance = _distance_along(track, stop)
        profiles.append((distance, track["speed"]))
        axis.plot(distance, track["speed"], color="#999999", linewidth=0.6,
                  alpha=0.55, zorder=2)
        # Where this run's pedestrian sat, in the same distance coordinate.
        if len(track["ped_east"]):
            nearest = int(np.argmin(np.hypot(
                track["east"] - np.median(track["ped_east"]),
                track["north"] - np.median(track["ped_north"]))))
            crossings.append(float(distance[nearest]))

    # Average on a common grid. Each run contributes only over the distance it
    # actually covered, so the mean is not dragged down by runs that had already
    # finished; nanmean over the padding does that for free.
    lowest = max(float(d.min()) for d, _ in profiles)
    highest = min(float(d.max()) for d, _ in profiles)
    # Grid on exact multiples of the step, so x = 0 is always a sample point.
    # Starting the grid at the data's minimum instead leaves 0 between two
    # samples, and the averaged curve then bottoms out either side of the stop
    # at ~0.7 m/s even though every run is anchored at exactly 0.
    grid = np.arange(np.ceil(lowest / PROFILE_STEP_M),
                     np.floor(highest / PROFILE_STEP_M) + 1) * PROFILE_STEP_M
    stacked = np.vstack([np.interp(grid, d, v, left=np.nan, right=np.nan)
                         for d, v in profiles])
    mean = np.nanmean(stacked, axis=0)
    deviation = np.nanstd(stacked, axis=0)

    axis.fill_between(grid, mean - deviation, mean + deviation,
                      color="#2a78d6", alpha=0.22, linewidth=0, zorder=3,
                      label="mean ± 1 s.d.")
    axis.plot(grid, mean, color="#2a78d6", linewidth=2.0, zorder=4, label="mean speed")
    axis.plot([], [], color="#999999", linewidth=0.8, alpha=0.8,
              label=f"individual runs ({len(tracks)})")
    if skipped:
        # Named rather than silently dropped: a run with no stop has no anchor,
        # and it is also the one interesting failure in the set.
        axis.plot([], [], linestyle="", label=f"excluded, never stopped: "
                                              f"{', '.join(sorted(skipped))}")

    if crossings:
        crossing = float(np.median(crossings))
        axis.axvline(crossing, color="#444444", linestyle="--", linewidth=1.0, zorder=1)
        axis.axvline(0.0, color="#2a78d6", linestyle=":", linewidth=1.0, zorder=1)
        axis.annotate("pedestrian", xy=(crossing, axis.get_ylim()[1]),
                      xytext=(-6, -10), textcoords="offset points",
                      fontsize=9, color="#444444", ha="right", va="top")

    axis.set_xlabel("Distance travelled, relative to where the vehicle stopped (m)")
    axis.set_ylabel("Vehicle speed (m/s)")
    axis.set_title("PL-03: speed profile along the trial", loc="left", fontsize=11)
    axis.set_xlim(grid.min(), grid.max())
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, fontsize=9, loc="lower left")
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
    parser.add_argument("--plot-only", action="store_true",
                        help="Redraw the figures from the cached geometry instead of "
                             "re-reading the recordings. Falls back to a full run if "
                             "no cache is present.")
    parser.add_argument("--speed-cmap", default=DEFAULT_SPEED_CMAP,
                        help=f"Colour ramp for speed (default {DEFAULT_SPEED_CMAP}; "
                             f"'coolwarm' for a literal blue-to-red)")
    args = parser.parse_args(argv)

    cache_path = args.output_dir / pl03.CACHE_NAME
    summary_path = args.output_dir / f"{PREFIX}.json"

    results, summary = None, None
    if args.plot_only:
        tracks = pl03.load_tracks(cache_path)
        if tracks and summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            results = [pl03.RunYield.from_row(row) for row in summary["runs_detail"]]
            for item in results:
                item.track = tracks.get(item.run)
            print(f"Redrawing from {cache_path.name} ({len(tracks)} cached runs); "
                  f"the recordings were not read.")
        else:
            missing = "cache" if not tracks else "summary"
            print(f"No {missing} in {args.output_dir}; running the full analysis "
                  f"once to build it.")

    if results is None:
        results = analyse(args.data_root)
        summary = pl03.summarise(results, args.target_pct)
        summary["sessions"] = [str(root) for root in args.data_root]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=float))
        rows = summary["runs_detail"]
        with open(args.output_dir / f"{PREFIX}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if pl03.save_tracks(results, cache_path):
            print(f"Cached trajectories -> {cache_path}")

    plot_summary(results, summary, args.output_dir / f"{PREFIX}.png")
    plot_trajectories(results, args.output_dir / TRAJECTORY_NAME, args.speed_cmap)
    plot_speed_profile(results, args.output_dir / SPEED_PROFILE_NAME)

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
