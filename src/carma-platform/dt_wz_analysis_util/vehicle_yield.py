"""Does the vehicle yield to the pedestrian it was warned about?

Every other measurement in the suite asks whether the SDSM arrived and how
quickly. This one asks what CARMA Platform did with it: a target share of the
**valid** runs must end with the vehicle stopping short of the pedestrian.

A run is valid only if its detection data was sound, so the metric isolates
vehicle behaviour from sensing faults. Invalid runs are still scored and reported
with the reason they were excluded.

Three properties of the recordings shape how a trial is located, each found by
probing the data rather than assumed:

**The engaged window is the trial.** Each recording also contains the vehicle
driving *back* to the start for the next run. Taking the last contact with the
start fence and the first contact with the end fence over the whole recording
inverts the window on 17 of 30 runs. Bounding the search by ``/guidance/state``
fixes it.

**Arc length along the GPS track is unusable.** The vehicle idles at the start
for seconds, and summing noisy position deltas accrues tens of metres of phantom
travel -- enough to shift the pedestrian's apparent position by 27 m between
runs. Every distance here is point-to-point.

**The route curves.** The vehicle deviates up to 11.5 m from a straight
start-to-end line, so projecting onto that line misplaces the crossing. Measured
against the driven path the pedestrian's closest approach is 0.1-2.9 m, which
confirms they really do cross it.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from . import config as cfg  # noqa: E402
from . import camera_detections, dataset, metrics, report  # noqa: E402
from .cascade.plots import stage_colors  # noqa: E402
from .readers import kafka_log, mcap_reader  # noqa: E402

# The geodesy lives in the carma-streets tree, beside the spoofing verifier.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "carma-streets"))
from sdsm_location_spoofing_verification import geodetic_to_enu  # noqa: E402

# Every threshold, fence and topic name this module used to define is now in
# ``config.Pl03Config``, ``config.SiteGeometry`` and ``config.VehicleTopics``,
# and is taken as an argument. The names below are kept as re-exports so the
# cache file and the plot code have one place to read them from.
CACHE_NAME = cfg.VEHICLE_YIELD.cache_name
YIELD_TOLERANCE_M = cfg.VEHICLE_YIELD.yield_tolerance_m


@dataclass
class RunYield:
    """One run's yield assessment."""

    run: str
    condition: str
    dwell_sec: int
    valid: bool = True
    invalid_reason: Optional[str] = None
    scored: bool = False
    note: Optional[str] = None

    trial_duration_sec: Optional[float] = None
    peak_speed_mps: Optional[float] = None
    stop_count: int = 0
    stop_time_sec: Optional[float] = None
    stop_duration_sec: Optional[float] = None
    distance_to_pedestrian_m: Optional[float] = None
    pedestrian_ahead_m: Optional[float] = None
    approach_distance_m: Optional[float] = None
    resume_clearance_m: Optional[float] = None
    start_gap_m: Optional[float] = None
    end_gap_m: Optional[float] = None
    yielded: Optional[bool] = None

    # Raw geometry, kept for the trajectory plot rather than the table.
    track: Optional[Dict] = field(default=None, repr=False)

    def as_row(self) -> Dict:
        return {
            "run": self.run,
            "condition": self.condition,
            "dwell_sec": self.dwell_sec,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "yielded": self.yielded,
            "trial_duration_sec": _round(self.trial_duration_sec, 1),
            "peak_speed_mps": _round(self.peak_speed_mps, 2),
            "stop_count": self.stop_count,
            "stop_time_sec": _round(self.stop_time_sec, 1),
            "stop_duration_sec": _round(self.stop_duration_sec, 1),
            "distance_to_pedestrian_m": _round(self.distance_to_pedestrian_m, 1),
            "pedestrian_ahead_m": _round(self.pedestrian_ahead_m, 1),
            "approach_distance_m": _round(self.approach_distance_m, 1),
            "start_gap_m": _round(self.start_gap_m, 1),
            "end_gap_m": _round(self.end_gap_m, 1),
            "resume_clearance_m": _round(self.resume_clearance_m, 1),
            "note": self.note,
        }


    @classmethod
    def from_row(cls, row: Dict) -> "RunYield":
        """Rebuild the fields the plots need from a saved CSV/JSON row."""
        item = cls(run=row["run"], condition=row["condition"],
                   dwell_sec=int(row.get("dwell_sec") or 0))
        for field_name in ("valid", "yielded"):
            value = row.get(field_name)
            item.__dict__[field_name] = (
                value if isinstance(value, bool)
                else str(value).lower() == "true" if value not in (None, "") else None
            )
        for field_name in ("distance_to_pedestrian_m", "pedestrian_ahead_m",
                           "approach_distance_m", "stop_duration_sec"):
            value = row.get(field_name)
            item.__dict__[field_name] = (
                float(value) if value not in (None, "") else None)
        item.invalid_reason = row.get("invalid_reason") or None
        return item


def _round(value, places):
    return None if value is None else round(float(value), places)


# Arrays saved per run. ``stop`` is stored as two values, NaN when the run never
# stopped, so the file stays a flat array archive rather than needing pickling.
_TRACK_ARRAYS = ("times", "east", "north", "speed", "ped_east", "ped_north")


def save_tracks(results: List["RunYield"], cache_path) -> Optional[Path]:
    """Cache each run's geometry so the plots can be redrawn without the MCAPs.

    Re-reading 30 recordings costs minutes and yields exactly the same points,
    so iterating on a figure should not pay for it.
    """
    payload = {}
    names = []
    for item in results:
        if not item.track:
            continue
        names.append(item.run)
        for key in _TRACK_ARRAYS:
            payload[f"{item.run}|{key}"] = np.asarray(item.track[key], dtype=float)
        stop = item.track.get("stop")
        payload[f"{item.run}|stop"] = np.array(
            stop if stop else (np.nan, np.nan), dtype=float)
    if not names:
        return None
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, runs=np.array(names), **payload)
    return cache_path


def load_tracks(cache_path) -> Dict[str, Dict]:
    """Read back what ``save_tracks`` wrote, or {} if there is no cache."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return {}
    with np.load(cache_path, allow_pickle=False) as archive:
        tracks = {}
        for run in archive["runs"]:
            run = str(run)
            track = {key: archive[f"{run}|{key}"] for key in _TRACK_ARRAYS}
            stop = archive[f"{run}|stop"]
            track["stop"] = None if np.isnan(stop).any() else (float(stop[0]), float(stop[1]))
            tracks[run] = track
    return tracks


def to_enu(latitudes, longitudes,
           origin_latlon=None) -> Tuple[np.ndarray, np.ndarray]:
    """Local east/north metres about ``origin_latlon``, the trial's start point.

    Public because the plots work in the same frame and must use the same
    origin; a second conversion with a different origin would silently shift
    every figure relative to the table.
    """
    origin = origin_latlon if origin_latlon is not None else cfg.SITE.start_latlon
    east, north = geodetic_to_enu(
        np.asarray(latitudes, dtype=float), np.asarray(longitudes, dtype=float), *origin
    )
    return np.atleast_1d(east), np.atleast_1d(north)


# Kept as the private spelling the module already used internally.
_enu = to_enu


def _read_topic(mcap_path, topic, extract, window=None):
    """(times, values) for one topic, windowed only if a window is given."""
    counts = mcap_reader.topic_message_counts(mcap_path)
    if not counts.get(topic):
        return np.array([]), []
    reader, _type_map, _start = mcap_reader.open_bagfile(str(mcap_path), topics=[topic])
    times, values = [], []
    while reader.has_next():
        _topic, message, log_time_ns = reader.read_next()
        seconds = log_time_ns / 1e9
        if window is None or window[0] <= seconds <= window[1]:
            times.append(seconds)
            values.append(extract(message.to_dict()))
    return np.array(times, dtype=float), values


def halt_groups(times, speeds, distance_from_start,
                config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD):
    """Index groups for each halt after the vehicle has set off.

    The stationary start is excluded by distance travelled rather than by
    ignoring the first stop: a run can legitimately halt twice, and a run can
    idle for a variable time before moving.

    Returned as index groups rather than times because the plots need to pick a
    sample *within* a halt -- the slowest one -- to anchor on, and rebuilding
    that rule beside the scoring is how the two drift apart.
    """
    if len(times) == 0:
        return []
    halted = ((speeds < config.stop_speed_mps)
              & (distance_from_start > config.moved_from_start_m))
    if not halted.any():
        return []
    indices = np.flatnonzero(halted)
    groups = []
    for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if times[group[-1]] - times[group[0]] >= config.min_stop_sec:
            groups.append(group)
    return groups


def find_stops(times, speeds, distance_from_start,
               config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD) -> List[Tuple[float, float]]:
    """(start_time, duration) for each halt, from ``halt_groups``."""
    return [
        (float(times[group[0]]), float(times[group[-1]] - times[group[0]]))
        for group in halt_groups(times, speeds, distance_from_start, config)
    ]


def score_stop(stop_time, vehicle_times, east, north, pedestrian_east, pedestrian_north,
               config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD):
    """(distance, ahead) from the vehicle at ``stop_time`` to the pedestrian.

    ``ahead`` is the pedestrian's distance projected onto the vehicle's heading:
    positive means in front, negative means the vehicle has already passed. That
    sign is what separates a yield from a stop beyond the crossing, and a plain
    distance cannot express it.
    """
    x = float(np.interp(stop_time, vehicle_times, east))
    y = float(np.interp(stop_time, vehicle_times, north))
    distance = float(np.hypot(pedestrian_east - x, pedestrian_north - y))

    index = int(np.argmin(np.abs(vehicle_times - stop_time)))
    earlier = max(index - config.heading_lookback, 0)
    heading = np.array([east[index] - east[earlier], north[index] - north[earlier]])
    magnitude = float(np.hypot(*heading))
    if magnitude == 0:
        return distance, distance  # no heading available; treat as straight ahead
    ahead = float(np.dot([pedestrian_east - x, pedestrian_north - y], heading) / magnitude)
    return distance, ahead


def measure_run(run, window, config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD,
                site: cfg.SiteGeometry = cfg.SITE,
                topics: cfg.VehicleTopics = cfg.TOPICS) -> RunYield:
    """Score one run's yield behaviour over its engaged window.

    ``config`` carries the fences, the halt rule and the yield tolerance;
    ``site`` the surveyed route endpoints; ``topics`` the vehicle's GPS and
    speed topic names. Nothing about this trial is fixed inside the function.
    """
    result = RunYield(run=run.name, condition=run.condition, dwell_sec=run.dwell_sec)

    # Read the whole GPS track once: the trial is scored inside the engaged
    # window, but the end fence has to be checked against the full recording.
    all_t, all_fixes = _read_topic(run.mcap, topics.gps_fix, lambda d: (d["latitude"], d["longitude"]))
    if len(all_t) < 10:
        result.note = "no vehicle GPS in this recording"
        return result
    origin = site.start_latlon
    all_east, all_north = to_enu(
        [f[0] for f in all_fixes], [f[1] for f in all_fixes], origin)
    in_window = (all_t >= window[0]) & (all_t <= window[1])
    if in_window.sum() < 10:
        result.note = "no vehicle GPS in the engaged window"
        return result
    fixes_t = all_t[in_window]
    east, north = all_east[in_window], all_north[in_window]

    speeds_t, speeds = _read_topic(
        run.mcap, topics.twist, lambda d: d["twist"]["linear"]["x"], window)
    if len(speeds_t) < 10:
        result.note = "no vehicle speed in the engaged window"
        return result
    speeds = np.asarray(speeds, dtype=float)

    positions = metrics.sdsm_object_positions(run.mcap, window)
    if not positions:
        result.note = "no SDSM object positions in the engaged window"
        return result
    ped_east, ped_north = to_enu(
        [p["latitude"] for p in positions], [p["longitude"] for p in positions], origin)
    crossing_east = float(np.median(ped_east))
    crossing_north = float(np.median(ped_north))

    # Confirm the vehicle drove the intended route. The start is checked inside
    # the engaged window; the end over the whole recording, since guidance
    # normally hands back before the vehicle gets there.
    end_east, end_north = to_enu([site.end_latlon[0]], [site.end_latlon[1]], origin)
    start_gap = float(np.hypot(east, north).min())
    end_gap = float(np.hypot(all_east - end_east[0], all_north - end_north[0]).min())
    result.start_gap_m = start_gap
    result.end_gap_m = end_gap
    if start_gap > config.start_tolerance_m:
        result.note = f"never within {config.start_tolerance_m:g} m of the start point"
        return result
    if end_gap > config.end_tolerance_m:
        result.note = f"never within {config.end_tolerance_m:g} m of the end point"
        return result

    result.scored = True
    result.trial_duration_sec = float(fixes_t[-1] - fixes_t[0])
    result.peak_speed_mps = float(speeds.max())

    travelled = np.interp(speeds_t, fixes_t, np.hypot(east - east[0], north - north[0]))
    stops = find_stops(speeds_t, speeds, travelled, config)
    result.stop_count = len(stops)

    # Of the qualifying stops, the one nearest the pedestrian is the yield
    # candidate; a vehicle may also halt for unrelated reasons.
    best = None
    for stop_time, duration in stops:
        distance, ahead = score_stop(
            stop_time, fixes_t, east, north, crossing_east, crossing_north, config)
        if best is None or distance < best[2]:
            best = (stop_time, duration, distance, ahead)

    if best is not None:
        stop_time, duration, distance, ahead = best
        result.stop_time_sec = stop_time - window[0]
        result.stop_duration_sec = duration
        result.distance_to_pedestrian_m = distance
        result.pedestrian_ahead_m = ahead
        result.yielded = bool(ahead > -config.yield_tolerance_m
                              and distance <= config.yield_max_distance_m)

        # Where the pedestrian was when the vehicle pulled away again.
        resumed = speeds_t[(speeds_t > stop_time + duration) & (speeds > config.stop_speed_mps)]
        if len(resumed):
            times = np.array([p["receive_time_sec"] for p in positions])
            nearest = int(np.argmin(np.abs(times - resumed[0])))
            result.resume_clearance_m = float(
                np.hypot(ped_east[nearest] - crossing_east, ped_north[nearest] - crossing_north))
    else:
        result.yielded = False
        result.note = "vehicle never stopped during the trial"

    # Characterisation: how far the vehicle still had to travel when the
    # pedestrian was first reported. Replaces the 40 m validity gate.
    first_seen = min(p["receive_time_sec"] for p in positions)
    x = float(np.interp(first_seen, fixes_t, east))
    y = float(np.interp(first_seen, fixes_t, north))
    result.approach_distance_m = float(np.hypot(crossing_east - x, crossing_north - y))

    result.track = {
        "times": fixes_t, "east": east, "north": north,
        "speed": np.interp(fixes_t, speeds_t, speeds),
        "ped_east": ped_east, "ped_north": ped_north,
        "stop": None if best is None else (
            float(np.interp(best[0], fixes_t, east)), float(np.interp(best[0], fixes_t, north))),
    }
    return result


def apply_validity(results: List[RunYield], drops, sdsm_drop_pct: Dict[str, float],
                   config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD) -> None:
    """Mark runs invalid where the detection data cannot support the test.

    Both criteria are applied and the reason is recorded, so the root cause of
    every excluded run is logged as the test description requires.
    """
    by_run = {item.run: item for item in drops}
    for result in results:
        reasons = []
        detection = by_run.get(result.run)
        if detection is not None:
            gap = max(detection.gap_spans_ms) / 1000.0 if detection.gap_spans_ms else 0.0
            if gap >= config.detection_gap_limit_sec:
                reasons.append(f"camera detection gap of {gap:.1f} s")
            received = detection.received_frames or 0
            lost = detection.lost_after_camera or 0
            lost_pct = (lost / received * 100.0) if received else 0.0
            if lost_pct > config.plugin_loss_limit_pct:
                reasons.append(
                    f"{lost} of {received} frames ({lost_pct:.1f}%) lost inside the "
                    f"V2XHub plugin")
        drop = sdsm_drop_pct.get(result.run)
        if drop is not None and drop > config.sdsm_drop_limit_pct:
            reasons.append(f"SDSM drop rate {drop:.2f}% over "
                           f"{config.sdsm_drop_limit_pct:g}%")
        if not result.scored and result.note:
            reasons.append(result.note)
        if reasons:
            result.valid = False
            result.invalid_reason = "; ".join(reasons)


def summarise(results: List[RunYield],
              target_pct: float = cfg.VEHICLE_YIELD.target_success_pct,
              config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD,
              metric: str = "vehicle yield") -> Dict:
    """Success rate over the valid runs, with the invalid ones accounted for."""
    valid = [item for item in results if item.valid and item.yielded is not None]
    succeeded = [item for item in valid if item.yielded]
    rate = len(succeeded) / len(valid) * 100.0 if valid else None

    approach = [item.approach_distance_m for item in results
                if item.approach_distance_m is not None]
    by_condition: Dict[str, Dict] = {}
    for item in results:
        bucket = by_condition.setdefault(
            item.condition, {"runs": 0, "valid": 0, "yielded": 0})
        bucket["runs"] += 1
        if item.valid and item.yielded is not None:
            bucket["valid"] += 1
            bucket["yielded"] += 1 if item.yielded else 0
    for bucket in by_condition.values():
        bucket["success_rate_pct"] = (
            round(bucket["yielded"] / bucket["valid"] * 100.0, 1) if bucket["valid"] else None)

    return {
        "metric": metric,
        "target_success_pct": target_pct,
        "runs": len(results),
        "valid_runs": len(valid),
        "invalid_runs": len(results) - len(valid),
        "yielded": len(succeeded),
        "failed": len(valid) - len(succeeded),
        "success_rate_pct": round(rate, 1) if rate is not None else None,
        "is_passed": None if rate is None else bool(rate >= target_pct),
        "headline": (
            f"{len(succeeded)} of {len(valid)} valid runs yielded ({rate:.1f}%)"
            if rate is not None else "no valid runs"),
        "yield_tolerance_m": config.yield_tolerance_m,
        # Characterisation, replacing the 40 m validity gate.
        "approach_distance_m": {
            "mean": round(float(np.mean(approach)), 1),
            "median": round(float(np.median(approach)), 1),
            "min": round(float(np.min(approach)), 1),
            "max": round(float(np.max(approach)), 1),
        } if approach else None,
        "failed_runs": [item.run for item in valid if not item.yielded],
        "invalid_detail": [
            {"run": item.run, "reason": item.invalid_reason}
            for item in results if not item.valid
        ],
        "by_condition": by_condition,
        "runs_detail": [item.as_row() for item in results],
    }


# --------------------------------------------------------------------------
# Driving the analysis
# --------------------------------------------------------------------------

def _fmt(value):
    return "-" if value is None else f"{value:.1f}"


def analyse(data_roots, config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD,
            site: cfg.SiteGeometry = cfg.SITE,
            topics: cfg.VehicleTopics = cfg.TOPICS,
            camera_config: cfg.CameraDetectionConfig = cfg.CAMERA_DETECTIONS,
            layout: cfg.SessionLayout = cfg.LAYOUT) -> List[RunYield]:
    """Score every run, then mark the ones whose detection data was unsound.

    The two validity criteria reuse the camera-detection and detection-delivery
    measurements rather than reimplementing them, so "the camera detected
    consistently" means the same thing here as it does in the reports those
    measurements produce.
    """
    results: List[RunYield] = []
    for root in data_roots:
        runs, session = dataset.load_session(root, layout)
        print(f"{root.name}: {len(runs)} runs")

        # Criterion [1]: camera detection consistency.
        try:
            drops = camera_detections.analyse_session(runs, session, camera_config)
        except Exception as error:
            print(f"  WARNING: camera detection check unavailable: {error}")
            drops = []

        # Criterion [3]: SDSMs consistently received.
        records = (kafka_log.parse_kafka_log_records(session.kafka_detected_object)
                   if session.kafka_detected_object else [])

        session_results, windows = [], {}
        for run in runs:
            try:
                window = report.engaged_window(run)
                windows[run.name] = window
                result = measure_run(run, window, config, site, topics)
            except Exception as error:
                print(f"  ERROR on {run.name}: {error}")
                result = RunYield(run=run.name, condition=run.condition,
                                  dwell_sec=run.dwell_sec, note=str(error))
            session_results.append(result)
            verdict = ("YIELD" if result.yielded else
                       "NO YIELD" if result.yielded is False else "not scored")
            print(f"  {run.name:<12} stop={_fmt(result.distance_to_pedestrian_m)}m "
                  f"ahead={_fmt(result.pedestrian_ahead_m)}m "
                  f"approach={_fmt(result.approach_distance_m)}m  {verdict}")

        drop_pct = {}
        for run in runs:
            if not records or run.name not in windows:
                continue
            try:
                _passed, stats = metrics.detection_to_sdsm_drop_rate(
                    run.mcap, records, windows[run.name])
                if stats.get("drop_rate_pct") is not None:
                    drop_pct[run.name] = stats["drop_rate_pct"]
            except Exception:
                pass

        apply_validity(session_results, drops, drop_pct, config)
        results.extend(session_results)
    return results


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_summary(results: List[RunYield], summary: Dict, output_path, title: str,
                 config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD):
    """Per-run distance from the stop to the pedestrian, by dwell condition."""
    scored = [item for item in results if item.distance_to_pedestrian_m is not None]
    if not scored:
        return None
    conditions = list(summary["by_condition"])
    colors = dict(zip(conditions, stage_colors(range(max(len(conditions), 2)))))

    figure, axis = plt.subplots(figsize=(11, 4.6))
    axis.bar(range(len(scored)), [item.pedestrian_ahead_m for item in scored],
             color=[colors[item.condition] for item in scored],
             edgecolor="white", linewidth=0.8,
             hatch=["" if item.valid else "//" for item in scored])

    axis.axhline(0, color="#444444", linewidth=1.0)
    cfg.threshold_line(axis, -config.yield_tolerance_m,
                       f"{config.yield_tolerance_m:g} m tolerance",
                       right_at=len(scored) - 0.4, va="top")

    axis.set_xticks(range(len(scored)))
    axis.set_xticklabels([item.run for item in scored], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("Pedestrian ahead of the stopped vehicle (m)")
    axis.set_title(f"{title} — {summary['headline']}", loc="left", fontsize=11)

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=7,
                          color=colors[condition], label=condition)
               for condition in conditions]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#444444",
                                 hatch="//", label="invalid run"))
    axis.legend(handles=handles, frameon=False, fontsize=8, ncol=2)
    cfg.grid(axis)
    cfg.despine(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=cfg.STYLE.figure_dpi)
    plt.close(figure)
    return output_path


def _draw_corridor(axis, tracks, cmap, norm, slowest, span, limits,
                   style: cfg.PlotStyle = cfg.STYLE):
    """Draw one panel: the speed-coloured paths plus the pedestrian density."""
    for track in tracks:
        points = np.column_stack([track["east"], track["north"]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Width runs *inverse* to speed, alongside the colour. Two encodings of
        # one quantity, which is redundancy rather than clutter: a halted vehicle
        # draws a thick dark stroke and a fast one a thin bright thread, so the
        # yield is legible from the line's shape even where the ramp's dark end
        # is hard to separate by eye. Setting the two widths equal in the style
        # turns this off and leaves colour to carry speed alone.
        speed = track["speed"][:-1]
        fraction = (speed - slowest) / span if span > 0 else np.zeros_like(speed)
        widths = style.linewidth_slow + fraction * (
            style.linewidth_fast - style.linewidth_slow)

        collection = LineCollection(segments, cmap=cmap, norm=norm,
                                    linewidths=widths, alpha=style.trajectory_alpha)
        collection.set_array(speed)
        axis.add_collection(collection)

    # Filled, edgeless, and very transparent. Thousands of reports land in a few
    # square metres, so at this alpha they accumulate into a density: the darker
    # the patch, the more often the pedestrian stood there. An outline on each
    # marker would draw thousands of rings and lose that.
    if tracks:
        axis.scatter(np.concatenate([t["ped_east"] for t in tracks]),
                     np.concatenate([t["ped_north"] for t in tracks]),
                     s=10, color=style.pedestrian_color, edgecolors="none",
                     alpha=style.pedestrian_alpha, zorder=3)

    # Set the limits from the data, then let the axes *box* honour equal aspect.
    # With adjustable="datalim" matplotlib widens the short axis instead, which
    # here padded 25 m of data out to 80 m of mostly empty plot.
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2, linewidth=style.grid_linewidth)
    axis.set_axisbelow(True)
    cfg.despine(axis)


def plot_trajectories(results: List[RunYield], output_path, title: str, cmap=None,
                      site: cfg.SiteGeometry = cfg.SITE,
                      style: cfg.PlotStyle = cfg.STYLE):
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
    cmap = cmap or style.speed_colormap
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
        _draw_corridor(axis, group, cmap, norm, slowest, span, limits, style)
        axis.set_ylabel("North of the start point (m)")
        axis.set_title(f"{label} runs ({len(group)})", loc="left", fontsize=10)
        if not group:
            axis.text(0.5, 0.5, f"no {label} runs", transform=axis.transAxes,
                      ha="center", va="center", fontsize=10, color="#666666")
    axes[-1].set_xlabel("East of the start point (m)")

    axes[0].legend(handles=[
        plt.Line2D([], [], color=plt.get_cmap(cmap)(0.6), linewidth=3.2, alpha=0.7,
                   label="vehicle trajectory"),
        # Drawn at a readable alpha: at the alpha used on the plot a single
        # legend marker would be invisible.
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, alpha=0.55,
                   color=style.pedestrian_color, label="pedestrian location"),
    ], frameon=False, fontsize=9, loc="lower right")

    # Name the phases where they happen, on the valid panel only: repeating them
    # below would label the same places twice.
    stops = [t["stop"] for t in groups[0][1] if t["stop"]]
    end_east, end_north = to_enu([site.end_latlon[0]], [site.end_latlon[1]],
                                 site.start_latlon)
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

    figure.suptitle(f"{title} — vehicle trajectories and pedestrian positions "
                    f"({len(tracks)} runs)", x=0.01, ha="left", fontsize=11)
    figure.savefig(output_path, dpi=style.figure_dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def stop_index(track, config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD) -> Optional[int]:
    """Index of the slowest sample in the first genuine halt, or None.

    The halt itself comes from ``halt_groups``, the same rule the scoring uses,
    so the plot and the table can never disagree about what a stop is. Within
    that halt the *slowest* sample is the anchor, not the first one to cross the
    threshold: anchoring on the crossing leaves each run aligned a fraction of a
    metre before it actually came to rest, and the averaged curve then bottoms
    out near 0.7 m/s instead of at zero.
    """
    travelled = np.hypot(track["east"] - track["east"][0],
                         track["north"] - track["north"][0])
    groups = halt_groups(track["times"], track["speed"], travelled, config)
    if not groups:
        return None
    first = groups[0]
    return int(first[int(np.argmin(track["speed"][first]))])


def distance_along(track, anchor: int) -> np.ndarray:
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


def plot_speed_profile(results: List[RunYield], output_path, title: str,
                       config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD,
                       style: cfg.PlotStyle = cfg.STYLE):
    """Speed against distance travelled: every run, plus the mean and spread.

    The trajectory plot shows *where* the vehicle slowed; this shows *how much*
    and over what distance, which is what says whether the yield was a
    controlled deceleration or a late stop.
    """
    # Align every run on its own stop, so distance 0 is where the vehicle halted
    # and the averaged curve genuinely reaches zero there. Anchoring at the trial
    # start instead smears the trough, because the runs stop at slightly
    # different distances and the mean never touches 0.
    with_track = [item for item in results if item.track]
    anchors = [(item, stop_index(item.track, config)) for item in with_track]
    skipped = [item.run for item, index in anchors if index is None]
    tracks = [(item.track, index) for item, index in anchors if index is not None]
    if not tracks:
        return None

    figure, axis = plt.subplots(figsize=(11, 4.8))

    profiles, crossings = [], []
    for track, stop in tracks:
        distance = distance_along(track, stop)
        profiles.append((distance, track["speed"]))
        axis.plot(distance, track["speed"], color=style.individual_color,
                  linewidth=0.6, alpha=0.55, zorder=2)
        # Where this run's pedestrian sat, in the same distance coordinate.
        if len(track["ped_east"]):
            nearest = int(np.argmin(np.hypot(
                track["east"] - np.median(track["ped_east"]),
                track["north"] - np.median(track["ped_north"]))))
            crossings.append(float(distance[nearest]))

    # Average on a common grid. Each run contributes only over the distance it
    # actually covered, so the mean is not dragged down by runs that had already
    # finished; nanmean over the padding does that for free.
    step_m = config.profile_step_m
    lowest = max(float(d.min()) for d, _ in profiles)
    highest = min(float(d.max()) for d, _ in profiles)
    # Grid on exact multiples of the step, so x = 0 is always a sample point.
    # Starting the grid at the data's minimum instead leaves 0 between two
    # samples, and the averaged curve then bottoms out either side of the stop
    # at ~0.7 m/s even though every run is anchored at exactly 0.
    grid = np.arange(np.ceil(lowest / step_m),
                     np.floor(highest / step_m) + 1) * step_m
    stacked = np.vstack([np.interp(grid, d, v, left=np.nan, right=np.nan)
                         for d, v in profiles])
    mean = np.nanmean(stacked, axis=0)
    deviation = np.nanstd(stacked, axis=0)

    axis.fill_between(grid, mean - deviation, mean + deviation,
                      color=style.mean_color, alpha=0.22, linewidth=0, zorder=3,
                      label="mean ± 1 s.d.")
    axis.plot(grid, mean, color=style.mean_color, linewidth=2.0, zorder=4,
              label="mean speed")
    axis.plot([], [], color=style.individual_color, linewidth=0.8, alpha=0.8,
              label=f"individual runs ({len(tracks)})")
    if skipped:
        # Named rather than silently dropped: a run with no stop has no anchor,
        # and it is also the one interesting failure in the set.
        axis.plot([], [], linestyle="",
                  label=f"excluded, never stopped: {', '.join(sorted(skipped))}")

    if crossings:
        crossing = float(np.median(crossings))
        axis.axvline(crossing, color="#444444", linestyle="--", linewidth=1.0, zorder=1)
        axis.axvline(0.0, color=style.mean_color, linestyle=":", linewidth=1.0, zorder=1)
        axis.annotate("pedestrian", xy=(crossing, axis.get_ylim()[1]),
                      xytext=(-6, -10), textcoords="offset points",
                      fontsize=9, color="#444444", ha="right", va="top")

    axis.set_xlabel("Distance travelled, relative to where the vehicle stopped (m)")
    axis.set_ylabel("Vehicle speed (m/s)")
    axis.set_title(f"{title} — speed profile along the trial", loc="left", fontsize=11)
    axis.set_xlim(grid.min(), grid.max())
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, fontsize=9, loc="lower left")
    axis.grid(alpha=0.2, linewidth=style.grid_linewidth)
    axis.set_axisbelow(True)
    cfg.despine(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=style.figure_dpi)
    plt.close(figure)
    return output_path


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_outputs(results: List[RunYield], summary: Dict, output_dir: Path,
                  case: cfg.TestCase,
                  config: cfg.VehicleYieldConfig = cfg.VEHICLE_YIELD,
                  site: cfg.SiteGeometry = cfg.SITE,
                  style: cfg.PlotStyle = cfg.STYLE,
                  speed_cmap=None, write_tables: bool = True) -> None:
    """The JSON, the per-run CSV, the three figures and the console rollup.

    ``case`` supplies the file stems and the headings. ``write_tables`` is
    False when redrawing from the cache: the tables already on disk are what
    produced the figures, and rewriting them from a reloaded summary would
    round the numbers a second time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if write_tables:
        (output_dir / f"{case.prefix}.json").write_text(
            json.dumps(summary, indent=2, default=float))
        rows = summary["runs_detail"]
        with open(output_dir / f"{case.prefix}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    plot_summary(results, summary, output_dir / f"{case.prefix}.png",
                 case.title, config)
    plot_trajectories(results, output_dir / config.trajectory_plot,
                      case.metric, speed_cmap, site, style)
    plot_speed_profile(results, output_dir / config.speed_profile_plot,
                       case.metric, config, style)

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
    print(f"{case.metric}: {summary['headline']} — {verdict} "
          f"(target {summary['target_success_pct']:g}%)")
    print(f"  -> {output_dir}")
