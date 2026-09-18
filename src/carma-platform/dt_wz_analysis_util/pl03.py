"""PL-03: does the vehicle yield to the pedestrian it was warned about?

Every other DT-WZ metric measures whether the SDSM arrived and how quickly. This
one measures what CARMA Platform did with it: across 30 runs, at least 90% of the
**valid** ones must end with the vehicle stopping short of the pedestrian.

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

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import metrics
from .portable import mcap_backend

# The geodesy lives in the carma-streets tree; cs01 already imports from there.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "carma-streets"))
from sdsm_location_spoofing_verification import geodetic_to_enu  # noqa: E402

GPS_TOPIC = "/hardware_interface/novatel/oem7/fix"
TWIST_TOPIC = "/hardware_interface/vehicle/twist"

# The trial's fixed endpoints, used to confirm the vehicle drove the intended
# route. They are a sanity check on the run, not the trial window -- the engaged
# interval is that.
START_LATLON = (38.95503211512239, -77.14757752506146)
END_LATLON = (38.95497471661251, -77.1491050181905)

# Checked inside the engaged window, and looser than the nominal 5 m because the
# vehicle can creep forward before guidance engages: the measured start gap
# reaches 8.8 m.
START_TOLERANCE_M = 10.0

# Checked over the **whole recording**, not the engaged window. CARMA routinely
# disengages once it is past the pedestrian, leaving up to 37 m of the route
# undriven under guidance, yet every run reaches the end point within 0.6 m
# afterwards. Gating on the engaged window would fail 19 of 30 runs for behaving
# normally.
END_TOLERANCE_M = 5.0

# A stop is speed below this for at least this long.
STOP_SPEED_MPS = 0.2
MIN_STOP_SEC = 0.5

# How far the vehicle must have travelled before a low-speed period counts as a
# stop rather than the stationary start it always begins from.
MOVED_FROM_START_M = 5.0

# How far past the pedestrian's crossing point a stop may sit and still count as
# a yield. The crossing point is a median of noisy SDSM reports and GPS has its
# own error, so a metre or two past it is measurement noise, not an overshoot.
YIELD_TOLERANCE_M = 5.0

# Beyond this the vehicle stopped for something other than the pedestrian.
YIELD_MAX_DISTANCE_M = 30.0

# Validity, criterion [1]: a detection burst containing a gap this long means the
# camera stalled and the run cannot test vehicle behaviour.
DETECTION_GAP_LIMIT_SEC = 1.0

# Validity, criterion [1] continued: frames lost between the camera and Kafka.
# Proportional rather than any-loss, on the same 2% bar as the SDSM criterion. A
# single frame missing from a ~200-frame run is not "inconsistent detection";
# the plugin faults worth excluding lost 5-6% of the run.
PLUGIN_LOSS_LIMIT_PCT = 2.0

# Validity, criterion [3]: SDSMs must have been consistently received.
SDSM_DROP_LIMIT_PCT = 2.0

# Samples used to estimate the vehicle's heading at a stop. At ~50 Hz this looks
# back about 0.4 s, long enough to outrun GPS jitter and short enough to reflect
# the direction the vehicle was actually travelling.
HEADING_LOOKBACK = 20


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


def _round(value, places):
    return None if value is None else round(float(value), places)


def _enu(latitudes, longitudes) -> Tuple[np.ndarray, np.ndarray]:
    """Local east/north metres about the trial's start point."""
    east, north = geodetic_to_enu(
        np.asarray(latitudes, dtype=float), np.asarray(longitudes, dtype=float), *START_LATLON
    )
    return np.atleast_1d(east), np.atleast_1d(north)


def _read_topic(mcap_path, topic, extract, window=None):
    """(times, values) for one topic, windowed only if a window is given."""
    counts = mcap_backend.topic_message_counts(mcap_path)
    if not counts.get(topic):
        return np.array([]), []
    reader, _type_map, _start = mcap_backend.open_bagfile(str(mcap_path), topics=[topic])
    times, values = [], []
    while reader.has_next():
        _topic, message, log_time_ns = reader.read_next()
        seconds = log_time_ns / 1e9
        if window is None or window[0] <= seconds <= window[1]:
            times.append(seconds)
            values.append(extract(message.to_dict()))
    return np.array(times, dtype=float), values


def find_stops(times, speeds, distance_from_start) -> List[Tuple[float, float]]:
    """(start_time, duration) for each halt after the vehicle has set off.

    The stationary start is excluded by distance travelled rather than by
    ignoring the first stop: a run can legitimately halt twice, and a run can
    idle for a variable time before moving.
    """
    if len(times) == 0:
        return []
    moving_area = (speeds < STOP_SPEED_MPS) & (distance_from_start > MOVED_FROM_START_M)
    if not moving_area.any():
        return []
    indices = np.flatnonzero(moving_area)
    stops = []
    for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        duration = times[group[-1]] - times[group[0]]
        if duration >= MIN_STOP_SEC:
            stops.append((float(times[group[0]]), float(duration)))
    return stops


def score_stop(stop_time, vehicle_times, east, north, pedestrian_east, pedestrian_north):
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
    earlier = max(index - HEADING_LOOKBACK, 0)
    heading = np.array([east[index] - east[earlier], north[index] - north[earlier]])
    magnitude = float(np.hypot(*heading))
    if magnitude == 0:
        return distance, distance  # no heading available; treat as straight ahead
    ahead = float(np.dot([pedestrian_east - x, pedestrian_north - y], heading) / magnitude)
    return distance, ahead


def measure_run(run, window) -> RunYield:
    """Score one run's yield behaviour over its engaged window."""
    result = RunYield(run=run.name, condition=run.condition, dwell_sec=run.dwell_sec)

    # Read the whole GPS track once: the trial is scored inside the engaged
    # window, but the end fence has to be checked against the full recording.
    all_t, all_fixes = _read_topic(run.mcap, GPS_TOPIC, lambda d: (d["latitude"], d["longitude"]))
    if len(all_t) < 10:
        result.note = "no vehicle GPS in this recording"
        return result
    all_east, all_north = _enu([f[0] for f in all_fixes], [f[1] for f in all_fixes])
    in_window = (all_t >= window[0]) & (all_t <= window[1])
    if in_window.sum() < 10:
        result.note = "no vehicle GPS in the engaged window"
        return result
    fixes_t = all_t[in_window]
    east, north = all_east[in_window], all_north[in_window]

    speeds_t, speeds = _read_topic(
        run.mcap, TWIST_TOPIC, lambda d: d["twist"]["linear"]["x"], window)
    if len(speeds_t) < 10:
        result.note = "no vehicle speed in the engaged window"
        return result
    speeds = np.asarray(speeds, dtype=float)

    positions = metrics.sdsm_object_positions(run.mcap, window)
    if not positions:
        result.note = "no SDSM object positions in the engaged window"
        return result
    ped_east, ped_north = _enu(
        [p["latitude"] for p in positions], [p["longitude"] for p in positions])
    crossing_east = float(np.median(ped_east))
    crossing_north = float(np.median(ped_north))

    # Confirm the vehicle drove the intended route. The start is checked inside
    # the engaged window; the end over the whole recording, since guidance
    # normally hands back before the vehicle gets there.
    end_east, end_north = _enu([END_LATLON[0]], [END_LATLON[1]])
    start_gap = float(np.hypot(east, north).min())
    end_gap = float(np.hypot(all_east - end_east[0], all_north - end_north[0]).min())
    result.start_gap_m = start_gap
    result.end_gap_m = end_gap
    if start_gap > START_TOLERANCE_M:
        result.note = f"never within {START_TOLERANCE_M:g} m of the start point"
        return result
    if end_gap > END_TOLERANCE_M:
        result.note = f"never within {END_TOLERANCE_M:g} m of the end point"
        return result

    result.scored = True
    result.trial_duration_sec = float(fixes_t[-1] - fixes_t[0])
    result.peak_speed_mps = float(speeds.max())

    travelled = np.interp(speeds_t, fixes_t, np.hypot(east - east[0], north - north[0]))
    stops = find_stops(speeds_t, speeds, travelled)
    result.stop_count = len(stops)

    # Of the qualifying stops, the one nearest the pedestrian is the yield
    # candidate; a vehicle may also halt for unrelated reasons.
    best = None
    for stop_time, duration in stops:
        distance, ahead = score_stop(
            stop_time, fixes_t, east, north, crossing_east, crossing_north)
        if best is None or distance < best[2]:
            best = (stop_time, duration, distance, ahead)

    if best is not None:
        stop_time, duration, distance, ahead = best
        result.stop_time_sec = stop_time - window[0]
        result.stop_duration_sec = duration
        result.distance_to_pedestrian_m = distance
        result.pedestrian_ahead_m = ahead
        result.yielded = bool(ahead > -YIELD_TOLERANCE_M and distance <= YIELD_MAX_DISTANCE_M)

        # Where the pedestrian was when the vehicle pulled away again.
        resumed = speeds_t[(speeds_t > stop_time + duration) & (speeds > STOP_SPEED_MPS)]
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


def apply_validity(results: List[RunYield], drops, sdsm_drop_pct: Dict[str, float]) -> None:
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
            if gap >= DETECTION_GAP_LIMIT_SEC:
                reasons.append(f"camera detection gap of {gap:.1f} s")
            received = detection.received_frames or 0
            lost = detection.lost_after_camera or 0
            lost_pct = (lost / received * 100.0) if received else 0.0
            if lost_pct > PLUGIN_LOSS_LIMIT_PCT:
                reasons.append(
                    f"{lost} of {received} frames ({lost_pct:.1f}%) lost inside the "
                    f"V2XHub plugin")
        drop = sdsm_drop_pct.get(result.run)
        if drop is not None and drop > SDSM_DROP_LIMIT_PCT:
            reasons.append(f"SDSM drop rate {drop:.2f}% over {SDSM_DROP_LIMIT_PCT:g}%")
        if not result.scored and result.note:
            reasons.append(result.note)
        if reasons:
            result.valid = False
            result.invalid_reason = "; ".join(reasons)


def summarise(results: List[RunYield], target_pct: float = 90.0) -> Dict:
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
        "metric": "PL-03",
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
        "yield_tolerance_m": YIELD_TOLERANCE_M,
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
