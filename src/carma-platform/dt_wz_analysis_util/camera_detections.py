"""FLIR camera detection drop rate, measured at the camera's websocket.

Measures how many of the camera frames that *should* have arrived during each
pedestrian dwell actually did. At a nominal 10 Hz, a 20-second dwell should
yield 200 frames.

**Measured at the websocket, not at Kafka.** This is a property of the camera,
so it has to be measured as close to the camera as the logs allow. The
pc2 V2XHub log records every websocket message verbatim, before the plugin
parses, queues or forwards anything.

The difference is not academic. Comparing the two sources across these 30 runs,
24 frames arrived intact over the websocket and never reached Kafka -- and their
``dataNumber`` values were contiguous, so the camera had sent them and the
plugin had received them. Counting at Kafka charges those 24 frames to the
camera. They belong to the plugin.

The Kafka topic is still read for comparison, and the difference between the two
counts is reported per run as ``lost_after_camera``, which is the plugin-side
loss stated separately instead of hidden inside the camera's figure.

The test plan spans two recording sessions -- 5/10/15 s dwells on 2026-09-14 and
20 s dwells on 2026-09-15, 30 runs in total -- and the result is pooled across
both.

**The denominator is the measured burst, not the nominal dwell.** The plan's
nominal 4500 frames assume the pedestrian stood in the zone for exactly the
labelled time. They did not: the 15 s runs average about 13.9 s, and one lasts
12.71 s. Counting against 150 frames there reports 22 drops for a stream that is
continuous at 10.07 Hz with no gap over 150 ms -- it measures the pedestrian's
timing, not the camera.

So each run is measured against the frames a continuous stream would hold over
that run's own burst span. What remains is the camera's own reliability.

**A stall is not a drop.** A camera websocket stall delays frames without losing
them: during the 2026-09-15 stalls every frame still arrived, just late and in a
burst, and one stall window shows no gap over 150 ms at all. Because the count
is keyed on the camera's own capture time, a late frame still lands in the dwell
window it belongs to, so a stall does not inflate this figure. The downstream
staleness rejections that fail the detection-delivery check are likewise
invisible here, since they happen well after the websocket.

**Two numbers come with the count.** ``counter_gaps`` is how many camera frames
the ``dataNumber`` sequence accounts for that carried no message -- frames in
which the camera detected nobody while the pedestrian was present.
``lost_after_camera`` is how many arrived over the websocket but never reached
Kafka. Neither is part of the drop count; they say where the remaining losses
sit.

**Finding the dwell.** The entry time was not recorded, so the dwell is found in
the data. Frames are split into bursts, and the burst whose duration is closest
to the run's condition is taken as the dwell. That rejects false starts:
2026-09-15 run 1 opens with a 1.7 s burst and a 3.3 s burst before the real
19.7 s dwell, and only the last is close to 20 s.

Each run's chosen burst is reported with its start time, end time and duration,
so the window the number came from can be checked against the recording.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as cfg  # noqa: E402
from . import dataset  # noqa: E402
from . import report  # noqa: E402
from .cascade.plots import stage_colors  # noqa: E402
from .readers import flir_websocket, kafka_log  # noqa: E402


def _iso(epoch_ms: Optional[float]) -> Optional[str]:
    """Local (America/New_York) wall clock, so a burst can be checked by eye."""
    if epoch_ms is None:
        return None
    return datetime.fromtimestamp(
        epoch_ms / 1000.0, cfg.SESSION_TZ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


@dataclass
class RunDrops:
    """One run's frame accounting."""

    run: str
    condition: str
    dwell_sec: int
    expected_frames: int
    received_frames: int
    first_detection_ms: Optional[float] = None
    last_detection_ms: Optional[float] = None
    burst_duration_sec: float = 0.0
    bursts_in_window: int = 0
    gap_spans_ms: List[float] = field(default_factory=list)
    # Camera frames the counter accounts for but that carried no message: frames
    # in which the camera detected nobody while the pedestrian was present.
    counter_gaps: int = 0
    # Frames that arrived over the websocket but never reached Kafka. Not a
    # camera fault; reported so the plugin-side loss stays visible and separate.
    lost_after_camera: Optional[int] = None
    # Link stalls inside the dwell: frames that reached V2XHub late while the
    # camera kept producing. See ``flir_websocket.find_stalls``. A stall does
    # not change the drop count -- the frames still arrive -- but it is what
    # makes them stale by the time they reach the SDSM service.
    stalls: List[Dict] = field(default_factory=list)
    max_lag_ms: Optional[float] = None

    @property
    def stalled_frames(self) -> int:
        return sum(stall["frames"] for stall in self.stalls)

    @property
    def dropped_frames(self) -> int:
        """Frames missing from the dwell, never negative.

        A run can return one frame *more* than nominal when the camera runs a
        fraction fast over the window; that is not a negative drop, it is a
        cadence rounding artefact, so it floors at zero here and the raw counts
        stay visible in ``expected_frames``/``received_frames``.
        """
        return max(self.expected_frames - self.received_frames, 0)

    @property
    def drop_pct(self) -> float:
        return self.dropped_frames / self.expected_frames * 100.0 if self.expected_frames else 0.0

    def as_row(self) -> Dict:
        return {
            "run": self.run,
            "condition": self.condition,
            "dwell_sec": self.dwell_sec,
            "burst_duration_sec": self.burst_duration_sec,
            "burst_start": _iso(self.first_detection_ms),
            "burst_end": _iso(self.last_detection_ms),
            "burst_start_ms": self.first_detection_ms,
            "burst_end_ms": self.last_detection_ms,
            "expected_frames": self.expected_frames,
            "received_frames": self.received_frames,
            "dropped_frames": self.dropped_frames,
            "drop_pct": round(self.drop_pct, 3),
            "counter_gaps": self.counter_gaps,
            "lost_after_camera": self.lost_after_camera,
            "bursts_in_window": self.bursts_in_window,
            "largest_gap_ms": round(max(self.gap_spans_ms), 1) if self.gap_spans_ms else 0.0,
            "max_lag_ms": None if self.max_lag_ms is None else round(self.max_lag_ms, 1),
            "stall_count": len(self.stalls),
            "stalled_frames": self.stalled_frames,
            "stall_detail": "; ".join(describe_stall(stall) for stall in self.stalls),
        }


def describe_stall(stall: Dict) -> str:
    """One stall as a short line a reader can check against the log."""
    counter = ("counter contiguous" if stall["counter_missing"] == 0
               else f"{stall['counter_missing']} counter values missing")
    return (f"{stall['max_lag_ms'] / 1000:.1f} s lag at {_iso(stall['first_camera_ms'])}, "
            f"{stall['frames']} frames held, {counter}, "
            f"backlog arrived within {stall['arrival_span_ms'] / 1000:.2f} s")


def camera_frame_times(detection_records) -> np.ndarray:
    """Sorted, de-duplicated camera frame timestamps (epoch ms).

    De-duplicated because a frame carrying two tracked objects produces two
    detection records sharing one camera timestamp; this counts camera frames,
    not objects, so those must collapse to one.
    """
    return np.array(
        sorted({float(record["timestamp"]) for record in detection_records if "timestamp" in record}),
        dtype=float,
    )


def _bursts(frames: np.ndarray, burst_gap_ms: float):
    """Split frames into continuous bursts on gaps longer than ``burst_gap_ms``."""
    if len(frames) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) > burst_gap_ms)
    return np.split(frames, breaks + 1)


def measure_run(run, frames: np.ndarray, config: cfg.CameraDetectionConfig = cfg.CAMERA_DETECTIONS,
                websocket_frames=None, kafka_frames=None,
                search_end_ms: Optional[float] = None,
                claimed: Optional[set] = None,
                engaged_window=None) -> RunDrops:
    """Frame accounting for one run's dwell window.

    ``frames`` is the sorted camera-time array the count is taken from -- the
    websocket times. ``websocket_frames`` are the full websocket
    records, used for the ``dataNumber`` gap count, and ``kafka_frames`` is the
    Kafka camera-time array, used to report the plugin-side loss separately.

    ``config`` supplies the nominal rate, the burst split and the search span;
    see ``config.CameraDetectionConfig``.

    The search stops at ``run.end_time``, the next row's start in runs.csv,
    since a trial cannot run past the start of the next one. ``search_end_ms``
    overrides that. ``claimed`` holds the first-frame times of bursts already
    given to another run, so none is counted twice. Both matter because runs
    can be closer together than the search span: on 2026-09-23 they were 2-4
    minutes apart, and without the bound four runs took their neighbour's
    burst, which happened to be nearer the nominal dwell.

    ``engaged_window`` is the run's ``(start, end)`` under guidance, in epoch
    seconds, from ``/guidance/state``. When given, the dwell is the burst that
    overlaps it most -- the pedestrian the vehicle was actually driving at.
    Burst length alone is not a reliable guide: the window can also hold the
    pedestrian stepping in beforehand or walking back afterwards, and those
    can be nearer the nominal dwell than the real one (2026-09-14 5sec_run3:
    the 7.9 s dwell lost to a 3.6 s walk-back) or longer than it (31 s after
    10sec_run3 and 20sec_run5). Without an engaged window, or where no burst
    overlaps it, the nearest-to-nominal rule is the fallback.
    """
    frame_interval_ms = config.frame_interval_ms
    expected = round(run.dwell_sec * config.detection_rate_hz)
    start = run.start_time.timestamp() * 1000.0
    end = start + config.search_after_ms
    if search_end_ms is None and getattr(run, "end_time", None) is not None:
        search_end_ms = run.end_time.timestamp() * 1000.0
    if search_end_ms is not None:
        end = min(end, search_end_ms)
    window = frames[(frames >= start - config.search_before_ms) & (frames < end)]

    result = RunDrops(
        run=run.name, condition=run.condition, dwell_sec=run.dwell_sec,
        expected_frames=expected, received_frames=0,
    )
    if len(window) == 0:
        return result

    # Split into bursts, then take the one belonging to the trial: the burst
    # that overlaps the engaged window most, ties and the no-window case going
    # to the burst whose length is nearest the nominal dwell. The pedestrian's
    # real dwell is never exactly the nominal time, so the burst itself then
    # defines the measurement window.
    bursts = _bursts(window, config.burst_gap_ms)
    if claimed:
        bursts = [burst for burst in bursts if float(burst[0]) not in claimed]
    result.bursts_in_window = len(bursts)
    if not bursts:
        return result
    nominal_ms = run.dwell_sec * 1000.0

    def off_nominal(burst):
        return abs((burst[-1] - burst[0]) - nominal_ms)

    def overlap(burst):
        if engaged_window is None:
            return 0.0
        low, high = engaged_window[0] * 1000.0, engaged_window[1] * 1000.0
        return max(0.0, min(float(burst[-1]), high) - max(float(burst[0]), low))

    best = min(bursts, key=lambda burst: (-overlap(burst), off_nominal(burst)))
    if claimed is not None:
        claimed.add(float(best[0]))

    dwell_frames = best
    span_ms = float(dwell_frames[-1] - dwell_frames[0])
    # Frames a continuous stream would hold over this same span. Fence-post: a
    # 1.0 s span at 10 Hz spans 11 frames, not 10, because both ends are frames.
    expected = int(round(span_ms / frame_interval_ms)) + 1 if len(dwell_frames) > 1 else 1

    result.expected_frames = expected
    result.received_frames = len(dwell_frames)
    result.first_detection_ms = float(dwell_frames[0])
    result.last_detection_ms = float(dwell_frames[-1])
    result.burst_duration_sec = round(span_ms / 1000.0, 3)
    if len(dwell_frames) > 1:
        gaps = np.diff(dwell_frames)
        result.gap_spans_ms = [
            float(gap) for gap in gaps
            if gap > frame_interval_ms * config.gap_report_factor
        ]

    # The burst is inclusive of its last frame, so the window closes just after it.
    low, high = float(dwell_frames[0]), float(dwell_frames[-1]) + 1.0
    if websocket_frames is not None:
        in_dwell = flir_websocket.frames_in_window(websocket_frames, low, high)
        result.counter_gaps = flir_websocket.counter_gaps(in_dwell)
        lags = [f["arrival_ms"] - f["camera_time_ms"] for f in in_dwell
                if f.get("arrival_ms") is not None]
        result.max_lag_ms = max(lags) if lags else None
        result.stalls = flir_websocket.find_stalls(in_dwell, config.stall_lag_ms)
    if kafka_frames is not None:
        in_kafka = kafka_frames[(kafka_frames >= low) & (kafka_frames < high)]
        # Both sides carry the camera's own capture time, so they are directly
        # comparable; round to the millisecond the logs are written at.
        reached = set(np.round(in_kafka).astype(np.int64).tolist())
        result.lost_after_camera = sum(
            1 for value in np.round(dwell_frames).astype(np.int64).tolist()
            if value not in reached
        )
    return result


def analyse_session(runs, session,
                    config: cfg.CameraDetectionConfig = cfg.CAMERA_DETECTIONS) -> List[RunDrops]:
    """Frame accounting for every run of one session.

    ``session`` is a ``dataset.SessionPaths``. The count comes from the pc2
    V2XHub log's websocket messages; the Kafka detection topic is read only to
    report the plugin-side loss alongside it.
    """
    if session.pc2_v2xhub is None:
        raise FileNotFoundError(
            "the pc2 V2XHub log is required: it carries the camera's websocket stream"
        )
    websocket_frames = flir_websocket.parse_camera_frames(session.pc2_v2xhub)
    frames = np.array([frame["camera_time_ms"] for frame in websocket_frames], dtype=float)

    kafka_frames = None
    if session.kafka_detected_object is not None:
        kafka_frames = camera_frame_times(
            kafka_log.parse_kafka_log_records(session.kafka_detected_object)
        )

    # runs.csv order is chronological and each run is bounded by the next
    # one's start (RunSpec.end_time); no burst is given to two runs; and the
    # dwell is the burst overlapping the run's engaged window.
    claimed: set = set()
    results = []
    for run in runs:
        try:
            engaged = report.engaged_window(run)
        except Exception as error:  # unreadable bag: fall back to burst length
            print(f"  {run.name}: no engaged window ({error}); choosing by dwell length")
            engaged = None
        results.append(measure_run(
            run, frames, config,
            websocket_frames=websocket_frames, kafka_frames=kafka_frames,
            claimed=claimed, engaged_window=engaged))
    return results


def analyse_sessions(data_roots, config: cfg.CameraDetectionConfig = cfg.CAMERA_DETECTIONS,
                     layout: cfg.SessionLayout = cfg.LAYOUT) -> List[RunDrops]:
    """Frame accounting over every run of every session, pooled into one list."""
    results: List[RunDrops] = []
    for root in data_roots:
        runs, session = dataset.load_session(root, layout)
        if session.pc2_v2xhub is None:
            raise FileNotFoundError(
                f"{root}: no pc2 V2XHub log; the count comes from the camera's "
                f"websocket stream"
            )
        print(f"{root.name}: {len(runs)} runs, camera frames from "
              f"{session.pc2_v2xhub.name}")
        results.extend(analyse_session(runs, session, config))
    return results


def summarise(results: List[RunDrops], config: cfg.CameraDetectionConfig = cfg.CAMERA_DETECTIONS) -> Dict:
    """Pooled totals plus a per-condition breakdown."""
    expected = sum(item.expected_frames for item in results)
    received = sum(item.received_frames for item in results)
    dropped = sum(item.dropped_frames for item in results)

    counter_gaps = sum(item.counter_gaps for item in results)
    lost_after = sum(item.lost_after_camera or 0 for item in results)

    by_condition: Dict[str, Dict] = {}
    for item in results:
        bucket = by_condition.setdefault(
            item.condition,
            {"runs": 0, "expected_frames": 0, "received_frames": 0, "dropped_frames": 0},
        )
        bucket["runs"] += 1
        bucket["expected_frames"] += item.expected_frames
        bucket["received_frames"] += item.received_frames
        bucket["dropped_frames"] += item.dropped_frames
    for bucket in by_condition.values():
        bucket["drop_pct"] = round(
            bucket["dropped_frames"] / bucket["expected_frames"] * 100.0, 3
        ) if bucket["expected_frames"] else 0.0

    return {
        "runs": len(results),
        "total_nominal_dwell_sec": sum(item.dwell_sec for item in results),
        "total_measured_burst_sec": round(sum(item.burst_duration_sec for item in results), 3),
        "total_expected_frames": expected,
        "total_received_frames": received,
        "total_dropped_frames": dropped,
        "total_drop_pct": round(dropped / expected * 100.0, 3) if expected else 0.0,
        "headline": f"{dropped} frames out of {expected} frames dropped",
        "detection_rate_hz": config.detection_rate_hz,
        "measured_at": "camera websocket (pc2 V2XHub log)",
        "camera_counter_gaps": counter_gaps,
        "lost_after_camera": lost_after,
        "stall_lag_ms": config.stall_lag_ms,
        "runs_with_stalls": [item.run for item in results if item.stalls],
        "total_stalled_frames": sum(item.stalled_frames for item in results),
        "by_condition": dict(sorted(by_condition.items(), key=lambda kv: kv[1]["runs"])),
        "runs_detail": [item.as_row() for item in results],
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def plot_drops(results: List[RunDrops], summary: Dict, output_path, title: str):
    """Per-run drop counts, grouped by dwell condition."""
    if not results:
        return None
    conditions = list(summary["by_condition"])
    colors = dict(zip(conditions, stage_colors(range(max(len(conditions), 2)))))

    figure, axis = plt.subplots(figsize=(11, 4.6))
    values = [item.dropped_frames for item in results]
    axis.bar(
        range(len(results)), values,
        color=[colors[item.condition] for item in results],
        edgecolor="white", linewidth=0.8,
    )
    for index, item in enumerate(results):
        if item.dropped_frames:
            axis.text(index, item.dropped_frames, f"{item.dropped_frames}",
                      ha="center", va="bottom", fontsize=7,
                      color=cfg.STYLE.annotation_color)

    axis.set_xticks(range(len(results)))
    axis.set_xticklabels([item.run for item in results],
                         rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("Camera frames dropped")
    axis.set_title(
        f"{title} — {summary['headline']} ({summary['total_drop_pct']:.2f}%)",
        loc="left", fontsize=11,
    )
    axis.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="", markersize=7,
                   color=colors[condition],
                   label=f"{condition} ({summary['by_condition'][condition]['runs']} runs)")
        for condition in conditions
    ], frameon=False, fontsize=8)
    cfg.grid(axis)
    cfg.despine(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=cfg.STYLE.figure_dpi)
    plt.close(figure)
    return output_path


def write_outputs(results: List[RunDrops], summary: Dict, output_dir: Path,
                  case: cfg.TestCase) -> None:
    """The JSON, the per-run CSV and the plot, and the console rollup.

    ``case`` supplies the file stem and the headings. This module measures
    camera frames and does not know which test asked it to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{case.prefix}.json").write_text(json.dumps(summary, indent=2))

    rows = summary["runs_detail"]
    with open(output_dir / f"{case.prefix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_drops(results, summary, output_dir / f"{case.prefix}.png", case.title)

    print()
    for condition, bucket in summary["by_condition"].items():
        print(f"  {condition:<8} {bucket['runs']:>2} runs  "
              f"{bucket['dropped_frames']:>4}/{bucket['expected_frames']:<5} dropped  "
              f"({bucket['drop_pct']:.2f}%)")
    print()
    print(f"{case.metric}: {summary['headline']} "
          f"({summary['total_drop_pct']:.2f}%)")
    print(f"       measured at the {summary['measured_at']}")
    print(f"       camera frames with no detection (counter gaps): "
          f"{summary['camera_counter_gaps']}")
    print(f"       additionally lost after the camera, inside the plugin: "
          f"{summary['lost_after_camera']}")
    print(f"       link stalls (lag > {summary['stall_lag_ms']:g} ms): "
          f"{len(summary['runs_with_stalls'])} runs, "
          f"{summary['total_stalled_frames']} frames held back")
    print(f"       across {summary['runs']} runs / "
          f"{summary['total_measured_burst_sec']} s of measured dwell")
    print(f"  -> {output_dir}")
