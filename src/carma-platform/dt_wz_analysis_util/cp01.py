"""CP-01: FLIR camera detection drop rate, measured at the camera's websocket.

Measures how many of the camera frames that *should* have arrived during each
pedestrian dwell actually did. At a nominal 10 Hz, a 20-second dwell should
yield 200 frames.

**Measured at the websocket, not at Kafka.** CP-01 asks for a property of the
camera, so it has to be measured as close to the camera as the logs allow. The
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
staleness rejections that fail CP-02 are likewise invisible here, since they
happen well after the websocket.

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

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from .portable import flir_websocket, kafka_log

# Nominal FLIR frame rate. 100 ms cadence, confirmed against the websocket logs:
# the median inter-arrival is 100.0 ms and ~90% of intervals fall in 90-110 ms.
DETECTION_RATE_HZ = 10.0

# Detections further apart than this start a new burst. Chosen from the data:
# the largest gap observed *inside* a dwell is 1.5 s, and the smallest pause
# separating a false start from the real dwell is 2.9 s. 2 s sits between them,
# so a dwell containing dropped frames stays one burst while a false start stays
# separate. A threshold below ~1.5 s would split a dwell at its own drops, which
# is the very thing being measured.
BURST_GAP_MS = 2000.0

# One frame period at the nominal rate.
FRAME_INTERVAL_MS = 1000.0 / 10.0

# runs.csv wall clock, used only to render burst times for a human to check.
SESSION_TZ = timezone(timedelta(hours=-4))

# How far around the nominal start time to look for a run's detections. Runs are
# at least four minutes apart, so this cannot reach a neighbouring run.
SEARCH_BEFORE_MS = 60_000.0
SEARCH_AFTER_MS = 240_000.0


def _iso(epoch_ms: Optional[float]) -> Optional[str]:
    """Local (America/New_York) wall clock, so a burst can be checked by eye."""
    if epoch_ms is None:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000.0, SESSION_TZ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


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
        }


def camera_frame_times(detection_records) -> np.ndarray:
    """Sorted, de-duplicated camera frame timestamps (epoch ms).

    De-duplicated because a frame carrying two tracked objects produces two
    detection records sharing one camera timestamp; CP-01 counts camera frames,
    not objects, so those must collapse to one.
    """
    return np.array(
        sorted({float(record["timestamp"]) for record in detection_records if "timestamp" in record}),
        dtype=float,
    )


def _bursts(frames: np.ndarray):
    """Split frames into continuous bursts on gaps longer than BURST_GAP_MS."""
    if len(frames) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) > BURST_GAP_MS)
    return np.split(frames, breaks + 1)


def measure_run(run, frames: np.ndarray, rate_hz: float = DETECTION_RATE_HZ,
                websocket_frames=None, kafka_frames=None) -> RunDrops:
    """Frame accounting for one run's dwell window.

    ``frames`` is the sorted camera-time array the count is taken from -- the
    websocket times for CP-01 proper. ``websocket_frames`` are the full websocket
    records, used for the ``dataNumber`` gap count, and ``kafka_frames`` is the
    Kafka camera-time array, used to report the plugin-side loss separately.
    """
    expected = round(run.dwell_sec * rate_hz)
    start = run.start_time.timestamp() * 1000.0
    window = frames[(frames >= start - SEARCH_BEFORE_MS) & (frames <= start + SEARCH_AFTER_MS)]

    result = RunDrops(
        run=run.name, condition=run.condition, dwell_sec=run.dwell_sec,
        expected_frames=expected, received_frames=0,
    )
    if len(window) == 0:
        return result

    # Split into bursts, then take the burst whose duration is closest to this
    # run's condition. The pedestrian's real dwell is never exactly the nominal
    # time, so the burst itself defines the measurement window.
    bursts = _bursts(window)
    result.bursts_in_window = len(bursts)
    nominal_ms = run.dwell_sec * 1000.0
    best = min(bursts, key=lambda burst: abs((burst[-1] - burst[0]) - nominal_ms))

    dwell_frames = best
    span_ms = float(dwell_frames[-1] - dwell_frames[0])
    # Frames a continuous stream would hold over this same span. Fence-post: a
    # 1.0 s span at 10 Hz spans 11 frames, not 10, because both ends are frames.
    expected = int(round(span_ms / FRAME_INTERVAL_MS)) + 1 if len(dwell_frames) > 1 else 1

    result.expected_frames = expected
    result.received_frames = len(dwell_frames)
    result.first_detection_ms = float(dwell_frames[0])
    result.last_detection_ms = float(dwell_frames[-1])
    result.burst_duration_sec = round(span_ms / 1000.0, 3)
    if len(dwell_frames) > 1:
        gaps = np.diff(dwell_frames)
        result.gap_spans_ms = [float(gap) for gap in gaps if gap > FRAME_INTERVAL_MS * 1.5]

    # The burst is inclusive of its last frame, so the window closes just after it.
    low, high = float(dwell_frames[0]), float(dwell_frames[-1]) + 1.0
    if websocket_frames is not None:
        result.counter_gaps = flir_websocket.counter_gaps(
            flir_websocket.frames_in_window(websocket_frames, low, high)
        )
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


def analyse_session(runs, session, rate_hz: float = DETECTION_RATE_HZ) -> List[RunDrops]:
    """CP-01 for every run of one session.

    ``session`` is a ``dt_wz_dataset.SessionPaths``. The count comes from the pc2
    V2XHub log's websocket messages; the Kafka detection topic is read only to
    report the plugin-side loss alongside it.
    """
    if session.pc2_v2xhub is None:
        raise FileNotFoundError(
            "CP-01 needs the pc2 V2XHub log: it carries the camera's websocket stream"
        )
    websocket_frames = flir_websocket.parse_camera_frames(session.pc2_v2xhub)
    frames = np.array([frame["camera_time_ms"] for frame in websocket_frames], dtype=float)

    kafka_frames = None
    if session.kafka_detected_object is not None:
        kafka_frames = camera_frame_times(
            kafka_log.parse_kafka_log_records(session.kafka_detected_object)
        )

    return [
        measure_run(run, frames, rate_hz,
                    websocket_frames=websocket_frames, kafka_frames=kafka_frames)
        for run in runs
    ]


def summarise(results: List[RunDrops], rate_hz: float = DETECTION_RATE_HZ) -> Dict:
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
        "detection_rate_hz": rate_hz,
        "measured_at": "camera websocket (pc2 V2XHub log)",
        "camera_counter_gaps": counter_gaps,
        "lost_after_camera": lost_after,
        "by_condition": dict(sorted(by_condition.items(), key=lambda kv: kv[1]["runs"])),
        "runs_detail": [item.as_row() for item in results],
    }
