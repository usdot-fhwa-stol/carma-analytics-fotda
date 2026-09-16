"""CP-01: FLIR camera detection drop rate, from the raw Kafka detection topic.

Measures how many of the camera frames that *should* have been produced during
each pedestrian dwell actually reached the ``v2xhub_sim_sensor_detected_object``
topic. At a nominal 10 Hz, a 20-second dwell should yield 200 frames.

The test plan spans two recording sessions:

    5 runs x  5 s  +  5 runs x 10 s  +  5 runs x 15 s   (2026-09-14)
    15 runs x 20 s                                       (2026-09-15)
    = 30 runs, 450 s, 4500 expected frames

so the headline result is pooled across sessions, reported as
"X frames out of 4500 frames dropped".

**Why the Kafka topic and not a later stage.** It sits upstream of the
sensor_data_sharing_service, so the SDSS's staleness rejection cannot remove
anything from it. A camera websocket stall delays frames without dropping them:
during the 2026-09-15 stalls the frames still arrive, just late and in a burst,
and one stall window shows zero gaps over 150 ms. So what this metric counts is
frames the camera never produced (or that were lost before Kafka), which is what
CP-01 is asking about -- not frames rejected downstream for being late.

**Anchoring the dwell window.** The test plan says the entry time need not be
recorded: measure the dwell from the first detection. Taken literally that
misreads any run with a false start. 2026-09-15 run 1 opens with an 18-frame
burst, a 2.9 s pause, a 34-frame burst, another pause, and only then the real
19.7 s dwell -- anchoring on the first frame there measures 52 of 200 and reports
148 phantom drops. Anchoring on the longest *burst* instead fails the opposite
way: a genuine mid-dwell gap splits the dwell in two and only the larger half is
measured.

So the anchor is the burst start whose dwell-length window contains the most
frames. That ignores false starts and still counts gaps inside the dwell, which
are exactly the drops being measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from portable import kafka_log

# Nominal FLIR frame rate. 100 ms cadence, confirmed against the websocket logs:
# the median inter-arrival is 100.0 ms and ~90% of intervals fall in 90-110 ms.
DETECTION_RATE_HZ = 10.0

# Detections further apart than this start a new burst. Well above the 100 ms
# cadence (and its jitter) and well below the pauses between runs.
BURST_GAP_MS = 500.0

# How far around the nominal start time to look for a run's detections. Runs are
# at least four minutes apart, so this cannot reach a neighbouring run.
SEARCH_BEFORE_MS = 60_000.0
SEARCH_AFTER_MS = 240_000.0


@dataclass
class RunDrops:
    """One run's frame accounting."""

    run: str
    condition: str
    dwell_sec: int
    expected_frames: int
    received_frames: int
    first_detection_ms: Optional[float] = None
    bursts_in_window: int = 0
    gap_spans_ms: List[float] = field(default_factory=list)

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
            "expected_frames": self.expected_frames,
            "received_frames": self.received_frames,
            "dropped_frames": self.dropped_frames,
            "drop_pct": round(self.drop_pct, 3),
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


def _burst_starts(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return frames
    breaks = np.flatnonzero(np.diff(frames) > BURST_GAP_MS)
    return np.r_[frames[0], frames[breaks + 1]]


def measure_run(run, frames: np.ndarray, rate_hz: float = DETECTION_RATE_HZ) -> RunDrops:
    """Frame accounting for one run's dwell window."""
    expected = round(run.dwell_sec * rate_hz)
    start = run.start_time.timestamp() * 1000.0
    window = frames[(frames >= start - SEARCH_BEFORE_MS) & (frames <= start + SEARCH_AFTER_MS)]

    result = RunDrops(
        run=run.name, condition=run.condition, dwell_sec=run.dwell_sec,
        expected_frames=expected, received_frames=0,
    )
    if len(window) == 0:
        return result

    span_ms = run.dwell_sec * 1000.0
    starts = _burst_starts(window)
    result.bursts_in_window = len(starts)

    counts = [int(((window >= anchor) & (window < anchor + span_ms)).sum()) for anchor in starts]
    best = int(np.argmax(counts))
    anchor = float(starts[best])

    dwell_frames = window[(window >= anchor) & (window < anchor + span_ms)]
    result.received_frames = len(dwell_frames)
    result.first_detection_ms = anchor
    if len(dwell_frames) > 1:
        gaps = np.diff(dwell_frames)
        result.gap_spans_ms = [float(gap) for gap in gaps if gap > BURST_GAP_MS]
    return result


def analyse_session(runs, detection_log_path, rate_hz: float = DETECTION_RATE_HZ) -> List[RunDrops]:
    """CP-01 for every run of one session."""
    frames = camera_frame_times(kafka_log.parse_kafka_log_records(detection_log_path))
    return [measure_run(run, frames, rate_hz) for run in runs]


def summarise(results: List[RunDrops], rate_hz: float = DETECTION_RATE_HZ) -> Dict:
    """Pooled totals plus a per-condition breakdown."""
    expected = sum(item.expected_frames for item in results)
    received = sum(item.received_frames for item in results)
    dropped = sum(item.dropped_frames for item in results)

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
        "total_dwell_sec": sum(item.dwell_sec for item in results),
        "total_expected_frames": expected,
        "total_received_frames": received,
        "total_dropped_frames": dropped,
        "total_drop_pct": round(dropped / expected * 100.0, 3) if expected else 0.0,
        "headline": f"{dropped} frames out of {expected} frames dropped",
        "detection_rate_hz": rate_hz,
        "by_condition": dict(sorted(by_condition.items(), key=lambda kv: kv[1]["runs"])),
        "runs_detail": [item.as_row() for item in results],
    }
