"""CS-01: verify that SDSMs place the spoofed pedestrian at the configured reference.

FLIRCameraDriver discards the camera's true location. It reports each detection
as a cartesian offset from a configured remote reference point, and it writes
that reference into the ``lat_0``/``lon_0`` of the detection's projection string.
CS-01 checks that each SDSM puts the object at the reference plus the same
offset, and that the reported heading agrees with the detection velocity.

The geometry and the pass criteria come from
``src/carma-streets/sdsm_location_spoofing_verification.py``, which is used
unchanged. This module adds the two things that file cannot do on its own:

**It finds the reference in the data.** The reference is not a constant. The
detection logs hold three distinct projection origins, about one metre apart,
because the driver was reconfigured on 2026-09-09. Hard-coding one of them makes
the metric silently verify the wrong session, so the reference is read from the
detections that fall inside the session's own runs.

**It windows the logs to the session.** A Kafka dump holds the broker's whole
retention. Unwindowed, CS-01 verifies every detection back to 2026-09-09 and
reports a figure for several days of testing rather than for the session asked
about.
"""

from __future__ import annotations

import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from portable import kafka_log

# carma-streets is a hyphenated directory, so it is not importable by name
sys.path.append(str(Path(__file__).resolve().parent.parent / "carma-streets"))
from sdsm_location_spoofing_verification import (  # noqa: E402
    DEFAULT_MAX_MEAN_HEADING_ERROR_DEG,
    DEFAULT_MAX_MEAN_POSITION_ERROR_M,
    parse_projection_origin,
    verify_location_spoofing,
)

# Kafka dumps are windowed on the broker's CreateTime. That sits within a few
# tens of milliseconds of the message it carries, which is far tighter than the
# minutes of margin used below, and it is the one field both the detection and
# the SDSM dumps share in the same format.
_CREATE_TIME = re.compile(r"CreateTime:\s*(\d{10,13})")

# Margin added around the session's runs before windowing.
WINDOW_MARGIN_MS = 120_000.0


def session_window_ms(runs) -> Tuple[float, float]:
    """Epoch-millisecond span that covers every run, plus a margin."""
    starts = [run.start_time.timestamp() * 1000.0 for run in runs]
    ends = [
        run.start_time.timestamp() * 1000.0 + run.dwell_sec * 1000.0 for run in runs
    ]
    return min(starts) - WINDOW_MARGIN_MS, max(ends) + WINDOW_MARGIN_MS


def detect_reference(detection_log_path, window_ms) -> Tuple[float, float, Dict]:
    """Find the projection origin the session actually used.

    Returns ``(lat, lon, detail)``. ``detail`` lists every origin seen inside the
    window with its detection count, so a session that straddles a reconfiguration
    is visible rather than silently resolved to the majority.
    """
    low, high = window_ms
    origins: Counter = Counter()
    for record in kafka_log.parse_kafka_log_records(detection_log_path):
        stamp = float(record.get("timestamp", record.get("create_time_ms", 0)))
        if not (low <= stamp <= high):
            continue
        proj = record.get("projString")
        if not proj:
            continue
        origins[parse_projection_origin(proj)] += 1

    if not origins:
        raise ValueError(
            f"No detections with a projection string inside the session window in "
            f"{detection_log_path}"
        )
    (lat, lon), count = origins.most_common(1)[0]
    detail = {
        "origins_in_window": [
            {"lat": key[0], "lon": key[1], "detections": value}
            for key, value in origins.most_common()
        ],
        "selected_detections": count,
        "total_in_window": sum(origins.values()),
    }
    return lat, lon, detail


def _window_kafka_log(source: Path, destination: Path, window_ms) -> int:
    """Copy the records of ``source`` whose CreateTime is inside the window.

    Multi-line JSON bodies are carried along with their header line, so the
    filtered file stays parseable by the same reader.
    """
    low, high = window_ms
    kept = 0
    keeping = False
    with open(source, encoding="utf8", errors="ignore") as reader, \
            open(destination, "w", encoding="utf8") as writer:
        for line in reader:
            if line.startswith("CreateTime"):
                match = _CREATE_TIME.match(line)
                keeping = bool(match) and low <= int(match.group(1)) <= high
                if keeping:
                    kept += 1
            if keeping:
                writer.write(line)
    return kept


def analyse_session(
    runs,
    session,
    plots_dir: Optional[Path] = None,
    max_mean_position_error_m: float = DEFAULT_MAX_MEAN_POSITION_ERROR_M,
    max_mean_heading_error_deg: float = DEFAULT_MAX_MEAN_HEADING_ERROR_DEG,
) -> Dict:
    """Run CS-01 over one session's runs."""
    if session.kafka_detected_object is None:
        raise FileNotFoundError("CS-01 needs the v2xhub_sim_sensor_detected_object log")
    if session.kafka_sdsm is None:
        raise FileNotFoundError("CS-01 needs the v2xhub_sdsm_sub log")

    window = session_window_ms(runs)
    ref_lat, ref_lon, reference_detail = detect_reference(
        session.kafka_detected_object, window
    )

    with tempfile.TemporaryDirectory(prefix="cs01_") as scratch:
        scratch_path = Path(scratch)
        detections = scratch_path / "v2xhub_sim_sensor_detected_object.log"
        sdsms = scratch_path / "v2xhub_sdsm_sub.log"
        kept_detections = _window_kafka_log(session.kafka_detected_object, detections, window)
        kept_sdsms = _window_kafka_log(session.kafka_sdsm, sdsms, window)

        result = verify_location_spoofing(
            detection_log_path=detections,
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            sdsm_log_path=sdsms,
            max_mean_position_error_m=max_mean_position_error_m,
            max_mean_heading_error_deg=max_mean_heading_error_deg,
            plots_dir=plots_dir,
        )

    source = result["sources"].get("kafka", {})
    return {
        "pass": bool(result["pass"]),
        "reference_lat": ref_lat,
        "reference_lon": ref_lon,
        "reference_detail": reference_detail,
        "driver_rotation_deg": result.get("rotation_deg"),
        "runs": len(runs),
        "window_start_ms": window[0],
        "window_end_ms": window[1],
        "detection_records_in_window": kept_detections,
        "sdsm_records_in_window": kept_sdsms,
        "max_mean_position_error_m": max_mean_position_error_m,
        "max_mean_heading_error_deg": max_mean_heading_error_deg,
        **{key: source[key] for key in sorted(source)},
        "rows": result["rows"],
    }
