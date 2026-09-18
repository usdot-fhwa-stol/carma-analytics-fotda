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

from .readers import kafka_log

# carma-streets is a hyphenated directory, so it is not importable by name
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "carma-streets"))
from sdsm_location_spoofing_verification import (  # noqa: E402
    DEFAULT_MAX_MEAN_HEADING_ERROR_DEG,
    DEFAULT_MAX_MEAN_POSITION_ERROR_M,
    REFERENCE_MATCH_TOLERANCE_DEG,
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


def _window_kafka_log(source: Path, destination: Path, window_ms, append: bool = False) -> int:
    """Copy the records of ``source`` whose CreateTime is inside the window.

    Multi-line JSON bodies are carried along with their header line, so the
    filtered file stays parseable by the same reader. With ``append`` the records
    are added to ``destination``, which is how several sessions are pooled into
    one log for a single verification.
    """
    low, high = window_ms
    kept = 0
    keeping = False
    with open(source, encoding="utf8", errors="ignore") as reader, \
            open(destination, "a" if append else "w", encoding="utf8") as writer:
        for line in reader:
            if line.startswith("CreateTime"):
                match = _CREATE_TIME.match(line)
                keeping = bool(match) and low <= int(match.group(1)) <= high
                if keeping:
                    kept += 1
            if keeping:
                writer.write(line)
    return kept


def analyse_sessions(
    specs,
    plots_dir: Optional[Path] = None,
    max_mean_position_error_m: float = DEFAULT_MAX_MEAN_POSITION_ERROR_M,
    max_mean_heading_error_deg: float = DEFAULT_MAX_MEAN_HEADING_ERROR_DEG,
) -> Dict:
    """Run CS-01 once over the pooled runs of one or more sessions.

    ``specs`` is a sequence of ``(name, runs, session)``. Each session is windowed
    to its own runs, then the windowed records are concatenated and verified
    together, so the result is a single figure for every run supplied.

    Pooling is only valid while every session used the same configured reference,
    so the references are compared and a mismatch raises instead of silently
    averaging two different geometries. Pooling is safe against recycled object
    ids because objects are matched on id *and* a timestamp within a few
    milliseconds, and the sessions are a day apart.
    """
    specs = list(specs)
    if not specs:
        raise ValueError("CS-01 needs at least one session")

    prepared = []
    for name, runs, session in specs:
        if session.kafka_detected_object is None:
            raise FileNotFoundError(f"{name}: CS-01 needs the v2xhub_sim_sensor_detected_object log")
        if session.kafka_sdsm is None:
            raise FileNotFoundError(f"{name}: CS-01 needs the v2xhub_sdsm_sub log")
        window = session_window_ms(runs)
        ref_lat, ref_lon, detail = detect_reference(session.kafka_detected_object, window)
        prepared.append((name, runs, session, window, ref_lat, ref_lon, detail))

    ref_lat, ref_lon = prepared[0][4], prepared[0][5]
    for name, _runs, _session, _window, lat, lon, _detail in prepared[1:]:
        if (abs(lat - ref_lat) > REFERENCE_MATCH_TOLERANCE_DEG
                or abs(lon - ref_lon) > REFERENCE_MATCH_TOLERANCE_DEG):
            raise ValueError(
                f"Sessions use different configured references, so they cannot be pooled: "
                f"{prepared[0][0]} at {ref_lat:.7f},{ref_lon:.7f} but {name} at {lat:.7f},{lon:.7f}. "
                f"Verify them separately."
            )

    per_session = []
    with tempfile.TemporaryDirectory(prefix="cs01_") as scratch:
        scratch_path = Path(scratch)
        detections = scratch_path / "v2xhub_sim_sensor_detected_object.log"
        sdsms = scratch_path / "v2xhub_sdsm_sub.log"

        for name, runs, session, window, lat, lon, detail in prepared:
            kept_detections = _window_kafka_log(
                session.kafka_detected_object, detections, window, append=True
            )
            kept_sdsms = _window_kafka_log(session.kafka_sdsm, sdsms, window, append=True)
            per_session.append({
                "session": name,
                "runs": len(runs),
                "reference_lat": lat,
                "reference_lon": lon,
                "window_start_ms": window[0],
                "window_end_ms": window[1],
                "detection_records_in_window": kept_detections,
                "sdsm_records_in_window": kept_sdsms,
                "origins_in_window": detail["origins_in_window"],
            })

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
        "driver_rotation_deg": result.get("rotation_deg"),
        "sessions": [item["session"] for item in per_session],
        "total_runs": sum(item["runs"] for item in per_session),
        "detection_records_in_window": sum(item["detection_records_in_window"] for item in per_session),
        "sdsm_records_in_window": sum(item["sdsm_records_in_window"] for item in per_session),
        "max_mean_position_error_m": max_mean_position_error_m,
        "max_mean_heading_error_deg": max_mean_heading_error_deg,
        **{key: source[key] for key in sorted(source)},
        "per_session": per_session,
        "rows": result["rows"],
    }


def analyse_session(runs, session, **kwargs) -> Dict:
    """Run CS-01 over a single session. Thin wrapper over ``analyse_sessions``."""
    return analyse_sessions([("session", runs, session)], **kwargs)
