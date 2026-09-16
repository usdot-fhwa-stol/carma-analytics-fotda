"""Per-run DT-WZ metrics for a verification session.

Each function analyses one run over its engaged window and returns
``(is_passed, stats)``, writing ``stats`` to a JSON file whose keys are additive
counters so the session rollup can glob-and-sum them, matching the convention
already used by ``run_session_analysis``.

``is_passed`` is tri-state: True, False, or **None meaning "not applicable"** --
the metric could not be evaluated because the data to evaluate it was never
recorded. That is distinct from a failure and from an error, and conflating them
is the difference between "SPAT regressed" and "SPAT was never recorded". These
MCAPs contain no ``/message/incoming_map`` or ``/message/incoming_spat`` at all,
so those two PL-01 checks are N/A for the whole session.

All timestamps here are epoch seconds unless a name says ``_ms``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from portable import kafka_log, mcap_backend
from portable import obu_capture as obu_capture_reader  # aliased: the parameter is also obu_capture
from portable.pcap_backend import extract_pcap_messages
from utils import calculate_error_statistics

# correlate_across_boundary lives in a hyphenated directory that is not importable
sys.path.append(str(Path(__file__).resolve().parent.parent / "j2735-pcap"))
from correlate_pcap_mcap import correlate_across_boundary  # noqa: E402

INCOMING_SDSM_TOPIC = "/message/incoming_sdsm"
INCOMING_J3224_TOPIC = "/message/incoming_j3224_sdsm"
BSM_OUTBOUND_TOPIC = "/message/bsm_outbound"

# Thresholds, carried over from carma_cooperative_perception_scripts so results
# stay comparable with previous sessions.
SDSM_DROP_RATE_THRESHOLD_PCT = 2.0
SDSM_DROP_RATE_MATCH_TOLERANCE_SEC = 0.05
RSU_TRANSMISSION_DROP_RATE_THRESHOLD_PCT = 2.0
DETECTION_TO_SDSM_RECEIPT_LATENCY_THRESHOLD_SEC = 0.3
DETECTION_TO_KAFKA_MEAN_LATENCY_THRESHOLD_SEC = 0.5
DETECTION_TO_KAFKA_LATE_THRESHOLD_SEC = 0.1
DETECTION_TO_KAFKA_MAX_LATE_PCT = 2.0
# Above this |rx - tx| a matched pair is counted as late rather than on time.
TRANSMISSION_LATE_THRESHOLD_MS = 200.0


def _write_stats(stats_dir, name: str, stats: Dict) -> None:
    if stats_dir is None:
        return
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / f"{name}.json", "w") as handle:
        json.dump(stats, handle, indent=2, default=float)


def _percentiles(values_ms) -> Optional[Dict[str, float]]:
    if len(values_ms) == 0:
        return None
    array = np.asarray(values_ms, dtype=float)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------

def engaged_window_epoch(mcap_path, engage_time: float, disengage_time: float) -> Tuple[float, float]:
    """Convert a relative engaged window into absolute epoch seconds.

    ``get_engage_time`` returns seconds since the recording start; every other
    source in a session (pcap, Kafka, V2XHub logs) is on an absolute clock, so
    the window has to be lifted before it can be used to filter them.
    """
    _reader, _type_map, start_ns = mcap_backend.open_bagfile(str(mcap_path))
    origin = start_ns / 1e9
    return origin + engage_time, origin + disengage_time


# --------------------------------------------------------------------------
# Decoding SDSM object detections out of the MCAP
# --------------------------------------------------------------------------

def sdsm_object_detections(mcap_path, window=None) -> List[Dict]:
    """Per-object detections from ``/message/incoming_j3224_sdsm``.

    Each entry carries the object's temporary id and the instant it was
    *observed*, reconstructed as ``sdsm_time_stamp - measurement_time``. The
    subtraction matters: the SDSM stamp is when the message was assembled, and
    the measurement offset says how far before that the observation happened, so
    adding instead of subtracting dates every detection ~2x the offset too new.
    """
    from portable.timeutil import sdsm_timestamp_to_epoch_ms

    counts = mcap_backend.topic_message_counts(mcap_path)
    if not counts.get(INCOMING_J3224_TOPIC):
        return []

    reader, _type_map, _start = mcap_backend.open_bagfile(str(mcap_path), topics=[INCOMING_J3224_TOPIC])
    detections: List[Dict] = []
    while reader.has_next():
        _topic, message, log_time_ns = reader.read_next()
        raw = message.to_dict()
        stamp = raw.get("sdsm_time_stamp") or {}
        # Each J3224 DDateTime component is a single-field wrapper struct.
        flat = {
            key: (list(value.values())[0] if isinstance(value, dict) else value)
            for key, value in stamp.items()
        }
        try:
            sdsm_ms = sdsm_timestamp_to_epoch_ms(flat)
        except (KeyError, ValueError, TypeError):
            continue
        receive_sec = log_time_ns / 1e9
        # j3224_v2x_msgs wraps the detection list in a DetectedObjectList struct,
        # so `objects` is {"detected_object_data": [...]}, not the list itself.
        objects = raw.get("objects", [])
        if isinstance(objects, dict):
            objects = objects.get("detected_object_data", [])
        for entry in objects:
            if not isinstance(entry, dict):
                continue
            # Entries are already DetectedObjectData; tolerate a further wrapper
            # in case a different message revision nests one.
            if "detected_object_common_data" not in entry:
                entry = entry.get("detected_object_data") or {}
            common = entry.get("detected_object_common_data") or {}
            object_id = common.get("detected_id", {})
            object_id = object_id.get("object_id") if isinstance(object_id, dict) else object_id
            measurement = common.get("measurement_time", {})
            measurement = (
                measurement.get("measurement_time_offset")
                if isinstance(measurement, dict) else measurement
            )
            if object_id is None or measurement is None:
                continue
            detections.append(
                {
                    "object_id": int(object_id),
                    "detection_time_sec": (float(sdsm_ms) - float(measurement)) / 1e3,
                    "sdsm_time_sec": float(sdsm_ms) / 1e3,
                    "receive_time_sec": receive_sec,
                }
            )
    if window is not None:
        low, high = window
        detections = [d for d in detections if low <= d["receive_time_sec"] <= high]
    return detections


# --------------------------------------------------------------------------
# CP-02: raw detection -> SDSM drop rate
# --------------------------------------------------------------------------

def detection_to_sdsm_drop_rate(
    mcap_path, detection_records, window, stats_dir=None,
    max_drop_rate_pct=SDSM_DROP_RATE_THRESHOLD_PCT,
    match_tolerance_sec=SDSM_DROP_RATE_MATCH_TOLERANCE_SEC,
):
    """Fraction of FLIR detections that never reached the vehicle inside an SDSM.

    Matching is per object id and consumes each SDSM detection at most once, so
    two raw detections cannot both claim the same reported one.
    """
    sdsm_detections = sdsm_object_detections(mcap_path, window)
    raw = kafka_log.window_records(detection_records, window[0] * 1e3, window[1] * 1e3)

    if not raw:
        stats = {
            "total_raw_detections": 0, "total_matched": 0, "total_dropped": 0,
            "drop_rate_pct": None, "max_drop_rate_pct": max_drop_rate_pct,
            "match_tolerance_sec": match_tolerance_sec, "is_passed": None,
            "note": "no raw detections inside the engaged window",
        }
        _write_stats(stats_dir, "cp02_sdsm_detection_drop_rate", stats)
        return None, stats

    reported: Dict[int, List[float]] = {}
    for detection in sdsm_detections:
        reported.setdefault(detection["object_id"], []).append(detection["detection_time_sec"])
    for times in reported.values():
        times.sort()

    consumed = {object_id: set() for object_id in reported}
    matched = 0
    dropped: List[Tuple[int, float]] = []
    per_object: Dict[str, Dict] = {}

    for record in sorted(raw, key=lambda r: float(r.get("timestamp", r["create_time_ms"]))):
        object_id = int(record["objectId"])
        raw_time = float(record.get("timestamp", record["create_time_ms"])) / 1e3
        candidates = reported.get(object_id, [])
        hit = None
        for index, candidate in enumerate(candidates):
            if index in consumed[object_id]:
                continue
            if abs(candidate - raw_time) <= match_tolerance_sec:
                hit = index
                break
        bucket = per_object.setdefault(
            str(object_id), {"raw_detections": 0, "matched": 0, "dropped": 0}
        )
        bucket["raw_detections"] += 1
        if hit is None:
            dropped.append((object_id, raw_time))
            bucket["dropped"] += 1
        else:
            consumed[object_id].add(hit)
            matched += 1
            bucket["matched"] += 1

    total = len(raw)
    drop_rate = len(dropped) / total * 100.0
    for bucket in per_object.values():
        bucket["drop_rate_pct"] = bucket["dropped"] / bucket["raw_detections"] * 100.0

    stats = {
        "total_raw_detections": total,
        "total_matched": matched,
        "total_dropped": len(dropped),
        "drop_rate_pct": drop_rate,
        "max_drop_rate_pct": max_drop_rate_pct,
        "match_tolerance_sec": match_tolerance_sec,
        "sdsm_object_detections": len(sdsm_detections),
        "is_passed": bool(drop_rate <= max_drop_rate_pct),
        "per_object_id": per_object,
    }
    _write_stats(stats_dir, "cp02_sdsm_detection_drop_rate", stats)
    return stats["is_passed"], stats


# --------------------------------------------------------------------------
# CP-03: RSU broadcast -> CARMA Platform receipt
# --------------------------------------------------------------------------

def rsu_transmission_drop_rate(
    mcap_path, rsu_pcap, window, stats_dir=None,
    max_drop_rate_pct=RSU_TRANSMISSION_DROP_RATE_THRESHOLD_PCT,
):
    """SDSMs the RSU broadcast that CARMA Platform never received.

    This session captured the broadcasting RSU itself, so tx and rx are matched
    byte-for-byte on the J2735 payload -- an exact identity, not a time-proximity
    guess. Broadcasts outside the engaged window are excluded rather than scored,
    since the vehicle was not listening then.
    """
    broadcasts = [
        message for message in extract_pcap_messages(rsu_pcap, ["SDSM"])
        if window[0] <= message["timestamp"] <= window[1]
    ]
    received = [
        message for message in mcap_backend.extract_mcap_binary_messages(mcap_path)["inbound"]
        if message["msg_type"] == "SDSM"
    ]

    if not broadcasts:
        stats = {
            "rsu_pcap": str(rsu_pcap), "total_broadcasts_checked": 0,
            "total_received": 0, "total_received_late": 0, "total_dropped": 0,
            "total_outside_recording_window": 0, "drop_rate_pct": None,
            "max_drop_rate_pct": max_drop_rate_pct, "is_passed": None,
            "note": "no RSU SDSM broadcasts inside the engaged window",
        }
        _write_stats(stats_dir, "cp03_rsu_sdsm_transmission_drop_rate", stats)
        return None, stats

    latencies, drops, stale, out_of_window = correlate_across_boundary(
        broadcasts, received, TRANSMISSION_LATE_THRESHOLD_MS
    )
    checked = len(broadcasts) - len(out_of_window)
    drop_rate = (len(drops) / checked * 100.0) if checked else None

    stats = {
        "rsu_pcap": str(rsu_pcap),
        "total_broadcasts_checked": checked,
        "total_received": len(latencies),
        "total_received_late": len(stale),
        "total_dropped": len(drops),
        "total_outside_recording_window": len(out_of_window),
        "drop_rate_pct": drop_rate,
        "max_drop_rate_pct": max_drop_rate_pct,
        "late_threshold_ms": TRANSMISSION_LATE_THRESHOLD_MS,
        "receive_latency_ms": _percentiles([entry[2] for entry in latencies]),
        "is_passed": None if drop_rate is None else bool(drop_rate <= max_drop_rate_pct),
    }
    _write_stats(stats_dir, "cp03_rsu_sdsm_transmission_drop_rate", stats)
    return stats["is_passed"], stats


# --------------------------------------------------------------------------
# CP-04: detection -> Kafka latency
# --------------------------------------------------------------------------

def detection_to_kafka_latency(
    detection_records, window, stats_dir=None,
    max_mean_latency_sec=DETECTION_TO_KAFKA_MEAN_LATENCY_THRESHOLD_SEC,
    late_threshold_sec=DETECTION_TO_KAFKA_LATE_THRESHOLD_SEC,
    max_late_pct=DETECTION_TO_KAFKA_MAX_LATE_PCT,
):
    """Delay between the camera's detection stamp and the broker accepting it."""
    raw = kafka_log.window_records(detection_records, window[0] * 1e3, window[1] * 1e3)
    if not raw:
        stats = {
            "total_detections": 0, "late_detections": 0, "late_pct": None,
            "mean_latency_s": None, "is_passed": None,
            "note": "no raw detections inside the engaged window",
        }
        _write_stats(stats_dir, "cp04_detection_to_kafka_latency", stats)
        return None, stats

    latency_sec = np.array(
        [(record["create_time_ms"] - float(record["timestamp"])) / 1e3 for record in raw],
        dtype=float,
    )
    detection_times = np.array([float(record["timestamp"]) / 1e3 for record in raw], dtype=float)
    late = int(np.sum(latency_sec > late_threshold_sec))
    late_pct = late / len(latency_sec) * 100.0
    mean_latency = float(np.mean(latency_sec))

    stats = {
        "latency_s": calculate_error_statistics(latency_sec),
        "total_detections": len(latency_sec),
        "late_detections": late,
        "late_pct": late_pct,
        "mean_latency_s": mean_latency,
        "max_mean_latency_s": max_mean_latency_sec,
        "late_threshold_s": late_threshold_sec,
        "max_late_pct": max_late_pct,
        "is_passed": bool(mean_latency < max_mean_latency_sec and late_pct < max_late_pct),
    }
    _write_stats(stats_dir, "cp04_detection_to_kafka_latency", stats)
    return stats["is_passed"], (stats | {"_detection_times_sec": detection_times, "_latency_s": latency_sec})


# --------------------------------------------------------------------------
# DT-05: detection -> SDSM receipt latency at the vehicle
# --------------------------------------------------------------------------

def detection_to_sdsm_receipt_latency(
    mcap_path, window, stats_dir=None,
    threshold_sec=DETECTION_TO_SDSM_RECEIPT_LATENCY_THRESHOLD_SEC,
):
    """End-to-end: camera observation -> CARMA Platform receiving it in an SDSM."""
    detections = sdsm_object_detections(mcap_path, window)
    if not detections:
        stats = {
            "sample_count": 0, "median_latency_s": None, "is_passed": None,
            "note": "no SDSM object detections inside the engaged window",
        }
        _write_stats(stats_dir, "dt05_detection_to_sdsm_receipt_latency", stats)
        return None, stats

    latency_sec = np.array(
        [d["receive_time_sec"] - d["detection_time_sec"] for d in detections], dtype=float
    )
    statistics = calculate_error_statistics(latency_sec)
    median = float(np.median(latency_sec))

    stats = {
        "latency_s": statistics,
        "sample_count": int(len(latency_sec)),
        "median_latency_s": median,
        "threshold_s": threshold_sec,
        "is_passed": bool(median < threshold_sec),
    }
    _write_stats(stats_dir, "dt05_detection_to_sdsm_receipt_latency", stats)
    return stats["is_passed"], (
        stats | {
            "_detection_times_sec": np.array([d["detection_time_sec"] for d in detections]),
            "_latency_s": latency_sec,
        }
    )


# --------------------------------------------------------------------------
# PL-01: message rates, and the OBU radio side
# --------------------------------------------------------------------------

def detection_intervals(detection_records, window, gap_sec=0.5) -> List[Tuple[float, float]]:
    """Periods inside ``window`` during which the camera was reporting detections.

    Consecutive detections more than ``gap_sec`` apart start a new period. Needed
    because SDSMs are only produced while an object is present: over a 32 s
    engaged window containing ~6 s of pedestrian, averaging SDSMs over the whole
    window reports ~1.8 Hz for a stream that is running at its nominal 10 Hz
    whenever it runs at all. Rate has to be measured over the active periods or
    the check measures how long the pedestrian was absent.
    """
    raw = kafka_log.window_records(detection_records, window[0] * 1e3, window[1] * 1e3)
    times = np.unique(
        np.array([float(r.get("timestamp", r["create_time_ms"])) / 1e3 for r in raw], dtype=float)
    )
    if len(times) < 2:
        return []
    breaks = np.flatnonzero(np.diff(times) > gap_sec)
    starts = np.r_[times[0], times[breaks + 1]]
    ends = np.r_[times[breaks], times[-1]]
    return [(float(a), float(b)) for a, b in zip(starts, ends) if b > a]


def message_rate(mcap_path, topic, expected_rate_hz, window, stats_dir=None,
                 tolerance_pct=0.2, stats_name=None, active_intervals=None):
    """Average receive/broadcast rate of one topic over the engaged window.

    ``active_intervals`` restricts both the message count and the duration to
    periods when the message was expected at all; pass it for event-driven
    topics such as SDSM.

    Returns ``is_passed=None`` when the topic is absent from the recording: the
    check is inapplicable, not failed. MAP and SPAT are in exactly that position
    for this session -- they were never recorded, so there is nothing to regress.
    """
    counts = mcap_backend.topic_message_counts(mcap_path)
    label = stats_name or topic.strip("/").replace("/", "_")
    if topic not in counts:
        stats = {
            "topic_name": topic, "expected_rate_hz": expected_rate_hz,
            "total_messages": 0, "average_rate_hz": None, "is_passed": None,
            "not_applicable": True,
            "note": "topic absent from this recording; nothing to evaluate",
        }
        _write_stats(stats_dir, f"pl01_{label}_rate", stats)
        return None, stats

    reader, _type_map, start_ns = mcap_backend.open_bagfile(str(mcap_path), topics=[topic])
    origin = start_ns / 1e9
    timestamps = []
    while reader.has_next():
        _topic, _message, log_time_ns = reader.read_next()
        seconds = log_time_ns / 1e9
        if window[0] <= seconds <= window[1]:
            timestamps.append(seconds)

    if active_intervals:
        counted = [
            stamp for stamp in timestamps
            if any(start <= stamp <= end for start, end in active_intervals)
        ]
        duration = sum(end - start for start, end in active_intervals)
    else:
        counted = timestamps
        duration = window[1] - window[0]

    average = len(counted) / duration if duration > 0 else None
    low = expected_rate_hz * (1 - tolerance_pct)
    high = expected_rate_hz * (1 + tolerance_pct)
    intervals = np.diff(sorted(counted)) if len(counted) > 1 else np.array([])

    stats = {
        "topic_name": topic,
        "expected_rate_hz": expected_rate_hz,
        "rate_tolerance_pct": tolerance_pct,
        "total_messages": len(counted),
        "total_messages_in_window": len(timestamps),
        "analysis_duration_s": duration,
        "active_intervals": len(active_intervals) if active_intervals else 0,
        "average_rate_hz": average,
        "interval_s": _percentiles(intervals) if len(intervals) else None,
        "is_passed": None if average is None else bool(low <= average <= high),
        "not_applicable": False,
    }
    if not counted:
        # Distinguish "never arrived" from "arrived too slowly". Both fail the
        # check, and both should: a topic that is silent when it is expected at
        # a given rate is a real finding. But the two have different causes and
        # the bare rate of 0.0 does not say which this is.
        stats["note"] = (
            "topic is present in the recording but no messages fall inside the "
            "analysis window; this is absence, not a slow rate"
        )
    _write_stats(stats_dir, f"pl01_{label}_rate", stats)
    return stats["is_passed"], stats


def obu_radio_activity(obu_capture, capture_date, mcap_path, window, stats_dir=None,
                       rsu_pcap=None):
    """OBU radio traffic against what CARMA Platform sent and received.

    Two count comparisons are always available and are reported as
    characterization only, with no pass/fail:

    * BSMs the radio put on the air vs. ``/message/bsm_outbound``
    * SDSMs the radio received vs. ``/message/incoming_sdsm``

    A count shortfall is evidence of loss; a count match is consistent with no
    loss but does not prove each message survived, since ordinal counting cannot
    distinguish a drop followed by a duplicate from clean one-to-one delivery.

    When the capture is a binary pcap it also carries payload bytes, and passing
    ``rsu_pcap`` then adds a genuine over-the-air reception rate: which of the
    RSU's individual broadcasts this OBU actually received, matched byte for
    byte. That is not available from a tcpdump text capture at any sample size.
    """
    capture = obu_capture_reader.read_obu_capture(obu_capture, capture_date)
    counts = obu_capture_reader.count_by_type(capture, window[0], window[1])
    topic_counts = mcap_backend.topic_message_counts(mcap_path)

    ros_bsm = 0
    ros_sdsm = 0
    for topic, expected in ((BSM_OUTBOUND_TOPIC, "bsm"), (INCOMING_SDSM_TOPIC, "sdsm")):
        if topic not in topic_counts:
            continue
        reader, _type_map, _start = mcap_backend.open_bagfile(str(mcap_path), topics=[topic])
        total = 0
        while reader.has_next():
            _topic, _message, log_time_ns = reader.read_next()
            if window[0] <= log_time_ns / 1e9 <= window[1]:
                total += 1
        if expected == "bsm":
            ros_bsm = total
        else:
            ros_sdsm = total

    bsm_radio = counts.get("BSM", 0)
    sdsm_radio = counts.get("SDSM", 0)
    received = obu_capture_reader.messages_of_type(capture, "SDSM", window[0], window[1])
    sdsm_times = [message["timestamp"] for message in received]
    sdsm_intervals = np.diff(sdsm_times) if len(sdsm_times) > 1 else np.array([])

    stats = {
        "obu_capture": str(obu_capture),
        "payload_matched": capture["payloads_available"],
        "note": (
            "binary pcap: messages identified by payload bytes"
            if capture["payloads_available"]
            else "tcpdump text capture: counts and timing only, no payload matching"
        ),
        "total_bsms_sent_by_platform": ros_bsm,
        "total_bsms_on_radio": bsm_radio,
        "bsm_shortfall": max(ros_bsm - bsm_radio, 0),
        "bsm_shortfall_pct": (max(ros_bsm - bsm_radio, 0) / ros_bsm * 100.0) if ros_bsm else None,
        "total_sdsms_received_by_platform": ros_sdsm,
        "total_sdsms_on_radio": sdsm_radio,
        "sdsm_shortfall": max(sdsm_radio - ros_sdsm, 0),
        "sdsm_radio_interval_s": _percentiles(sdsm_intervals) if len(sdsm_intervals) else None,
    }

    # Over-the-air reception, byte for byte. Only possible when the capture
    # carries payloads; a text capture cannot support this at any sample size.
    if capture["payloads_available"] and rsu_pcap is not None:
        broadcast = [
            message for message in extract_pcap_messages(rsu_pcap, ["SDSM"])
            if window[0] <= message["timestamp"] <= window[1]
        ]
        heard = {message["payload_hex"] for message in received}
        matched = sum(1 for message in broadcast if message["payload_hex"] in heard)
        stats["ota_broadcasts_in_window"] = len(broadcast)
        stats["ota_received_by_radio"] = matched
        stats["ota_missed_by_radio"] = len(broadcast) - matched
        stats["ota_reception_rate_pct"] = (
            matched / len(broadcast) * 100.0 if broadcast else None
        )
    _write_stats(stats_dir, "pl01_obu_radio_activity", stats)
    return stats
