"""Assemble the per-detection stage table for one run.

The session's V2XHub, SDSS and Kafka logs are continuous across every run -- pc2
alone is 1.76 M lines and the Kafka dumps hold the broker's whole retention,
opening ten days before the session -- so they are parsed **once** by
``SessionLogs`` and then windowed per run. Windowing before joining is not
optional: object ids are recycled within a day, so an unwindowed join attaches
detections from an earlier run, or an earlier week, to this run's SDSMs.

Stages are joined on the strongest key available at each hop:

1. **Detection identity** ``(object_id, t_flir_detect)`` for everything up to the
   SDSM being built. On the SDSM side this identity is *recovered* arithmetically
   as ``sdsm_time_stamp - measurement_time``, not guessed.
2. **UPER payload bytes** once the SDSM is encoded, through TENA, the RSU
   broadcast and the vehicle's inbound topic. Byte identity, so these hops cannot
   be mismatched.
3. **Ordinal position** for the OBU radio stage alone, which has no payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..readers import kafka_log, mcap_reader
from ..readers.pcap_reader import extract_pcap_messages
from ..timeutil import sdsm_timestamp_to_epoch_ms

from . import cascade_config as config
from . import parse_sdss, parse_v2xhub

DETECT_KEY = ["object_id", "t_flir_detect"]


@dataclass
class SessionLogs:
    """Continuous session logs, parsed once and shared by every run."""

    pc2: Dict[str, pd.DataFrame]
    pc2_chain: pd.DataFrame
    pc1: pd.DataFrame
    sdss: Dict[str, pd.DataFrame]
    kafka_sdo: pd.DataFrame
    kafka_sdsm: pd.DataFrame

    @classmethod
    def load(cls, session, verbose: bool = True) -> "SessionLogs":
        def announce(message):
            if verbose:
                print(f"  {message}", flush=True)

        announce(f"parsing pc2 V2XHub log ({session.pc2_v2xhub.name}) ...")
        pc2 = parse_v2xhub.parse_pc2(session.pc2_v2xhub)
        pc2_chain = parse_v2xhub.build_pc2_sdsm_chain(pc2)

        announce(f"parsing pc1 V2XHub log ({session.pc1_v2xhub.name}) ...")
        pc1 = parse_v2xhub.parse_pc1(session.pc1_v2xhub)

        announce(f"parsing SDSS log ({session.sdss.name}) ...")
        sdss = parse_sdss.parse_sdss(session.sdss)

        announce("parsing Kafka topic dumps ...")
        kafka_sdo = _kafka_detection_frame(session.kafka_detected_object)
        kafka_sdsm = _kafka_sdsm_frame(session.kafka_sdsm)

        return cls(pc2=pc2, pc2_chain=pc2_chain, pc1=pc1, sdss=sdss,
                   kafka_sdo=kafka_sdo, kafka_sdsm=kafka_sdsm)


def _kafka_detection_frame(path) -> pd.DataFrame:
    """Broker CreateTime per detection, keyed by detection identity."""
    if path is None:
        return pd.DataFrame(columns=["t_kafka_sdo_create", "object_id", "t_flir_detect"])
    rows = [
        {
            "t_kafka_sdo_create": float(record["create_time_ms"]),
            "object_id": int(record["objectId"]),
            "t_flir_detect": float(record["timestamp"]),
        }
        for record in kafka_log.parse_kafka_log_records(path)
        if "objectId" in record and "timestamp" in record
    ]
    return pd.DataFrame(rows)


def _kafka_sdsm_frame(path) -> pd.DataFrame:
    """Broker CreateTime per SDSM, keyed by the detection identity it encodes."""
    if path is None:
        return pd.DataFrame(columns=["t_kafka_sdsm_create", "object_id", "t_flir_detect"])
    rows = []
    for record in kafka_log.parse_kafka_log_records(path):
        stamp = record.get("sdsm_time_stamp")
        objects = record.get("objects") or []
        if not stamp or not objects:
            continue
        common = (objects[0].get("detected_object_data") or {}).get(
            "detected_object_common_data"
        ) or {}
        # The Kafka JSON flattens what the ROS message wraps: it carries
        # "object_id" directly where j3224_v2x_msgs nests it under "detected_id".
        object_id = common.get("object_id", common.get("detected_id"))
        if isinstance(object_id, dict):
            object_id = object_id.get("object_id")
        measurement = common.get("measurement_time")
        if object_id is None or measurement is None:
            continue
        try:
            sdsm_ms = sdsm_timestamp_to_epoch_ms(stamp)
        except (KeyError, ValueError, TypeError):
            continue
        rows.append(
            {
                "t_kafka_sdsm_create": float(record["create_time_ms"]),
                "object_id": int(object_id),
                "t_flir_detect": float(sdsm_ms) - float(measurement),
            }
        )
    return pd.DataFrame(rows)


def _window(frame: pd.DataFrame, column: str, window_ms) -> pd.DataFrame:
    """Rows whose ``column`` falls inside the run window."""
    if frame is None or frame.empty or column not in frame.columns:
        return frame if frame is not None else pd.DataFrame()
    low, high = window_ms
    return frame[(frame[column] >= low) & (frame[column] <= high)].copy()


def _merge_on_detection(base: pd.DataFrame, frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Left-merge stage columns onto the table using detection identity.

    The right side is de-duplicated on the key first: a repeated
    ``(object_id, t_flir_detect)`` on a stage would otherwise fan one detection
    out into several rows and silently inflate every downstream count.
    """
    if frame is None or frame.empty:
        return base
    present = [column for column in columns if column in frame.columns]
    if not present or not all(key in frame.columns for key in DETECT_KEY):
        return base
    right = frame[DETECT_KEY + present].drop_duplicates(subset=DETECT_KEY, keep="first")
    return base.merge(right, on=DETECT_KEY, how="left", validate="many_to_one")


def _merge_on_uper(base: pd.DataFrame, frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Left-merge stage columns onto the table using the SDSM's payload bytes."""
    if frame is None or frame.empty or "sdsm_uper_hex" not in base.columns:
        return base
    present = [column for column in columns if column in frame.columns]
    if not present or "sdsm_uper_hex" not in frame.columns:
        return base
    right = frame[["sdsm_uper_hex"] + present].dropna(subset=["sdsm_uper_hex"])
    right = right.drop_duplicates(subset=["sdsm_uper_hex"], keep="first")
    return base.merge(right, on="sdsm_uper_hex", how="left", validate="many_to_one")


# A broadcast and its radio reception sit ~7-25 ms apart, and consecutive SDSMs
# are ~100 ms apart, so this window admits the right packet and excludes the
# neighbours by a wide margin.
OBU_MATCH_WINDOW_MS = 60.0


def attach_obu_radio(table: pd.DataFrame, radio_messages: List[dict]) -> pd.DataFrame:
    """Attach the OBU radio receive stage, by payload bytes where possible.

    ``radio_messages`` are ``{timestamp, payload_hex}`` dicts with ``timestamp``
    already in epoch **milliseconds**. When ``payload_hex`` is present the join
    is an exact identity, the same as every other stage from ``t_streets_encode``
    onward. When it is None -- a tcpdump text capture, which records no payload --
    the code falls back to nearest-time matching.

    Both forms are in use: the 2026-09-14 session captured the OBU as text, the
    2026-09-15 session as binary pcap. The fallback is weaker and the QA output
    says which was used, because a latency from ordinal pairing is not the same
    measurement as one from byte identity.
    """
    table["t_obu_radio_rx"] = np.nan
    if not radio_messages or "t_rsu_broadcast" not in table.columns:
        return table

    if radio_messages[0].get("payload_hex") and "sdsm_uper_hex" in table.columns:
        by_payload = {}
        for message in radio_messages:
            by_payload.setdefault(message["payload_hex"], message["timestamp"])
        table["t_obu_radio_rx"] = table["sdsm_uper_hex"].map(by_payload)
        return table

    radio_times = [message["timestamp"] for message in radio_messages]

    ordered = table.dropna(subset=["t_rsu_broadcast"]).sort_values("t_rsu_broadcast")
    available = sorted(float(value) for value in radio_times)
    used = [False] * len(available)

    for index, broadcast in zip(ordered.index, ordered["t_rsu_broadcast"].to_numpy(dtype=float)):
        best, best_gap = None, None
        position = int(np.searchsorted(available, broadcast, side="left"))
        # Walk outwards from the insertion point; the window is small, so this
        # only ever inspects a couple of candidates.
        for candidate in range(max(position - 2, 0), min(position + 3, len(available))):
            if used[candidate]:
                continue
            gap = available[candidate] - broadcast
            if gap < -OBU_MATCH_WINDOW_MS or gap > OBU_MATCH_WINDOW_MS:
                continue
            if best_gap is None or abs(gap) < abs(best_gap):
                best, best_gap = candidate, gap
        if best is not None:
            used[best] = True
            table.loc[index, "t_obu_radio_rx"] = available[best]
    return table


def build_run_table(run, session_logs: SessionLogs, window_sec) -> pd.DataFrame:
    """One row per FLIR detection in this run, with every stage it reached."""
    window_ms = (window_sec[0] * 1e3, window_sec[1] * 1e3)

    flir = _window(session_logs.pc2.get("flir"), "t_flir_detect", window_ms)
    if flir is None or flir.empty:
        return pd.DataFrame(columns=list(config.STAGE_COLUMNS))

    table = flir[DETECT_KEY + ["t_flir_log"]].drop_duplicates(subset=DETECT_KEY, keep="first").copy()

    table = _merge_on_detection(table, session_logs.pc2.get("streets_rx"), ["t_streets_rx"])
    table = _merge_on_detection(table, session_logs.pc2.get("kafka_produce"), ["t_kafka_produce"])
    table = _merge_on_detection(table, session_logs.kafka_sdo, ["t_kafka_sdo_create"])
    table = _merge_on_detection(table, session_logs.sdss.get("consumed"), ["t_sdss_consume"])
    table = _merge_on_detection(table, session_logs.sdss.get("sent"), ["t_sdss_send"])
    table = _merge_on_detection(table, session_logs.kafka_sdsm, ["t_kafka_sdsm_create"])
    table = _merge_on_detection(
        table,
        session_logs.pc2_chain,
        ["t_streets_sdsm_rx", "t_streets_encode", "t_pc2_tena_rx", "t_pc2_tena_tx", "sdsm_uper_hex"],
    )
    table = _merge_on_uper(
        table, session_logs.pc1, ["t_pc1_tena_rx", "t_pc1_bus", "t_immediate_fwd"]
    )
    return table


def attach_run_sources(table: pd.DataFrame, run, window_sec, radio_messages=None) -> pd.DataFrame:
    """Add the per-run stages: RSU broadcast, OBU radio, and the vehicle's ROS topics."""
    if table.empty:
        return table

    broadcasts = pd.DataFrame(
        [
            {"sdsm_uper_hex": message["payload_hex"], "t_rsu_broadcast": message["timestamp"] * 1e3}
            for message in extract_pcap_messages(run.rsu_pcap, ["SDSM"])
            if window_sec[0] <= message["timestamp"] <= window_sec[1]
        ]
    )
    table = _merge_on_uper(table, broadcasts, ["t_rsu_broadcast"])

    inbound = pd.DataFrame(
        [
            {"sdsm_uper_hex": message["payload_hex"], "t_ros_inbound": message["timestamp"] * 1e3}
            for message in mcap_reader.extract_mcap_binary_messages(run.mcap)["inbound"]
            if message["msg_type"] == "SDSM"
            and window_sec[0] <= message["timestamp"] <= window_sec[1]
        ]
    )
    table = _merge_on_uper(table, inbound, ["t_ros_inbound"])

    table = _attach_ros_stages(table, run, window_sec)
    table = attach_obu_radio(table, radio_messages or [])
    return table


def _attach_ros_stages(table: pd.DataFrame, run, window_sec) -> pd.DataFrame:
    """Attach the J3224 receive time by detection identity, and the fused output by track."""
    from ..metrics import sdsm_object_detections

    detections = sdsm_object_detections(run.mcap, window_sec)
    if detections:
        j3224 = pd.DataFrame(
            [
                {
                    "object_id": d["object_id"],
                    "t_flir_detect": d["detection_time_sec"] * 1e3,
                    "t_ros_j3224": d["receive_time_sec"] * 1e3,
                }
                for d in detections
            ]
        )
        # Detection identity is reconstructed from millisecond fields on both
        # sides, so round before joining to absorb sub-millisecond drift.
        j3224["t_flir_detect"] = j3224["t_flir_detect"].round()
        table["_key"] = table["t_flir_detect"].round()
        j3224 = j3224.rename(columns={"t_flir_detect": "_key"}).drop_duplicates(
            subset=["object_id", "_key"], keep="first"
        )
        table = table.merge(j3224, on=["object_id", "_key"], how="left", validate="many_to_one")
        table = table.drop(columns=["_key"])
    else:
        table["t_ros_j3224"] = np.nan

    table["t_ros_fused"] = _fused_times(run, window_sec, table)
    return table


def _fused_times(run, window_sec, table) -> np.ndarray:
    """First fused publication after each detection reached the vehicle.

    The tracker reassigns ids, so this is a gated nearest-in-time match rather
    than an identity join: the first fused output published after the detection
    arrived, within ``FUSED_MATCH_WINDOW_MS``. It answers "when did this
    detection first influence a fused output", which is the question the
    end-to-end latency asks, and not "which track is this detection".
    """
    counts = mcap_reader.topic_message_counts(run.mcap)
    topic = "/environment/fused_external_objects"
    if not counts.get(topic) or "t_ros_j3224" not in table.columns:
        return np.full(len(table), np.nan)

    reader, _type_map, _start = mcap_reader.open_bagfile(str(run.mcap), topics=[topic])
    published = []
    while reader.has_next():
        _topic, _message, log_time_ns = reader.read_next()
        seconds = log_time_ns / 1e9
        if window_sec[0] <= seconds <= window_sec[1]:
            published.append(seconds * 1e3)
    published = np.array(sorted(published), dtype=float)
    if published.size == 0:
        return np.full(len(table), np.nan)

    arrival = table["t_ros_j3224"].to_numpy(dtype=float)
    result = np.full(len(table), np.nan)
    valid = ~np.isnan(arrival)
    if valid.any():
        index = np.searchsorted(published, arrival[valid], side="left")
        index = np.clip(index, 0, len(published) - 1)
        candidate = published[index]
        gap = candidate - arrival[valid]
        candidate[(gap < 0) | (gap > config.FUSED_MATCH_WINDOW_MS)] = np.nan
        result[valid] = candidate
    return result


def add_deltas(table: pd.DataFrame, stages=None) -> pd.DataFrame:
    """Add per-hop and total latencies, in milliseconds, relative to the detection."""
    if table.empty:
        return table
    stages = [stage for stage in (stages or config.STAGE_COLUMNS) if stage in table.columns]
    for earlier, later in zip(stages, stages[1:]):
        table[f"d_{earlier[2:]}__{later[2:]}_ms"] = table[later] - table[earlier]
    for stage in stages:
        if stage != "t_flir_detect":
            table[f"lat_{stage[2:]}_ms"] = table[stage] - table["t_flir_detect"]
    return table


def add_quality(table: pd.DataFrame) -> pd.DataFrame:
    """Flag how far each detection travelled and whether it reached the vehicle."""
    if table.empty:
        return table
    present = [stage for stage in config.STAGE_COLUMNS if stage in table.columns]
    reached = table[present].notna()
    table["stages_reached"] = reached.sum(axis=1)
    # Any vehicle-side stage counts as arrival, not t_ros_inbound alone: that one
    # is joined on payload bytes while the others are joined on detection
    # identity, so a detection can legitimately appear on the J3224 topic with no
    # matching inbound ByteArray. Keying arrival off a single stage would report
    # those as losses when they in fact arrived.
    ros_stages = [
        stage for stage in ("t_ros_inbound", "t_ros_j3224", "t_ros_fused")
        if stage in table.columns
    ]
    table["reached_vehicle"] = (
        table[ros_stages].notna().any(axis=1) if ros_stages else False
    )

    def last_stage(row):
        seen = [stage for stage in present if pd.notna(row[stage])]
        return seen[-1] if seen else None

    table["last_stage_reached"] = table.apply(last_stage, axis=1)
    return table
