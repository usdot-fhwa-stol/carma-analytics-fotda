"""Parse the pc2 and pc1 V2XHub text logs into per-stage event tables.

Both logs interleave many plugins on one stream, so every extractor is keyed on
a ``file.cpp (line)`` anchor rather than on message text, which keeps it stable
against log-level and wording changes.

Timestamps: the bracketed wall clock is UTC. Where a line embeds an epoch value
(a TMX ``header.timestamp``, or the detection's own ``timestamp``) we keep both
and prefer the embedded one, since it is set by the producing code rather than
by the logger.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from . import cascade_config as config
from ..portable.pairing import pair_forward
from ..portable.timeutil import parse_bracket_ts, sdsm_timestamp_to_epoch_ms

# --- line anchors -----------------------------------------------------------
A_FLIR = "FLIRCameraDriverPlugin.cpp (133)"
A_STREETS_SDO_RX = "CARMAStreetsPlugin.cpp (716)"
A_KAFKA_PRODUCE = "kafka_producer_worker.cpp (162)"
A_STREETS_SDSM_RX = "CARMAStreetsPlugin.cpp (673)"
A_STREETS_ENCODE = "CARMAStreetsPlugin.cpp (697)"
A_TENA_BUS_RX = "TenaV2Xplugin.cpp (163)"
A_TENA_SEND = "SendMessage.cpp (28)"
A_OBSERVER_RX = "Observer.cpp (42)"
A_PLUGIN_SENDING = "PluginClient.h (129)"
A_IMMEDIATE_FWD = "ImmediateForwardPlugin.cpp (326)"

_JSON_OBJ = re.compile(r"\{.*\}\s*$")
_PAYLOAD_HEX = re.compile(r'"payload"\s*:\s*"([0-9a-fA-F]+)"')
_HEADER_TS = re.compile(r'"timestamp"\s*:\s*"?(\d+)"?')


def _first_json(line: str) -> dict | None:
    """Decode the trailing JSON object on a log line, if any."""
    m = _JSON_OBJ.search(line)
    if m is None:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _as_int(value) -> int | None:
    # The TMX-side JSON stringifies every number; the Kafka-side JSON does not.
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sdsm_identity(sdsm: dict) -> tuple[int | None, float | None, int | None, int | None]:
    """(object_id, sdsm_epoch_ms, measurement_time, msg_cnt) from an SDSM dict.

    Only the first detected object is used: every SDSM in this run carries
    exactly one, because the service's detection map is keyed by object id and
    only two pedestrians were ever tracked.
    """
    try:
        sdsm_ms = sdsm_timestamp_to_epoch_ms(sdsm["sdsm_time_stamp"])
    except (KeyError, TypeError, ValueError):
        return None, None, None, None
    objects = sdsm.get("objects") or []
    if not objects:
        return None, sdsm_ms, None, _as_int(sdsm.get("msg_cnt"))
    common = objects[0].get("detected_object_data", {}).get("detected_object_common_data", {})
    return (
        _as_int(common.get("object_id")),
        sdsm_ms,
        _as_int(common.get("measurement_time")),
        _as_int(sdsm.get("msg_cnt")),
    )


def parse_pc2(path=None) -> dict[str, pd.DataFrame]:
    """Extract every pc2 stage. Returns a dict of event tables."""
    if path is None:
        raise ValueError("parse_pc2 requires the run's pc2 V2XHub log path")
    flir, streets_rx, kafka_prod, sdsm_rx, encode, tena_rx, tena_tx = ([] for _ in range(7))

    with open(path, errors="replace") as fh:
        for line in fh:
            ts = parse_bracket_ts(line, config.V2XHUB_UTC_OFFSET_H)
            if ts is None:
                continue

            if A_FLIR in line:
                obj = _first_json(line)
                if obj is not None:
                    flir.append(
                        {
                            "t_flir_log": ts,
                            "t_flir_detect": _as_int(obj.get("timestamp")),
                            "object_id": _as_int(obj.get("objectId")),
                            "sensor_id": obj.get("sensorId"),
                        }
                    )

            elif A_STREETS_SDO_RX in line:
                obj = _first_json(line)
                if obj is not None:
                    streets_rx.append(
                        {
                            "t_streets_rx": ts,
                            "t_flir_detect": _as_int(obj.get("timestamp")),
                            "object_id": _as_int(obj.get("objectId")),
                        }
                    )

            elif A_KAFKA_PRODUCE in line:
                # This anchor also carries BSMs; only detections have objectId.
                if '"objectId"' not in line:
                    continue
                obj = _first_json(line)
                if obj is not None:
                    kafka_prod.append(
                        {
                            "t_kafka_produce": ts,
                            "t_flir_detect": _as_int(obj.get("timestamp")),
                            "object_id": _as_int(obj.get("objectId")),
                        }
                    )

            elif A_STREETS_SDSM_RX in line:
                obj = _first_json(line)
                if obj is None:
                    continue
                object_id, sdsm_ms, meas, msg_cnt = _sdsm_identity(obj)
                sdsm_rx.append(
                    {
                        "t_streets_sdsm_rx": ts,
                        "object_id": object_id,
                        "sdsm_epoch_ms": sdsm_ms,
                        "measurement_time": meas,
                        "sdsm_msg_cnt": msg_cnt,
                    }
                )

            elif A_STREETS_ENCODE in line:
                hexm = _PAYLOAD_HEX.search(line)
                tsm = _HEADER_TS.search(line)
                if hexm is not None:
                    encode.append(
                        {
                            "t_streets_encode": ts,
                            "t_streets_encode_hdr": float(tsm.group(1)) if tsm else None,
                            "sdsm_uper_hex": hexm.group(1).lower(),
                        }
                    )

            elif A_TENA_BUS_RX in line:
                if "type: SDSM" in line:
                    tena_rx.append({"t_pc2_tena_rx": ts})

            elif A_TENA_SEND in line:
                tena_tx.append({"t_pc2_tena_tx": ts})

    return {
        "flir": pd.DataFrame(flir),
        "streets_rx": pd.DataFrame(streets_rx),
        "kafka_produce": pd.DataFrame(kafka_prod),
        "sdsm_rx": pd.DataFrame(sdsm_rx),
        "encode": pd.DataFrame(encode),
        "tena_rx": pd.DataFrame(tena_rx),
        "tena_tx": pd.DataFrame(tena_tx),
    }


def build_pc2_sdsm_chain(ev: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stitch the pc2 SDSM stages into one table keyed by the UPER payload.

    ``CARMAStreetsPlugin`` logs the consumed SDSM JSON (673) and the UPER
    hexstring it encoded from it (697) on consecutive lines from the same Kafka
    consumer callback, so a bounded forward pair is what links the SDSM's
    identity to the bytes that travel the rest of the way. The two TENA anchors
    log no content at all and are paired the same way.
    """
    sdsm_rx = ev["sdsm_rx"].sort_values("t_streets_sdsm_rx").reset_index(drop=True)
    encode = ev["encode"].sort_values("t_streets_encode").reset_index(drop=True)
    tena_rx = ev["tena_rx"].sort_values("t_pc2_tena_rx").reset_index(drop=True)
    tena_tx = ev["tena_tx"].sort_values("t_pc2_tena_tx").reset_index(drop=True)

    chain = sdsm_rx.copy()
    idx = pair_forward(chain["t_streets_sdsm_rx"].tolist(), encode["t_streets_encode"].tolist())
    for col in ("t_streets_encode", "t_streets_encode_hdr", "sdsm_uper_hex"):
        chain[col] = [encode.at[i, col] if i is not None else None for i in idx]

    have = chain["t_streets_encode"].notna()
    anchor = chain.loc[have, "t_streets_encode"].tolist()

    idx = pair_forward(anchor, tena_rx["t_pc2_tena_rx"].tolist())
    chain.loc[have, "t_pc2_tena_rx"] = [
        tena_rx.at[i, "t_pc2_tena_rx"] if i is not None else None for i in idx
    ]

    # Pair the TENA send off the bus-read it follows, not off the encode.
    rx_present = chain["t_pc2_tena_rx"].notna()
    idx = pair_forward(
        chain.loc[rx_present, "t_pc2_tena_rx"].tolist(), tena_tx["t_pc2_tena_tx"].tolist()
    )
    chain.loc[rx_present, "t_pc2_tena_tx"] = [
        tena_tx.at[i, "t_pc2_tena_tx"] if i is not None else None for i in idx
    ]

    # The service stamps each object with how stale it was at send time, which
    # recovers the exact originating detection without any fuzzy matching.
    chain["t_flir_detect"] = chain["sdsm_epoch_ms"] - chain["measurement_time"]
    return chain


def parse_pc1(path=None) -> pd.DataFrame:
    """Extract the pc1 SDSM stages, keyed by the UPER payload."""
    if path is None:
        raise ValueError("parse_pc1 requires the run's pc1 V2XHub log path")
    observer, bus, fwd = [], [], []

    with open(path, errors="replace") as fh:
        for line in fh:
            ts = parse_bracket_ts(line, config.V2XHUB_UTC_OFFSET_H)
            if ts is None:
                continue

            if A_OBSERVER_RX in line:
                # Observer logs the type but truncates the payload to "[".
                if "type: SDSM" in line:
                    observer.append({"t_pc1_tena_rx_log": ts})

            elif A_PLUGIN_SENDING in line:
                if '"subtype":"SDSM"' not in line:
                    continue
                hexm = _PAYLOAD_HEX.search(line)
                tsm = _HEADER_TS.search(line)
                if hexm is not None:
                    bus.append(
                        {
                            "t_pc1_bus": ts,
                            # Header stamp is set by TenaV2XPlugin when it lifts
                            # the message off TENA, i.e. the arrival instant.
                            "t_pc1_tena_rx": float(tsm.group(1)) if tsm else None,
                            "sdsm_uper_hex": hexm.group(1).lower(),
                        }
                    )

            elif A_IMMEDIATE_FWD in line:
                if "TmxType: SDSM" in line:
                    fwd.append({"t_immediate_fwd": ts})

    bus_df = pd.DataFrame(bus).sort_values("t_pc1_bus").reset_index(drop=True)
    fwd_df = pd.DataFrame(fwd).sort_values("t_immediate_fwd").reset_index(drop=True)
    obs_df = pd.DataFrame(observer).sort_values("t_pc1_tena_rx_log").reset_index(drop=True)

    if bus_df.empty:
        return bus_df

    idx = pair_forward(bus_df["t_pc1_bus"].tolist(), fwd_df["t_immediate_fwd"].tolist())
    bus_df["t_immediate_fwd"] = [
        fwd_df.at[i, "t_immediate_fwd"] if i is not None else None for i in idx
    ]

    # Keep the Observer line time as a cross-check on the header stamp.
    idx = pair_forward(
        obs_df["t_pc1_tena_rx_log"].tolist(),
        bus_df["t_pc1_bus"].tolist(),
        back_tolerance_ms=0.0,
    )
    log_times: list[float | None] = [None] * len(bus_df)
    for src, dst in enumerate(idx):
        if dst is not None:
            log_times[dst] = obs_df.at[src, "t_pc1_tena_rx_log"]
    bus_df["t_pc1_tena_rx_log"] = log_times
    return bus_df
