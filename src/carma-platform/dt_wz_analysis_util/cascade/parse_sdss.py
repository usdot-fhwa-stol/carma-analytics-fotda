"""Parse the sensor_data_sharing_service container log.

Two independent threads are interleaved here, and that distinction is the whole
point of these two stages:

* ``sensor_data_sharing_service.cpp:184`` -- the consumer thread has taken a
  detection off Kafka and inserted it into an in-memory map keyed by object id.
* ``sensor_data_sharing_service.cpp:241`` -- the producer thread, firing on a
  fixed ~100 ms timer, has snapshotted that map into one SDSM and cleared it.

So the gap between them is queueing against a timer, not processing cost, and it
is a sawtooth over the run. They are joined on content, never on log order.

This log is the one source written in local time (EDT); see config.SDSS_UTC_OFFSET_H.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from . import cascade_config as config
from .parse_v2xhub import _as_int, _first_json, _sdsm_identity
from ..portable.timeutil import parse_bracket_ts

# Anchored on message text, not on `file.cpp:line`. The service is rebuilt
# between runs and its line numbers move -- 20260911 logged "Sending SDSM" from
# :241, 20260914 does not. The wording has been stable across both.
A_CONSUMED = "Detected Object List Size"
A_SENDING = "Sending SDSM"
A_DELAY = "Detection delay calculated"

_DELAY = re.compile(r"Detection delay calculated:\s*([0-9.]+)\s*ms")
_LIST_SIZE = re.compile(r"Detected Object List Size (\d+)")


def parse_sdss(path=None) -> dict[str, pd.DataFrame]:
    if path is None:
        raise ValueError("parse_sdss requires the run's sdss log path")
    off = config.SDSS_UTC_OFFSET_H
    consumed, sent, delays = [], [], []

    with open(path, errors="replace") as fh:
        for line in fh:
            ts = parse_bracket_ts(line, off)
            if ts is None:
                continue

            if A_CONSUMED in line:
                obj = _first_json(line)
                if obj is None:
                    continue
                size = _LIST_SIZE.search(line)
                consumed.append(
                    {
                        "t_sdss_consume": ts,
                        "t_flir_detect": _as_int(obj.get("timestamp")),
                        "object_id": _as_int(obj.get("objectId")),
                        "sdss_list_size": int(size.group(1)) if size else None,
                    }
                )

            elif A_SENDING in line:
                obj = _first_json(line)
                if obj is None:
                    continue
                object_id, sdsm_ms, meas, msg_cnt = _sdsm_identity(obj)
                sent.append(
                    {
                        "t_sdss_send": ts,
                        "object_id": object_id,
                        "sdsm_epoch_ms": sdsm_ms,
                        "measurement_time": meas,
                        "sdsm_msg_cnt": msg_cnt,
                        # Recovered detection identity, same identity the pc2
                        # chain derives -- used to cross-check both.
                        "t_flir_detect": (sdsm_ms - meas) if (sdsm_ms and meas is not None) else None,
                    }
                )

            elif A_DELAY in line:
                m = _DELAY.search(line)
                if m:
                    delays.append({"t_log": ts, "sdss_reported_latency_ms": float(m.group(1))})

    return {
        "consumed": pd.DataFrame(consumed),
        "sent": pd.DataFrame(sent),
        "reported_delay": pd.DataFrame(delays),
    }
