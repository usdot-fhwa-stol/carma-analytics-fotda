"""Reader for ``kafka-console-consumer`` dumps, tolerant of the field delimiter.

Two dump styles are in circulation and they are not interchangeable:

    CreateTime:1757520202123\t{"objectId":1,...}                       tab
    CreateTime:1788530425666|Partition:0|Offset:46063|null|{...}       pipe

The tab form is what ``--property print.timestamp=true`` emits on its own; the
pipe form is what you get once ``print.partition``/``print.offset``/``print.key``
are added and ``key.separator`` is set. The 2026-09-14 verification session used
the pipe form.

Splitting on the wrong delimiter does not fail loudly. The original parser took
everything up to the first tab and stripped non-digits from it; on a pipe-
delimited line that is the *whole line*, so the partition, the offset and every
digit inside the JSON get concatenated onto the timestamp:

    CreateTime:1788530425666|Partition:0|Offset:46063|null|{"timestamp":...}
    -> 178853042566604606317885304256201177      (36 digits, not 13)

which silently becomes a timestamp ~10^23 years in the future, so every latency
is astronomically wrong and every time-window filter matches nothing. Hence:
find the JSON body first, and read the timestamp only from the header in front
of it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

# The epoch-ms value in the "CreateTime:<ms>" header, stopping at whichever
# delimiter follows -- tab, pipe, space, or the start of the JSON body.
_CREATE_TIME = re.compile(r"CreateTime:\s*(\d{10,13})")


def _split_record(line: str):
    """(create_time_ms, json_text) for one dump line, or None if it isn't a record."""
    if not line.startswith("CreateTime"):
        return None
    match = _CREATE_TIME.match(line)
    if match is None:
        return None
    brace = line.find("{")
    if brace < 0:
        # Header with no body on this line; the body may follow on continuation lines.
        return int(match.group(1)), ""
    return int(match.group(1)), line[brace:]


def parse_kafka_log_records(kafka_log_path) -> List[Dict]:
    """Parse a Kafka console dump into one dict per record.

    Each dict is the decoded JSON body plus ``create_time_ms`` (the broker's
    timestamp). Records whose body spans several lines are reassembled. Bodies
    that fail to decode are counted and skipped rather than raising, since a
    dump truncated mid-record is common and should not lose the whole file.
    """
    records: List[Dict] = []
    decode_failures = 0
    pending_time = None
    pending_body: List[str] = []

    def flush():
        nonlocal decode_failures, pending_time, pending_body
        if pending_time is None:
            return
        body = "".join(pending_body).strip()
        if body:
            try:
                record = json.loads(body)
                if isinstance(record, dict):
                    record["create_time_ms"] = pending_time
                    records.append(record)
                else:
                    decode_failures += 1
            except json.JSONDecodeError:
                decode_failures += 1
        pending_time = None
        pending_body = []

    with open(kafka_log_path, encoding="utf8", errors="ignore") as handle:
        for line in handle:
            split = _split_record(line)
            if split is None:
                if pending_time is not None:
                    pending_body.append(line)
                continue
            flush()
            pending_time, body = split
            pending_body = [body] if body else []
    flush()

    if decode_failures:
        print(f"Warning: {decode_failures} undecodable record(s) in {Path(kafka_log_path).name}")
    return records


def parse_kafka_log_timestamps(kafka_log_path, timestamp_field: str = "timestamp"):
    """Sorted epoch-ms timestamps of every record, falling back to the broker time."""
    import numpy as np

    records = parse_kafka_log_records(kafka_log_path)
    return np.sort(
        np.array([float(r.get(timestamp_field, r["create_time_ms"])) for r in records], dtype=float)
    )


def window_records(records: List[Dict], start_ms: float, end_ms: float,
                   timestamp_field: str = "timestamp") -> List[Dict]:
    """Records whose timestamp falls inside [start_ms, end_ms].

    These dumps hold the broker's whole retention -- the 2026-09-14 export opens
    on 2026-09-04 data and covers 56193 detections against ~2500 in any one run
    -- and object ids are recycled across that span. Every per-run metric has to
    window before it groups by object id, or detections from a different day join
    onto this run's SDSMs.
    """
    return [
        r for r in records
        if start_ms <= float(r.get(timestamp_field, r["create_time_ms"])) <= end_ms
    ]
