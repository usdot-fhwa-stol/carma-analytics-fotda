"""Read an OBU radio capture, in whichever of the two forms it was saved.

The OBU's capture format has changed between sessions and the files are named
``.pcap`` either way:

* **binary pcap** (2026-09-15 onward) -- carries payload bytes, so a received
  message can be matched to the RSU's broadcast by exact identity.
* **tcpdump console text** (2026-09-14) -- timestamps and lengths only. Messages
  can be counted and timed but not identified, so anything built on it has to
  fall back to matching by time.

Both come back as the same list of dicts, with ``payload_hex`` set to ``None``
for the text form and ``payloads_available`` telling the caller which it got.
That flag matters and should be carried into the output: a latency derived from
ordinal pairing is a weaker measurement than one derived from byte identity, and
once both are printed as milliseconds the difference is invisible.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import pcap_reader, tcpdump_text


def read_obu_capture(path, capture_date=None) -> Dict:
    """Read an OBU capture -> ``{"payloads_available": bool, "messages": [...]}``.

    Each message is ``{timestamp, msg_type, payload_hex}`` with ``timestamp`` in
    epoch seconds. ``capture_date`` supplies the day for the text form, which
    records a time of day and no date; it is ignored for binary pcap, which
    carries absolute timestamps of its own.
    """
    if tcpdump_text.is_tcpdump_text(path):
        if capture_date is None:
            raise ValueError(
                f"{path} is a tcpdump text capture and needs a capture_date: "
                f"it records times of day with no date"
            )
        messages = [
            {"timestamp": packet["timestamp"], "msg_type": packet["msg_type"], "payload_hex": None}
            for packet in tcpdump_text.parse_tcpdump_text(path, capture_date)
        ]
        return {"payloads_available": False, "messages": messages}

    return {"payloads_available": True, "messages": pcap_reader.extract_pcap_messages(path)}


def messages_of_type(capture: Dict, msg_type: str, start_sec=None, end_sec=None) -> List[Dict]:
    """Messages of one type inside an optional time window, in time order."""
    selected = [
        message for message in capture["messages"]
        if message["msg_type"] == msg_type
        and (start_sec is None or message["timestamp"] >= start_sec)
        and (end_sec is None or message["timestamp"] <= end_sec)
    ]
    selected.sort(key=lambda message: message["timestamp"])
    return selected


def count_by_type(capture: Dict, start_sec=None, end_sec=None) -> Dict[str, int]:
    """Per-message-type counts inside an optional time window."""
    counts: Dict[str, int] = {}
    for message in capture["messages"]:
        if start_sec is not None and message["timestamp"] < start_sec:
            continue
        if end_sec is not None and message["timestamp"] > end_sec:
            continue
        counts[message["msg_type"]] = counts.get(message["msg_type"], 0) + 1
    return counts
