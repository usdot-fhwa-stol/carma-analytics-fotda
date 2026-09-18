"""Reader for OBU captures saved as ``tcpdump`` console text rather than pcap.

The 2026-09-14 verification session's OBU files carry a ``.pcap`` extension but
are tcpdump's printed output:

    18:51:52.671839 IP6 80f8:...:2a8.2600 > ff02::1.2600: UDP, length 48
    18:51:52.973719 IP6 80f8:...:2a8.9000 > 80f8:...:2a8.9000: UDP, length 62

Two consequences, both structural:

* **No payload.** Packets cannot be byte-matched against the RSU broadcast or
  the MCAP the way a real pcap allows, so anything built on this is count- and
  time-based. Callers must label it as such -- a latency figure derived from
  ordinal pairing is not the same measurement as one derived from byte identity,
  and the difference is invisible once both are plotted as milliseconds.
* **No date.** tcpdump prints time-of-day only, so the caller supplies the date
  (from ``runs.csv``). The clock is UTC: run 1's first packet is 18:51:52, and
  ``runs.csv`` puts that run at 14:52:00 America/New_York.

Message type is inferred from UDP port and payload length, which is enough to
separate the two flows that matter: outbound BSMs to the ff02::1 multicast group
on port 2600, and inbound SDSMs delivered to port 9000.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# "18:51:52.671839 IP6 <src>.<sport> > <dst>.<dport>: UDP, length 48"
_LINE = re.compile(
    r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}\.\d+)\s+"
    r"IP6?\s+(?P<src>\S+?)\.(?P<sport>\d+)\s+>\s+(?P<dst>\S+?)\.(?P<dport>\d+):\s+"
    r"UDP,\s+length\s+(?P<length>\d+)"
)

# Port the OBU receives V2X messages on, and the link-local multicast group it
# broadcasts to. Both are fixed by the radio's configuration.
OBU_RX_PORT = 9000
V2X_MULTICAST_PORT = 2600
MULTICAST_DST = "ff02::1"


def _classify(dst: str, dport: int, length: int) -> Optional[str]:
    """('BSM'|'SDSM', direction) for a packet, or None if it is neither.

    Lengths are the UDP payload sizes observed for these two flows and are what
    separates them within a port; they are asserted rather than guessed because
    every run shows exactly two distinct lengths (48 and 62).
    """
    if dport == V2X_MULTICAST_PORT and dst.startswith(MULTICAST_DST):
        return "BSM", "outgoing"
    if dport == OBU_RX_PORT:
        return "SDSM", "incoming"
    return None


def parse_tcpdump_text(path, capture_date, tz=timezone.utc) -> List[Dict]:
    """Parse a tcpdump text capture into per-packet dicts.

    ``capture_date`` is a ``datetime.date`` supplying the day the clock-time
    stamps belong to; ``tz`` is the capture host's clock (UTC on these OBUs).
    Returns ``{timestamp, msg_type, direction, dst_port, length}`` with
    ``timestamp`` in epoch seconds, in file order.

    A capture that crosses midnight is handled by rolling the date forward when
    the time-of-day goes backwards.
    """
    packets: List[Dict] = []
    day_offset = 0
    previous_seconds = None

    with open(path, encoding="utf8", errors="ignore") as handle:
        for line in handle:
            match = _LINE.match(line)
            if match is None:
                continue
            seconds = (
                int(match.group("h")) * 3600
                + int(match.group("m")) * 60
                + float(match.group("s"))
            )
            if previous_seconds is not None and seconds < previous_seconds - 1.0:
                day_offset += 1  # wrapped past midnight
            previous_seconds = seconds

            classified = _classify(
                match.group("dst"), int(match.group("dport")), int(match.group("length"))
            )
            if classified is None:
                continue
            msg_type, direction = classified

            midnight = datetime(
                capture_date.year, capture_date.month, capture_date.day, tzinfo=tz
            ) + timedelta(days=day_offset)
            packets.append(
                {
                    "timestamp": midnight.timestamp() + seconds,
                    "msg_type": msg_type,
                    "direction": direction,
                    "dst_port": int(match.group("dport")),
                    "length": int(match.group("length")),
                }
            )
    return packets


def count_by_type(packets: List[Dict], start_sec=None, end_sec=None) -> Dict[str, int]:
    """Per-message-type counts, optionally restricted to a time window."""
    counts: Dict[str, int] = {}
    for packet in packets:
        if start_sec is not None and packet["timestamp"] < start_sec:
            continue
        if end_sec is not None and packet["timestamp"] > end_sec:
            continue
        counts[packet["msg_type"]] = counts.get(packet["msg_type"], 0) + 1
    return counts


def timestamps_of_type(packets: List[Dict], msg_type: str, start_sec=None, end_sec=None) -> List[float]:
    """Sorted epoch-second timestamps of one message type inside a window."""
    return sorted(
        packet["timestamp"]
        for packet in packets
        if packet["msg_type"] == msg_type
        and (start_sec is None or packet["timestamp"] >= start_sec)
        and (end_sec is None or packet["timestamp"] <= end_sec)
    )


def is_tcpdump_text(path) -> bool:
    """True if ``path`` is tcpdump text rather than a binary pcap.

    Lets callers accept either form for the same argument, since these files are
    named ``.pcap`` regardless of which they are.
    """
    try:
        with open(path, "rb") as handle:
            # Long enough to hold a whole tcpdump line including the trailing
            # "UDP, length N" that the pattern needs; IPv6 addresses make these
            # lines well over 100 characters.
            head = handle.read(4096)
    except OSError:
        return False
    if not head:
        return False
    if head[:4] in (
        b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4",
        b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d",
        b"\x0a\x0d\x0d\x0a",
    ):
        return False
    lines = head.decode("utf8", errors="ignore").splitlines()
    # Drop a possibly-truncated final line before testing.
    return any(_LINE.match(line) for line in lines[:-1] or lines)


def infer_capture_date(path, fallback_date=None):
    """Best-effort capture date, from the file's mtime, for callers without runs.csv."""
    if fallback_date is not None:
        return fallback_date
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc).date()
