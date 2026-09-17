"""Extract J2735 message payloads from a classic pcap, without tshark or pycrate.

``correlate_j2735_latency`` gets each frame's timestamp by shelling out to
``tshark`` and validates each message by running it through a pycrate-generated
ASN.1 decoder. Neither is needed for the drop-rate and latency metrics: the
timestamp is already in the pcap record header, and correlation is done on raw
payload bytes, never on decoded field values. So this reader walks the file with
``struct`` and returns the same ``{timestamp, msg_type, payload_hex}`` dicts.

What is given up is the ASN.1 validity check. In exchange the framing check is
made stricter (see ``_find_message``), which rejects the same false positives in
practice: a byte sequence that happens to look like a message id but is not a
real MessageFrame will almost never carry a length that lands exactly on the end
of the UDP payload.

Only classic pcap is read. PCAPNG raises with the ``editcap`` command to run.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional

PCAP_MAGIC_USEC_LE = 0xA1B2C3D4
PCAP_MAGIC_NSEC_LE = 0xA1B23C4D
PCAP_MAGIC_USEC_BE = 0xD4C3B2A1
PCAP_MAGIC_NSEC_BE = 0x4D3CB2A1
PCAPNG_MAGIC = 0x0A0D0D0A

# J2735 DSRCmsgID -> the short name the rest of the pipeline uses. These are the
# two the DT-WZ analysis correlates; add ids here rather than widening the scan.
J2735_MESSAGE_IDS = {
    b"\x00\x29": "SDSM",   # 41, SensorDataSharingMessage
    b"\x00\x14": "BSM",    # 20, BasicSafetyMessage
}

# Alias so pcap message names line up with ByteArray.message_type in the MCAP.
MCAP_TYPE_ALIASES = {"SensorDataSharingMessage": "SDSM"}


def _definite_length(packet: bytes, offset: int):
    """Decode an ASN.1 definite-form length at ``offset`` -> (value, header_len).

    Short form is one byte < 0x80. Long form has the top bit set and the low 7
    bits give how many bytes carry the length. The original SDSM-only reader
    handled short form alone, which is fine for a 60-byte SDSM but silently
    misses any message of 128 bytes or more.
    """
    if offset >= len(packet):
        return None, 0
    first = packet[offset]
    if first < 0x80:
        return first, 1
    count = first & 0x7F
    if count == 0 or count > 4 or offset + 1 + count > len(packet):
        return None, 0
    return int.from_bytes(packet[offset + 1: offset + 1 + count], "big"), 1 + count


def _find_message(packet: bytes, message_ids: Dict[bytes, str]):
    """Return (msg_type, MessageFrame bytes) for a J2735 message in ``packet``.

    A MessageFrame is a 2-byte DSRCmsgID then a definite-form length then the
    body, and it is the last thing in the WSM payload. So a candidate is
    accepted only when its declared length lands exactly on the end of the
    captured packet -- that end-alignment is what makes a bare two-byte scan
    safe. Scanning continues past a near-miss because the id bytes do occur
    inside message bodies.
    """
    for msg_id, msg_type in message_ids.items():
        start = 0
        while True:
            i = packet.find(msg_id, start)
            if i < 0:
                break
            length, header_len = _definite_length(packet, i + 2)
            if length is not None and i + 2 + header_len + length == len(packet):
                return msg_type, packet[i:]
            start = i + 1
    return None, None


def read_classic_pcap(pcap_path):
    """Yield (timestamp_sec, packet_bytes) for every record in a classic pcap."""
    path = Path(pcap_path)
    with open(path, "rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            raise ValueError(f"{path}: truncated pcap header")
        magic = struct.unpack("<I", header[:4])[0]
        if magic == PCAPNG_MAGIC or struct.unpack(">I", header[:4])[0] == PCAPNG_MAGIC:
            raise ValueError(
                f"{path}: PCAPNG is not supported. Convert first with: "
                f"editcap {path} {path.with_suffix('.pcap')}"
            )
        if magic in (PCAP_MAGIC_USEC_LE, PCAP_MAGIC_NSEC_LE):
            endian = "<"
            frac_div = 1e6 if magic == PCAP_MAGIC_USEC_LE else 1e9
        elif magic in (PCAP_MAGIC_USEC_BE, PCAP_MAGIC_NSEC_BE):
            endian = ">"
            frac_div = 1e6 if magic == PCAP_MAGIC_USEC_BE else 1e9
        else:
            raise ValueError(f"{path}: unsupported pcap magic 0x{magic:08x}")

        record_struct = struct.Struct(f"{endian}IIII")
        while True:
            record = handle.read(16)
            if len(record) < 16:
                break
            ts_sec, ts_frac, cap_len, _orig_len = record_struct.unpack(record)
            data = handle.read(cap_len)
            if len(data) < cap_len:
                break  # truncated trailing packet; keep what was read
            yield ts_sec + ts_frac / frac_div, data


def extract_pcap_messages(pcap_path, message_types=None) -> List[Dict]:
    """J2735 messages in one pcap as ``{timestamp, msg_type, payload_hex}`` dicts.

    ``timestamp`` is epoch seconds (float) from the pcap record header.
    ``message_types`` restricts the scan, e.g. ``["SDSM"]``; default is all known.
    Returned in capture order, which is already time order.
    """
    wanted = set(message_types) if message_types else set(J2735_MESSAGE_IDS.values())
    ids = {mid: name for mid, name in J2735_MESSAGE_IDS.items() if name in wanted}
    if not ids:
        raise ValueError(f"No known J2735 message ids for types {sorted(wanted)}")

    messages: List[Dict] = []
    for timestamp, packet in read_classic_pcap(pcap_path):
        msg_type, frame = _find_message(packet, ids)
        if frame is None:
            continue
        messages.append(
            {"timestamp": timestamp, "msg_type": msg_type, "payload_hex": frame.hex()}
        )
    return messages


def extract_pcap_messages_by_direction(pcap_path, message_types=None) -> Dict[str, List[Dict]]:
    """Same as ``extract_pcap_messages`` in the shape the correlators expect.

    The RSU capture is taken on the broadcasting side, so every J2735 message in
    it is outgoing. Direction is not recovered from the link-layer header here --
    nothing in the DT-WZ path needs it, and guessing it from an SLL header whose
    semantics vary by capture host would be worse than declaring it.
    """
    return {"outgoing": extract_pcap_messages(pcap_path, message_types), "incoming": []}


def pcap_time_bounds(pcap_path) -> Optional[tuple]:
    """(first, last) epoch-second timestamps in a pcap, or None if it is empty."""
    first = last = None
    for timestamp, _packet in read_classic_pcap(pcap_path):
        if first is None:
            first = timestamp
        last = timestamp
    return None if first is None else (first, last)
