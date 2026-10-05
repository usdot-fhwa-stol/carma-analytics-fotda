#!/usr/bin/env python3
"""
Correlates J2735 messages between two pcap captures of the same broadcast
stream at different points (e.g. an OBU's radio-receive interface vs. its
ethernet-out interface to a downstream host) to compute per-message latency
and drop/staleness rate.

Message location within each UDP payload is IEEE 1609.2-envelope-aware: it
scans for the `unsecuredData` marker (0x03 0x80) followed by a valid OER
length field, then reads the J2735 message ID at the computed content
offset. This is deliberately NOT a blind hex substring search for markers
like "0012"/"0013" - an earlier version of this script (and decodeJ2735.py,
which it was based on) used that approach and it silently missed MAP
messages entirely, because a coincidental "0012" byte pair elsewhere in a
packet's payload would be found first and fail validation, while the real
MAP marker - only locatable by walking the actual envelope structure - was
never reached. Payloads with no unsecuredData marker (a WSMP header or a
vendor envelope in front of the MessageFrame) fall back to scanning for a
bare `00 <msgid>` MessageFrame that actually UPER-decodes. The
pcap/link-layer/UDP parsing and envelope decoding here are ported from
carma-platform/engineering_tools/check_v2x_security_direction.py, which also
classifies packet direction (incoming vs. outgoing) using the
Linux-cooked-capture SLL header; that distinction matters for correlation,
see below.

Direction labels describe the capturing interface (it received vs. sent the
packet), so which labels hold the sender's and receiver's copy of a message
depends on where the two captures were taken (--topology):

  same-device (default): two interfaces of one device, e.g. an OBU's radio
    (tx) and its ethernet link to the host (rx). Messages pass through the
    device: radio `incoming` is matched against host-link `incoming`/`other`,
    and host requests (rx `outgoing`) against the radio broadcast
    (tx `outgoing`).
  cross-device: each device's own radio capture, e.g. RSU and OBU. Messages
    go from one device to the other: sender `outgoing` is matched against
    receiver `incoming`, in both directions, and the clock offset between
    the two devices is estimated from the two directions' latencies.

Each message's timestamp is sourced from `tshark -e frame.time_epoch` for
the packet it was actually found in (matched back by packet number), since
the pure-Python pcap reader used for envelope parsing doesn't expose
per-packet timestamps.

Usage:
    python3 correlate_j2735_latency.py --tx-pcap earlier.pcap --rx-pcap later.pcap
    python3 correlate_j2735_latency.py --topology cross-device --tx-pcap rsu.pcap --rx-pcap obu.pcap
"""
import contextlib
import io
import json
import re
import statistics
import struct
import subprocess
import sys
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Optional
from binascii import unhexlify

sys.path.insert(0, str(Path(__file__).resolve().parent))
import J2735_201603_2023_06_22  # noqa: E402
from correlation_plots import plot_flow, find_duplicate_content_types, windowed_count_report  # noqa: E402

DROP_LATENCY_THRESHOLD_MS = 200  # matches v2xhub_messaging_performance_analyzer.py's threshold

DLT_EN10MB = 1
DLT_LINUX_SLL = 113
ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_IPV6 = 0x86DD
ETHERTYPE_VLAN = 0x8100
ETHERTYPE_WSMP = 0x88DC

# SAE J2735 DSRCmsgID values worth identifying by name.
J2735_MESSAGE_NAMES = {
    18: "MAP",
    19: "SPAT",
    20: "BSM",
    27: "SRM",
    28: "SSM",
    29: "TIM",
    30: "SSM",
    31: "TIM",
    32: "PSM",
    41: "SDSM",
}

# CARMA Platform's own mobility messages, carried under PSID 0xBFEE per
# wave.json in v2x-ros-driver. Not part of standard SAE J2735 - they won't
# UPER-decode against the J2735 ASN.1 module below, so message ID
# recognition alone is used for these (no decode-validity check).
CARMA_MOBILITY_MESSAGE_IDS = {240, 241, 242, 243}
J2735_MESSAGE_NAMES.update({
    240: "MobilityRequest",
    241: "MobilityResponse",
    242: "MobilityPath",
    243: "MobilityOperation",
})


@dataclass
class PcapPacket:
    number: int
    data: bytes


@dataclass
class UdpPacket:
    payload: bytes
    direction: str


def read_classic_pcap(path: Path) -> Iterator[PcapPacket]:
    """Minimal classic-pcap reader. Stops (rather than raising) on a
    truncated trailing packet, yielding everything read so far - real
    capture files from flaky links are sometimes cut short mid-packet."""
    with path.open("rb") as stream:
        global_header = stream.read(24)
        if len(global_header) != 24:
            return
        magic = global_header[:4]
        if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            endian = "<"
        elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            endian = ">"
        elif magic == b"\x0a\x0d\x0d\x0a":
            raise ValueError("PCAPNG is not supported. Convert first with: editcap input.pcapng output.pcap")
        else:
            raise ValueError(f"Unsupported PCAP magic number: {magic.hex()}")

        packet_number = 0
        while True:
            packet_header = stream.read(16)
            if len(packet_header) != 16:
                return
            _, _, captured_length, _ = struct.unpack(endian + "IIII", packet_header)
            packet_data = stream.read(captured_length)
            if len(packet_data) != captured_length:
                print(f"WARNING: {path}: packet {packet_number + 1} is truncated - "
                      f"stopping here, using partial results", file=sys.stderr)
                return
            packet_number += 1
            yield PcapPacket(packet_number, packet_data)


def parse_udp_with_linktype(data: bytes, linktype: int, wsmp_direction: str = "unknown") -> Optional[UdpPacket]:
    direction = "unknown"
    if linktype == DLT_EN10MB:
        if len(data) < 14:
            return None
        ethertype = struct.unpack("!H", data[12:14])[0]
        offset = 14
        if ethertype == ETHERTYPE_VLAN:
            if len(data) < 18:
                return None
            ethertype = struct.unpack("!H", data[16:18])[0]
            offset = 18
    elif linktype == DLT_LINUX_SLL:
        if len(data) < 16:
            return None
        packet_type = struct.unpack("!H", data[0:2])[0]
        ethertype = struct.unpack("!H", data[14:16])[0]
        offset = 16
        direction = {
            0: "incoming", 1: "incoming-broadcast", 2: "incoming-multicast",
            3: "incoming-otherhost", 4: "outgoing",
        }.get(packet_type, f"sll-type-{packet_type}")
    else:
        return None

    # --wsmp-direction only fills in for links with no direction bit (plain Ethernet).
    if ethertype == ETHERTYPE_WSMP:
        return UdpPacket(payload=data[offset:],
                         direction=wsmp_direction if direction == "unknown" else direction)

    if ethertype == ETHERTYPE_IPV4:
        if len(data) < offset + 20:
            return None
        version_ihl = data[offset]
        if version_ihl >> 4 != 4:
            return None
        ihl = (version_ihl & 0x0F) * 4
        if ihl < 20 or len(data) < offset + ihl or data[offset + 9] != 17:
            return None
        udp_offset = offset + ihl
    elif ethertype == ETHERTYPE_IPV6:
        if len(data) < offset + 40:
            return None
        if data[offset] >> 4 != 6:
            return None
        next_header = data[offset + 6]
        udp_offset = offset + 40
        while next_header in (0, 43, 44, 60):
            if len(data) < udp_offset + 8:
                return None
            if next_header == 44:
                next_header = data[udp_offset]
                udp_offset += 8
            else:
                next_header = data[udp_offset]
                udp_offset += (data[udp_offset + 1] + 1) * 8
        if next_header != 17:
            return None
    else:
        return None

    if len(data) < udp_offset + 8:
        return None
    _, _, udp_length, _ = struct.unpack("!HHHH", data[udp_offset:udp_offset + 8])
    if udp_length < 8:
        return None
    payload_end = min(len(data), udp_offset + udp_length)
    return UdpPacket(payload=data[udp_offset + 8:payload_end], direction=direction)


def read_link_types(path: Path):
    with path.open("rb") as stream:
        header = stream.read(24)
    magic = header[:4]
    endian = "<" if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
    return struct.unpack(endian + "I", header[20:24])[0]


def decode_oer_length(data: bytes, offset: int):
    if offset >= len(data):
        return None
    first = data[offset]
    if first < 0x80:
        return first, offset + 1
    byte_count = first & 0x7F
    if byte_count == 0 or byte_count > 4:
        return None
    end = offset + 1 + byte_count
    if end > len(data):
        return None
    return int.from_bytes(data[offset + 1:end], "big"), end


def find_message_offset(payload: bytes):
    """Scan for the IEEE 1609.2 unsecuredData envelope (03 80 <OER length>)
    and return (content_offset, message_id, message_name) for the first
    recognizable J2735 message found, or None."""
    for offset in range(max(0, len(payload) - 4)):
        if payload[offset:offset + 2] != b"\x03\x80":
            continue
        decoded = decode_oer_length(payload, offset + 2)
        if decoded is None:
            continue
        declared_length, content_offset = decoded
        if content_offset + 2 > len(payload):
            continue
        if payload[content_offset] != 0:
            continue
        message_id = payload[content_offset + 1]
        if message_id not in J2735_MESSAGE_NAMES:
            continue
        if declared_length > len(payload) - content_offset:
            continue
        return content_offset, message_id, J2735_MESSAGE_NAMES[message_id]
    return None

def find_raw_j2735_message(payload: bytes):
    """Fallback for payloads with no 1609.2 unsecuredData marker: a bare
    J2735 MessageFrame (00 <DSRCmsgID> ...) at any offset, e.g. behind a WSMP
    header or a vendor envelope. A coincidental 00 <id> byte pair is common,
    so a candidate only counts if it UPER-decodes; CARMA mobility messages
    can't be decoded against J2735, so they are only accepted at offset 0."""
    frame_asn1 = J2735_201603_2023_06_22.DSRC.MessageFrame
    for offset in range(max(0, len(payload) - 1)):
        if payload[offset] != 0:
            continue
        message_id = payload[offset + 1]
        if message_id not in J2735_MESSAGE_NAMES:
            continue
        if message_id in CARMA_MOBILITY_MESSAGE_IDS:
            if offset == 0:
                return 0, message_id, J2735_MESSAGE_NAMES[message_id]
            continue
        try:
            # pycrate prints diagnostics on failed decodes; probing offsets would flood stdout.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                frame_asn1.from_uper(payload[offset:])
        except Exception:
            continue
        return offset, message_id, J2735_MESSAGE_NAMES[message_id]
    return None

def convert_bytes(obj):
    if isinstance(obj, dict):
        return {k: convert_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_bytes(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.hex()
    return obj


def get_frame_timestamps(pcap_path):
    proc = subprocess.run(
        ["tshark", "-r", str(pcap_path), "-Tfields", "-e", "frame.number", "-e", "frame.time_epoch"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"WARNING: tshark exited {proc.returncode} reading {pcap_path} "
              f"(likely truncated capture) - using partial output. stderr: {proc.stderr.strip()}",
              file=sys.stderr)
    ts_by_number = {}
    for line in proc.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) == 2 and parts[1]:
            ts_by_number[int(parts[0])] = float(parts[1])
    return ts_by_number


def extract_udp_messages(pcap_path, wsmp_direction="unknown"):
    """Extracts J2735 messages from a raw-UDP, 1609.2-enveloped capture
    (e.g. an RSU broadcast-receive interface). Returns (by_direction,
    decode_fail_count). by_direction has keys 'incoming', 'outgoing',
    'other', each a list of dicts with timestamp (float seconds or None),
    msg_type (str), payload_hex (str, from the message marker onward)."""
    frame_asn1 = J2735_201603_2023_06_22.DSRC.MessageFrame
    linktype = read_link_types(Path(pcap_path))
    ts_by_number = get_frame_timestamps(pcap_path)

    by_direction = {"incoming": [], "outgoing": [], "other": []}
    decode_fail = 0

    for packet in read_classic_pcap(Path(pcap_path)):
        udp = parse_udp_with_linktype(packet.data, linktype, wsmp_direction)
        if udp is None:
            continue
        found = find_message_offset(udp.payload)

        if found is None:
            found = find_raw_j2735_message(udp.payload)

        if found is None:
            continue

        content_offset, message_id, message_name = found
        data_hex = udp.payload[content_offset:].hex()

        if message_id not in CARMA_MOBILITY_MESSAGE_IDS:
            try:
                frame_asn1.from_uper(unhexlify(data_hex))
                convert_bytes(frame_asn1())  # forces full decode, validates it's well-formed
            except Exception:
                decode_fail += 1
                continue

        entry = {
            "timestamp": ts_by_number.get(packet.number),
            "msg_type": message_name,
            "payload_hex": data_hex,
        }
        if udp.direction.startswith("incoming"):
            by_direction["incoming"].append(entry)
        elif udp.direction.startswith("outgoing"):
            by_direction["outgoing"].append(entry)
        else:
            by_direction["other"].append(entry)

    return by_direction, decode_fail


def extract_mqtt_messages(pcap_path):
    """Extracts J2735 messages from an MQTT-over-TCP capture, as used by
    Ettifos-vendor OBUs on their ethernet-facing interface. Uses tshark's
    mqtt dissector (handles TCP reassembly) rather than a hand-rolled TCP
    parser. MQTT payloads here are raw J2735 MessageFrame bytes
    (00 <msgid> ...) with no 1609.2 envelope wrapper. Topic naming encodes
    direction: '/ind/' = indication delivered from the OBU to the host
    (role: incoming); '/req/' = the host's request for the OBU to transmit
    (role: outgoing). Returns (by_direction, decode_fail_count), same shape
    as extract_udp_messages."""
    frame_asn1 = J2735_201603_2023_06_22.DSRC.MessageFrame
    proc = subprocess.run(
        ["tshark", "-r", str(pcap_path), "-Y", "mqtt.msgtype==3",
         "-Tfields", "-e", "frame.time_epoch", "-e", "mqtt.topic", "-e", "mqtt.msg"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"WARNING: tshark exited {proc.returncode} reading {pcap_path} for MQTT - "
              f"using partial output. stderr: {proc.stderr.strip()}", file=sys.stderr)

    by_direction = {"incoming": [], "outgoing": [], "other": []}
    decode_fail = 0

    for line in proc.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) != 3 or not parts[0] or not parts[2]:
            continue
        ts_str, topics_field, msgs_field = parts
        # A single TCP segment can carry multiple MQTT PUBLISH messages;
        # tshark joins repeated field occurrences within one frame with
        # commas, 1:1 between the topic and payload fields.
        topics = topics_field.split(',')
        msgs = msgs_field.split(',')
        if len(topics) != len(msgs):
            continue
        for topic, data_hex in zip(topics, msgs):
            if len(data_hex) < 4 or data_hex[0:2] != "00":
                continue
            try:
                message_id = int(data_hex[2:4], 16)
            except ValueError:
                continue
            if message_id not in J2735_MESSAGE_NAMES:
                continue
            message_name = J2735_MESSAGE_NAMES[message_id]

            if message_id not in CARMA_MOBILITY_MESSAGE_IDS:
                try:
                    frame_asn1.from_uper(unhexlify(data_hex))
                    convert_bytes(frame_asn1())
                except Exception:
                    decode_fail += 1
                    continue

            entry = {
                "timestamp": float(ts_str),
                "msg_type": message_name,
                "payload_hex": data_hex,
            }
            if "/ind/" in topic:
                by_direction["incoming"].append(entry)
            elif "/req/" in topic:
                by_direction["outgoing"].append(entry)
            else:
                by_direction["other"].append(entry)

    return by_direction, decode_fail


def extract_commsignia_request_messages(pcap_path):
    """Extracts J2735 messages from a Commsignia-vendor "Tx Request"
    channel: a plain-ASCII, newline-separated key=value UDP protocol the
    host uses to ask the OBU to broadcast a message, distinct from the
    1609.2-enveloped UDP traffic on the OBU's normal broadcast-forwarding
    port. Looks like:

        Version=0.7
        Type=BSM
        PSID=0020
        Priority=6
        ...
        Payload=<hex, raw unsecured J2735 MessageFrame - no 1609.2 envelope>

    Not tied to a specific port number, since it may vary by deployment -
    detected by content (presence of 'Type=' and 'Payload=' lines) instead.
    Always classified as 'outgoing' (a host request for the OBU to
    transmit), the same role as an MQTT '/req/' publish. Returns
    (by_direction, decode_fail_count), same shape as extract_udp_messages."""
    frame_asn1 = J2735_201603_2023_06_22.DSRC.MessageFrame
    proc = subprocess.run(
        ["tshark", "-r", str(pcap_path), "-Y", "udp", "-Tfields", "-e", "frame.time_epoch", "-e", "data.data"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"WARNING: tshark exited {proc.returncode} reading {pcap_path} for Commsignia Tx Requests - "
              f"using partial output. stderr: {proc.stderr.strip()}", file=sys.stderr)

    by_direction = {"incoming": [], "outgoing": [], "other": []}
    decode_fail = 0

    for line in proc.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        ts_str, data_field = parts
        try:
            text = unhexlify(data_field).decode("ascii")
        except (ValueError, UnicodeDecodeError):
            continue
        if "Type=" not in text or "Payload=" not in text:
            continue

        fields = {}
        for text_line in text.splitlines():
            key, sep, value = text_line.partition("=")
            if sep:
                fields[key] = value
        message_name = fields.get("Type")
        data_hex = fields.get("Payload", "")
        if message_name not in J2735_MESSAGE_NAMES.values() or not data_hex:
            continue
        message_id = next(k for k, v in J2735_MESSAGE_NAMES.items() if v == message_name)

        if message_id not in CARMA_MOBILITY_MESSAGE_IDS:
            try:
                frame_asn1.from_uper(unhexlify(data_hex))
                convert_bytes(frame_asn1())
            except Exception:
                decode_fail += 1
                continue

        by_direction["outgoing"].append({
            "timestamp": float(ts_str),
            "msg_type": message_name,
            "payload_hex": data_hex,
        })

    return by_direction, decode_fail


def extract_messages(pcap_path, wsmp_direction="unknown"):
    """Extracts J2735 messages from a pcap, trying all three known
    ethernet-facing formats, since different OBU vendors (and different
    channels on the same vendor) expose messages differently: raw UDP with
    a 1609.2 envelope (broadcast-forwarding), MQTT-over-TCP (Ettifos), and
    a plain-ASCII key=value UDP "Tx Request" protocol (Commsignia's
    host-to-OBU broadcast request channel). A given file only produces
    results from whichever protocol(s) it actually carries - the other
    paths harmlessly find nothing."""
    udp_by_dir, udp_fail = extract_udp_messages(pcap_path, wsmp_direction)
    mqtt_by_dir, mqtt_fail = extract_mqtt_messages(pcap_path)
    request_by_dir, request_fail = extract_commsignia_request_messages(pcap_path)
    by_direction = {key: udp_by_dir[key] + mqtt_by_dir[key] + request_by_dir[key] for key in udp_by_dir}
    return by_direction, udp_fail + mqtt_fail + request_fail


def correlate(tx_messages, rx_messages, drop_threshold_ms=DROP_LATENCY_THRESHOLD_MS, match_mode="exact"):
    """Match tx (earlier) messages to rx (later) messages with identical
    payload (match_mode="exact") or where the tx payload is an exact
    byte-prefix of the rx payload (match_mode="prefix" - use this when rx
    is expected to carry additional appended content, e.g. an SCMS-signed
    IEEE 1609.2 SignedData envelope wrapped around the unsecured content
    the host originally sent, which is longer than but starts identically
    to the pre-signing payload). Each rx message is consumed at most once.
    Returns (latencies, drops, stale, out_of_window):
      - latencies: matched, latency <= drop_threshold_ms - (tx_timestamp, msg_type, latency_ms)
      - drops: no matching rx payload found, tx fell within the rx capture's
        own time window (so rx *was* recording and simply never saw a
        match) - (tx_timestamp, msg_type, reason)
      - stale: matched, but latency > drop_threshold_ms - (tx_timestamp, msg_type, latency_ms)
      - out_of_window: no matching rx payload found, but tx's timestamp
        falls before the rx capture started or after it stopped - the two
        pcaps simply weren't both recording at that moment, so this isn't
        evidence of anything being lost. Two independent tcpdump/tshark
        captures essentially never start and stop at exactly the same
        instant - (tx_timestamp, msg_type, reason)
    drop_threshold_ms models periodic-message freshness (matching
    v2xhub_messaging_performance_analyzer.py's calculate_messsage_performance):
    a safety broadcast that arrives late has likely been superseded by a
    newer instance already, so it's practically as bad as a drop. That
    doesn't hold for one-off/request-driven messages, so `stale` is kept
    separate with its own latency numbers rather than folded into `drops`
    - a message that took 1.6s to go from host request to over-the-air
    broadcast is a real, measurable latency, not evidence it was lost."""
    rx_by_type = {}
    for i, m in enumerate(rx_messages):
        rx_by_type.setdefault(m["msg_type"], []).append(i)

    rx_timestamps = [m["timestamp"] for m in rx_messages if m["timestamp"] is not None]
    rx_window = (min(rx_timestamps), max(rx_timestamps)) if rx_timestamps else None

    consumed = set()
    latencies = []      # (tx_timestamp, msg_type, latency_ms)
    drops = []          # (tx_timestamp, msg_type, reason)
    stale = []          # (tx_timestamp, msg_type, latency_ms)
    out_of_window = []  # (tx_timestamp, msg_type, reason)

    for tx in tx_messages:
        if tx["timestamp"] is None:
            continue
        candidates = rx_by_type.get(tx["msg_type"], [])
        match_idx = None
        for i in candidates:
            if i in consumed:
                continue
            rx = rx_messages[i]
            if rx["timestamp"] is None:
                continue

            if(match_mode == "exact"):
                payloads_match = rx["payload_hex"] == tx["payload_hex"]
            else:
                payloads_match = (rx["payload_hex"].startswith(tx["payload_hex"]) or tx["payload_hex"].startswith(rx["payload_hex"]))
                
            if payloads_match and rx["timestamp"] >= tx["timestamp"]:
                match_idx = i
                break
        if match_idx is None:
            if rx_window and not (rx_window[0] <= tx["timestamp"] <= rx_window[1]):
                out_of_window.append((tx["timestamp"], tx["msg_type"],
                                       "rx capture wasn't recording yet/anymore at this timestamp"))
            else:
                drops.append((tx["timestamp"], tx["msg_type"], "no matching rx payload found"))
            continue
        consumed.add(match_idx)
        latency_ms = (rx_messages[match_idx]["timestamp"] - tx["timestamp"]) * 1000.0
        if latency_ms > drop_threshold_ms:
            stale.append((tx["timestamp"], tx["msg_type"], latency_ms))
        else:
            latencies.append((tx["timestamp"], tx["msg_type"], latency_ms))
    return latencies, drops, stale, out_of_window


def summarize(latencies):
    vals = sorted(l[2] for l in latencies)
    n = len(vals)
    return {
        "min": vals[0],
        "mean": sum(vals) / n,
        "median": vals[n // 2],
        "p95": vals[int(0.95 * (n - 1))],
        "p99": vals[int(0.99 * (n - 1))],
        "max": vals[-1],
    }

def message_breakout(latencies):
    latency_dict = {}
    for entry in latencies:
        if(entry[1] not in latency_dict):
            latency_dict[entry[1]]={"latency_list":[]}
        latency_dict[entry[1]].get("latency_list").append(entry)
    return latency_dict

def print_latency(latency_dict:dict, latency_type:str):
    for key,value in latency_dict.items():
        value |= summarize(value.get("latency_list"))
        value.pop("latency_list",None)
        print(f"{key} {latency_type} (ms): min={value['min']:.1f} mean={value['mean']:.1f} "
                    f"median={value['median']:.1f} p95={value['p95']:.1f} "
                    f"p99={value['p99']:.1f} max={value['max']:.1f}")

def report_correlation(title, tx_messages, rx_messages, drop_threshold_ms, match_mode="exact"):
    """Runs correlate() and prints a summary. Returns (latencies, drops,
    stale, out_of_window, latency_stats, stale_stats) for callers that also
    want the raw numbers (e.g. for --json-out)."""
    latencies, drops, stale, out_of_window = correlate(tx_messages, rx_messages, drop_threshold_ms, match_mode)
    latency_breakout = message_breakout(latencies)
    stale_breakout = message_breakout(stale)
    n_total = len(tx_messages)
    print(f"\n-- {title} (tx candidates: {n_total}, rx pool: {len(rx_messages)}) --")
    if n_total:
        print(f"Matched (fresh): {len(latencies)}/{n_total} ({100*len(latencies)/n_total:.1f}%)")
        print(f"Matched but stale (> {drop_threshold_ms:.0f} ms): {len(stale)}/{n_total} ({100*len(stale)/n_total:.1f}%)")
        print(f"Dropped (no match found, within rx's own recording window): "
              f"{len(drops)}/{n_total} ({100*len(drops)/n_total:.1f}%)")
        print(f"Outside rx's recording window (not a real drop - the two captures "
              f"didn't start/stop at the same instant): {len(out_of_window)}/{n_total} "
              f"({100*len(out_of_window)/n_total:.1f}%)")
    else:
        print("Matched (fresh): 0/0")
        print("Matched but stale: 0/0")
        print("Dropped (no match found, within rx's own recording window): 0/0")
        print("Outside rx's recording window: 0/0")

    if latencies:
        print_latency(latency_breakout,"Fresh Latency")

    if stale:
        print_latency(stale_breakout, "Stale Latency")

    if drops:
        print(f"Drops by message type: {dict(Counter(d[1] for d in drops))}")

    return latencies, drops, stale, out_of_window, latency_breakout, stale_breakout


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


# A flow pairs the sender's copy of each message with the receiver's copy.
# Direction labels describe the capturing interface (sent vs received), so
# which buckets hold each copy depends on where the two captures were taken:
#   same-device:  a message passes *through* one device. It is received on
#                 tx (radio) and re-sent on rx (host link). The rx host link is
#                 typically plain Ethernet with no direction bit, hence "other".
#   cross-device: a message goes *from* one device *to* another, so it is
#                 outgoing at the sender and incoming at the receiver. Both
#                 devices send something, so both directions are reported.
#                 "other" is excluded: with no direction bit, a third-party
#                 broadcast heard by both devices would match itself.
# (json_key, plot_suffix, title, (sender_file, buckets), (receiver_file, buckets), default_match_mode)
FLOWS = {
    "same-device": [
        ("incoming_flow", "incoming", "Incoming-flow correlation (radio receipt -> forwarded to host)",
         ("tx", ("incoming",)), ("rx", ("incoming", "other")), "exact"),
        # Prefix: with SCMS enabled the OBU wraps the host's unsecured request in
        # a signed 1609.2 envelope, so the broadcast is longer than, but starts
        # identically to, what the host sent.
        ("outgoing_request_flow", "outgoing",
         "Outgoing/request-flow correlation (host request -> over-the-air broadcast)",
         ("rx", ("outgoing",)), ("tx", ("outgoing",)), "prefix"),
    ],
    "cross-device": [
        ("tx_to_rx_flow", "tx_to_rx", "tx device -> rx device (tx outgoing -> rx incoming)",
         ("tx", ("outgoing",)), ("rx", ("incoming",)), "exact"),
        ("rx_to_tx_flow", "rx_to_tx", "rx device -> tx device (rx outgoing -> tx incoming)",
         ("rx", ("outgoing",)), ("tx", ("incoming",)), "exact"),
    ],
}


def run_flow(title, sender, receiver, match_mode, args, label, plot_suffix):
    """Returns (json_summary or None if skipped, fresh latencies in ms)."""
    if not sender:
        print(f"\n-- {title} - no sender-side messages, skipped (receiver-side counts: "
              f"{dict(Counter(m['msg_type'] for m in receiver))}) --")
        return None, []
    latencies, drops, stale, out_of_window, latency_stats, stale_stats = report_correlation(
        title, sender, receiver, args.drop_threshold_ms, match_mode)
    windowed_count_report(title, sender, receiver)
    if args.plot_dir:
        plot_flow(
            args.plot_dir / f"{slug(label)}_{plot_suffix}.png", f"{label} - {title}",
            latencies, stale, drops, out_of_window,
            args.drop_threshold_ms, duplicate_content_types=find_duplicate_content_types(sender),
        )
    return {
        "match_mode": match_mode,
        "matched": len(latencies),
        "stale": len(stale),
        "dropped": len(drops),
        "out_of_window": len(out_of_window),
        "drops_by_type": dict(Counter(d[1] for d in drops)),
        "latency_stats_ms": latency_stats,
        "stale_latency_stats_ms": stale_stats,
    }, [l[2] for l in latencies]


def estimate_clock_offset(tx_to_rx_latencies, rx_to_tx_latencies):
    """Each measured one-way latency includes the clock offset between the two
    devices with opposite signs per direction (fwd = L + offset, rev = L - offset),
    so with both directions measured the offset and latency can be separated,
    assuming the true latency is the same both ways."""
    if not tx_to_rx_latencies or not rx_to_tx_latencies:
        return None
    fwd = statistics.median(tx_to_rx_latencies)
    rev = statistics.median(rx_to_tx_latencies)
    return {"rx_minus_tx_clock_ms": (fwd - rev) / 2, "one_way_latency_ms": (fwd + rev) / 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx-pcap", required=True,
                    help="same-device: earlier capture point (e.g. radio interface); "
                         "cross-device: either device's capture (e.g. RSU)")
    ap.add_argument("--rx-pcap", required=True,
                    help="same-device: later capture point (e.g. ethernet-out to host); "
                         "cross-device: the other device's capture (e.g. OBU)")
    ap.add_argument("--topology", choices=sorted(FLOWS), default="same-device",
                    help="same-device: two interfaces of one device (message passes through it); "
                         "cross-device: two devices' radio captures (message goes from one to the other, "
                         "both directions reported)")
    ap.add_argument("--drop-threshold-ms", type=float, default=DROP_LATENCY_THRESHOLD_MS,
                     help=f"Matched messages slower than this are counted as drops (default {DROP_LATENCY_THRESHOLD_MS} ms)")
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional path to write results as JSON")
    ap.add_argument("--plot-dir", type=Path, default=None,
                     help="Optional directory to write per-flow latency/drop PNG plots to")
    ap.add_argument("--match-mode", default=None, choices=["exact", "prefix"],
                    help="Override every flow's payload match mode (default: per flow - prefix for "
                         "the same-device host-request flow, exact otherwise)")
    ap.add_argument("--wsmp-direction", default="unknown", choices=["unknown", "incoming", "outgoing"],
                    help="Direction to assign WSMP packets on links with no direction bit (plain Ethernet); "
                         "Linux-cooked captures always use their own direction bit")
    args = ap.parse_args()
    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    by_file = {}
    fails = {}
    by_file["tx"], fails["tx"] = extract_messages(args.tx_pcap, args.wsmp_direction)
    by_file["rx"], fails["rx"] = extract_messages(args.rx_pcap, args.wsmp_direction)

    label = args.label or f"{Path(args.tx_pcap).name} -> {Path(args.rx_pcap).name}"
    print(f"=== {label} ({args.topology}) ===")
    for iface in ("tx", "rx"):
        counts = {d: dict(Counter(m["msg_type"] for m in msgs)) for d, msgs in by_file[iface].items() if msgs}
        print(f"{iface}: {counts} (decode failures: {fails[iface]})")
        if args.topology == "cross-device" and by_file[iface]["other"]:
            print(f"WARNING: {len(by_file[iface]['other'])} {iface} messages have no direction information "
                  f"(capture link type has no direction bit) and are excluded from cross-device flows. "
                  f"Capture with a Linux-cooked link type (e.g. tcpdump -i any) to include them.")

    flow_results = {}
    flow_latencies = {}
    for key, plot_suffix, title, (s_file, s_buckets), (r_file, r_buckets), default_mode in FLOWS[args.topology]:
        sender = [m for b in s_buckets for m in by_file[s_file][b]]
        receiver = [m for b in r_buckets for m in by_file[r_file][b]]
        flow_results[key], flow_latencies[key] = run_flow(
            title, sender, receiver, args.match_mode or default_mode, args, label, plot_suffix)

    clock = None
    if args.topology == "cross-device":
        clock = estimate_clock_offset(flow_latencies["tx_to_rx_flow"], flow_latencies["rx_to_tx_flow"])
        if clock:
            print(f"\nEstimated clock offset (rx clock minus tx clock): {clock['rx_minus_tx_clock_ms']:.2f} ms, "
                  f"estimated one-way latency: {clock['one_way_latency_ms']:.2f} ms "
                  f"(from median fresh latency in each direction; assumes equal latency both ways)")
        else:
            print("\nClock offset not estimated: needs matched messages in both directions.")

    if args.json_out:
        result = {
            "label": label,
            "topology": args.topology,
            "tx_pcap": str(args.tx_pcap),
            "rx_pcap": str(args.rx_pcap),
            "tx_counts": {d: dict(Counter(m["msg_type"] for m in msgs)) for d, msgs in by_file["tx"].items() if msgs},
            "rx_counts": {d: dict(Counter(m["msg_type"] for m in msgs)) for d, msgs in by_file["rx"].items() if msgs},
            "tx_decode_failures": fails["tx"],
            "rx_decode_failures": fails["rx"],
            **flow_results,
        }
        if args.topology == "cross-device":
            result["clock_offset_estimate"] = clock
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"Results saved to: {args.json_out}")

if __name__ == "__main__":
    main()
