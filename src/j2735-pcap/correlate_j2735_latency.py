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
never reached. The pcap/link-layer/UDP parsing and envelope decoding here
are ported from carma-platform/engineering_tools/check_v2x_security_direction.py,
which also classifies packet direction (incoming vs. outgoing) using the
Linux-cooked-capture SLL header; that distinction matters for correlation,
see below.

Only messages captured going toward the OBU (`incoming`) are matched
between tx and rx - a capture may also contain `outgoing` messages (e.g. a
BSM the OBU broadcasts itself), which are a structurally different flow
that isn't expected to reappear on a downstream/forwarding capture, so
folding them into drop-rate math produces a false "100% drop" reading.
`outgoing` messages are reported separately, for visibility only.

Each message's timestamp is sourced from `tshark -e frame.time_epoch` for
the packet it was actually found in (matched back by packet number), since
the pure-Python pcap reader used for envelope parsing doesn't expose
per-packet timestamps.

Usage:
    python3 correlate_j2735_latency.py --tx-pcap earlier.pcap --rx-pcap later.pcap
"""
import json
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

DROP_LATENCY_THRESHOLD_MS = 200  # matches v2xhub_messaging_performance_analyzer.py's threshold

DLT_EN10MB = 1
DLT_LINUX_SLL = 113
ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_IPV6 = 0x86DD
ETHERTYPE_VLAN = 0x8100

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


def parse_udp_with_linktype(data: bytes, linktype: int) -> Optional[UdpPacket]:
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


def extract_udp_messages(pcap_path):
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
        udp = parse_udp_with_linktype(packet.data, linktype)
        if udp is None:
            continue
        found = find_message_offset(udp.payload)
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


def extract_messages(pcap_path):
    """Extracts J2735 messages from a pcap, trying both the raw-UDP
    (1609.2-enveloped) and MQTT-over-TCP extraction paths, since different
    OBU vendors expose their ethernet-facing interface differently (e.g.
    Commsignia: raw UDP; Ettifos: MQTT). A given file only produces results
    from whichever protocol it actually carries - the other path harmlessly
    finds nothing."""
    udp_by_dir, udp_fail = extract_udp_messages(pcap_path)
    mqtt_by_dir, mqtt_fail = extract_mqtt_messages(pcap_path)
    by_direction = {key: udp_by_dir[key] + mqtt_by_dir[key] for key in udp_by_dir}
    return by_direction, udp_fail + mqtt_fail


def correlate(tx_messages, rx_messages, drop_threshold_ms=DROP_LATENCY_THRESHOLD_MS, match_mode="exact"):
    """Match tx (earlier) messages to rx (later) messages with identical
    payload (match_mode="exact") or where the tx payload is an exact
    byte-prefix of the rx payload (match_mode="prefix" - use this when rx
    is expected to carry additional appended content, e.g. an SCMS-signed
    IEEE 1609.2 SignedData envelope wrapped around the unsecured content
    the host originally sent, which is longer than but starts identically
    to the pre-signing payload). Each rx message is consumed at most once.
    Returns (latencies, drops, stale):
      - latencies: matched, latency <= drop_threshold_ms - (tx_timestamp, msg_type, latency_ms)
      - drops: no matching rx payload found at all - (tx_timestamp, msg_type, reason)
      - stale: matched, but latency > drop_threshold_ms - (tx_timestamp, msg_type, latency_ms)
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

    consumed = set()
    latencies = []  # (tx_timestamp, msg_type, latency_ms)
    drops = []      # (tx_timestamp, msg_type, reason)
    stale = []      # (tx_timestamp, msg_type, latency_ms)

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
            payloads_match = (
                rx["payload_hex"] == tx["payload_hex"] if match_mode == "exact"
                else rx["payload_hex"].startswith(tx["payload_hex"])
            )
            if payloads_match and rx["timestamp"] >= tx["timestamp"]:
                match_idx = i
                break
        if match_idx is None:
            drops.append((tx["timestamp"], tx["msg_type"], "no matching rx payload found"))
            continue
        consumed.add(match_idx)
        latency_ms = (rx_messages[match_idx]["timestamp"] - tx["timestamp"]) * 1000.0
        if latency_ms > drop_threshold_ms:
            stale.append((tx["timestamp"], tx["msg_type"], latency_ms))
        else:
            latencies.append((tx["timestamp"], tx["msg_type"], latency_ms))
    return latencies, drops, stale


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


def report_correlation(title, tx_messages, rx_messages, drop_threshold_ms, match_mode="exact"):
    """Runs correlate() and prints a summary. Returns (latencies, drops,
    stale, latency_stats, stale_stats) for callers that also want the raw
    numbers (e.g. for --json-out)."""
    latencies, drops, stale = correlate(tx_messages, rx_messages, drop_threshold_ms, match_mode)
    n_total = len(tx_messages)
    print(f"\n-- {title} (tx candidates: {n_total}, rx pool: {len(rx_messages)}) --")
    if n_total:
        print(f"Matched (fresh): {len(latencies)}/{n_total} ({100*len(latencies)/n_total:.1f}%)")
        print(f"Matched but stale (> {drop_threshold_ms:.0f} ms): {len(stale)}/{n_total} ({100*len(stale)/n_total:.1f}%)")
        print(f"Dropped (no match found): {len(drops)}/{n_total} ({100*len(drops)/n_total:.1f}%)")
    else:
        print("Matched (fresh): 0/0")
        print("Matched but stale: 0/0")
        print("Dropped (no match found): 0/0")

    latency_stats = None
    if latencies:
        latency_stats = summarize(latencies)
        print(f"Latency (ms): min={latency_stats['min']:.1f} mean={latency_stats['mean']:.1f} "
              f"median={latency_stats['median']:.1f} p95={latency_stats['p95']:.1f} "
              f"p99={latency_stats['p99']:.1f} max={latency_stats['max']:.1f}")

    stale_stats = None
    if stale:
        stale_stats = summarize(stale)
        print(f"Stale latency (ms): min={stale_stats['min']:.1f} mean={stale_stats['mean']:.1f} "
              f"median={stale_stats['median']:.1f} p95={stale_stats['p95']:.1f} "
              f"p99={stale_stats['p99']:.1f} max={stale_stats['max']:.1f} "
              f"(by type: {dict(Counter(s[1] for s in stale))})")

    if drops:
        print(f"Drops by message type: {dict(Counter(d[1] for d in drops))}")

    return latencies, drops, stale, latency_stats, stale_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx-pcap", required=True, help="Earlier capture point (e.g. radio-receive interface)")
    ap.add_argument("--rx-pcap", required=True, help="Later capture point (e.g. ethernet-out to downstream host)")
    ap.add_argument("--drop-threshold-ms", type=float, default=DROP_LATENCY_THRESHOLD_MS,
                     help=f"Matched messages slower than this are counted as drops (default {DROP_LATENCY_THRESHOLD_MS} ms)")
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional path to write results as JSON")
    args = ap.parse_args()

    tx_by_dir, tx_fail = extract_messages(args.tx_pcap)
    rx_by_dir, rx_fail = extract_messages(args.rx_pcap)

    label = args.label or f"{Path(args.tx_pcap).name} -> {Path(args.rx_pcap).name}"
    print(f"=== {label} ===")
    for iface, by_dir, fail in (("tx", tx_by_dir, tx_fail), ("rx", rx_by_dir, rx_fail)):
        counts = {d: dict(Counter(m["msg_type"] for m in msgs)) for d, msgs in by_dir.items() if msgs}
        print(f"{iface}: {counts} (decode failures: {fail})")

    # Incoming flow: message arrives at the OBU from the infrastructure
    # (tx's "incoming"), then is forwarded to the host (rx's "incoming" or,
    # for interfaces with no direction bit like plain Ethernet, "other").
    tx_incoming = tx_by_dir["incoming"]
    rx_incoming_pool = rx_by_dir["incoming"] + rx_by_dir["other"]
    incoming_latencies, incoming_drops, incoming_stale, incoming_stats, incoming_stale_stats = report_correlation(
        "Incoming-flow correlation (radio receipt -> forwarded to host)",
        tx_incoming, rx_incoming_pool, args.drop_threshold_ms,
    )

    # Outgoing/request flow: the host asks the OBU to transmit something
    # (rx's "outgoing", e.g. an MQTT '/req/' publish or a request captured
    # on the ethernet interface), then the OBU actually broadcasts it over
    # the air (tx's "outgoing"). Chronologically rx happens first here, so
    # roles are swapped relative to the incoming-flow call above. Matched
    # by prefix, not exact equality: when SCMS security is enabled, the
    # OBU wraps the host's unsecured request in a signed IEEE 1609.2
    # SignedData envelope (certificate + signature) before transmitting,
    # so the broadcast payload is longer than, but starts identically to,
    # what the host sent.
    request_messages = rx_by_dir["outgoing"]
    broadcast_pool = tx_by_dir["outgoing"]
    outgoing_latencies = outgoing_drops = outgoing_stale = outgoing_stats = outgoing_stale_stats = None
    if request_messages and broadcast_pool:
        outgoing_latencies, outgoing_drops, outgoing_stale, outgoing_stats, outgoing_stale_stats = report_correlation(
            "Outgoing/request-flow correlation (host request -> over-the-air broadcast)",
            request_messages, broadcast_pool, args.drop_threshold_ms, match_mode="prefix",
        )
    elif broadcast_pool:
        print(f"\n-- Outgoing flow (tx's own broadcast, e.g. BSM) - "
              f"no host-request-side capture to compare against, informational only --")
        print(f"tx outgoing counts: {dict(Counter(m['msg_type'] for m in broadcast_pool))}")

    if args.json_out:
        result = {
            "label": label,
            "tx_pcap": str(args.tx_pcap),
            "rx_pcap": str(args.rx_pcap),
            "tx_counts": {d: dict(Counter(m["msg_type"] for m in msgs)) for d, msgs in tx_by_dir.items() if msgs},
            "rx_counts": {d: dict(Counter(m["msg_type"] for m in msgs)) for d, msgs in rx_by_dir.items() if msgs},
            "tx_decode_failures": tx_fail,
            "rx_decode_failures": rx_fail,
            "incoming_flow": {
                "matched": len(incoming_latencies),
                "stale": len(incoming_stale),
                "dropped": len(incoming_drops),
                "drops_by_type": dict(Counter(d[1] for d in incoming_drops)),
                "latency_stats_ms": incoming_stats,
                "stale_latency_stats_ms": incoming_stale_stats,
            },
            "outgoing_request_flow": None if outgoing_latencies is None else {
                "matched": len(outgoing_latencies),
                "stale": len(outgoing_stale),
                "dropped": len(outgoing_drops),
                "drops_by_type": dict(Counter(d[1] for d in outgoing_drops)),
                "latency_stats_ms": outgoing_stats,
                "stale_latency_stats_ms": outgoing_stale_stats,
            },
        }
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"Results saved to: {args.json_out}")


if __name__ == "__main__":
    main()
