#!/usr/bin/env python3
"""
Correlates J2735 messages between an OBU's host-facing ethernet pcap capture
and the ROS2 `carma_driver_msgs/msg/ByteArray` topics that the vehicle's
comms driver publishes/subscribes to, to check whether any messages are being
dropped crossing that pcap<->ROS boundary:

  - inbound:  eth0 pcap  -> /hardware_interface/comms/inbound_binary_msg
              (message arrives at the host over ethernet from the OBU, then
              the driver node is expected to publish it to ROS)
  - outbound: /hardware_interface/comms/outbound_binary_msg -> eth0 pcap
              (ROS software asks the driver to transmit a message, then the
              driver is expected to put it on the wire toward the OBU)

Reuses pcap-side message extraction (`extract_messages`) from
correlate_j2735_latency.py unchanged - same three vendor protocols
(raw-UDP/1609.2, MQTT, Commsignia ASCII Tx-Request), same envelope-aware
parsing. Only the mcap side and the tx/rx wiring are new.

The `ByteArray.content` field (see carma_driver_msgs/msg/ByteArray.msg) is
the raw J2735 MessageFrame bytes (`00 <msgid> ...`), with no 1609.2 envelope
- this is exactly the same on-the-wire format `extract_messages()` already
produces as `payload_hex` for all three pcap protocols, so messages can be
matched byte-for-byte between the two sides with no format translation.

mcap side is read with the low-level sequential `StreamReader` rather than
`McapReader.iter_messages()`: every capture in data/ here is missing its
trailing Footer/magic (the recording process was killed rather than shut
down cleanly), which makes the indexed/summary-based reader fail outright.
Sequential reading tolerates that the same way `read_classic_pcap` tolerates
a truncated trailing packet - stop at the first unreadable record and use
whatever was read so far.

Usage:
    python3 correlate_pcap_mcap.py --eth0-pcap eth0.pcap --mcap run.mcap [--label NAME] [--json-out results.json]
"""
import sys
import re
import argparse
import json
from pathlib import Path
from collections import Counter

from mcap.stream_reader import StreamReader
from mcap.records import Schema, Channel, Message
from mcap_ros2.decoder import DecoderFactory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correlate_j2735_latency as base  # noqa: E402
from correlation_plots import plot_flow, find_duplicate_content_types, windowed_count_report  # noqa: E402

INBOUND_TOPIC = "/hardware_interface/comms/inbound_binary_msg"
OUTBOUND_TOPIC = "/hardware_interface/comms/outbound_binary_msg"

# ByteArray.message_type spells this one out in full, while
# correlate_j2735_latency.py's J2735_MESSAGE_NAMES (shared vocabulary for
# matching against the pcap side) abbreviates it like the other J2735 types.
MCAP_MESSAGE_TYPE_ALIASES = {"SensorDataSharingMessage": "SDSM"}

# How far apart (either direction) a pcap timestamp and an mcap header.stamp
# for the *same* payload can be before we stop calling it "the same crossing
# event" at all. Both timestamps come from the same onboard clock, but pcap
# frame-capture time and ROS message-stamp-assignment time aren't causally
# ordered at sub-millisecond granularity - confirmed on run-1 commsignia,
# where genuinely-matching payloads showed the mcap stamp up to ~0.5ms
# *before* the pcap timestamp. base.correlate()'s `rx_ts >= tx_ts` ordering
# requirement (correct for its actual use case: radio-receive-hop vs.
# forward-hop, where causality is real and coarser) misclassifies those as
# 100% drops here. A generous symmetric window absorbs clock jitter while
# still catching genuine multi-second staleness/drops.
MAX_ABS_MATCH_WINDOW_S = 10.0


def correlate_across_boundary(tx_messages, rx_messages, drop_threshold_ms, match_mode="exact"):
    """Like base.correlate(), but for matching a pcap-side capture against an
    mcap-side ROS topic across the same host's network<->ROS boundary: does
    NOT require rx_ts >= tx_ts (see MAX_ABS_MATCH_WINDOW_S above), and picks
    the closest-in-time unconsumed candidate by absolute difference rather
    than the first one satisfying a causal order. Returns the same
    (latencies, drops, stale, out_of_window) shape as base.correlate(), with
    latency_ms signed (rx - tx; small negative values are expected jitter,
    not evidence of anything)."""
    rx_by_type = {}
    for i, m in enumerate(rx_messages):
        rx_by_type.setdefault(m["msg_type"], []).append(i)

    rx_timestamps = [m["timestamp"] for m in rx_messages if m["timestamp"] is not None]
    rx_window = (min(rx_timestamps), max(rx_timestamps)) if rx_timestamps else None

    consumed = set()
    latencies = []
    drops = []
    stale = []
    out_of_window = []

    for tx in tx_messages:
        if tx["timestamp"] is None:
            continue
        candidates = rx_by_type.get(tx["msg_type"], [])
        best_idx = None
        best_diff = None
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
            if not payloads_match:
                continue
            diff = rx["timestamp"] - tx["timestamp"]
            if abs(diff) > MAX_ABS_MATCH_WINDOW_S:
                continue
            if best_idx is None or abs(diff) < abs(best_diff):
                best_idx = i
                best_diff = diff
        if best_idx is None:
            if rx_window and not (rx_window[0] <= tx["timestamp"] <= rx_window[1]):
                out_of_window.append((tx["timestamp"], tx["msg_type"],
                                       "rx capture wasn't recording yet/anymore at this timestamp"))
            else:
                drops.append((tx["timestamp"], tx["msg_type"], "no matching rx payload found"))
            continue
        consumed.add(best_idx)
        latency_ms = best_diff * 1000.0
        if abs(latency_ms) > drop_threshold_ms:
            stale.append((tx["timestamp"], tx["msg_type"], latency_ms))
        else:
            latencies.append((tx["timestamp"], tx["msg_type"], latency_ms))
    return latencies, drops, stale, out_of_window


def report_boundary_correlation(title, tx_messages, rx_messages, drop_threshold_ms, match_mode="exact"):
    latencies, drops, stale, out_of_window = correlate_across_boundary(
        tx_messages, rx_messages, drop_threshold_ms, match_mode)
    n_total = len(tx_messages)
    print(f"\n-- {title} (tx candidates: {n_total}, rx pool: {len(rx_messages)}) --")
    if n_total:
        print(f"Matched (fresh): {len(latencies)}/{n_total} ({100*len(latencies)/n_total:.1f}%)")
        print(f"Matched but stale (|latency| > {drop_threshold_ms:.0f} ms): {len(stale)}/{n_total} "
              f"({100*len(stale)/n_total:.1f}%)")
        print(f"Dropped (no match found, within rx's own recording window): "
              f"{len(drops)}/{n_total} ({100*len(drops)/n_total:.1f}%)")
        print(f"Outside rx's recording window (not a real drop): {len(out_of_window)}/{n_total} "
              f"({100*len(out_of_window)/n_total:.1f}%)")
    else:
        print("Matched (fresh): 0/0")
        print("Matched but stale: 0/0")
        print("Dropped (no match found, within rx's own recording window): 0/0")
        print("Outside rx's recording window: 0/0")

    latency_stats = None
    if latencies:
        latency_stats = base.summarize(latencies)
        print(f"Latency (ms, signed rx-tx): min={latency_stats['min']:.1f} mean={latency_stats['mean']:.1f} "
              f"median={latency_stats['median']:.1f} p95={latency_stats['p95']:.1f} "
              f"p99={latency_stats['p99']:.1f} max={latency_stats['max']:.1f}")

    stale_stats = None
    if stale:
        stale_stats = base.summarize(stale)
        print(f"Stale latency (ms, signed rx-tx): min={stale_stats['min']:.1f} mean={stale_stats['mean']:.1f} "
              f"median={stale_stats['median']:.1f} p95={stale_stats['p95']:.1f} "
              f"p99={stale_stats['p99']:.1f} max={stale_stats['max']:.1f} "
              f"(by type: {dict(Counter(s[1] for s in stale))})")

    if drops:
        print(f"Drops by message type: {dict(Counter(d[1] for d in drops))}")

    return latencies, drops, stale, out_of_window, latency_stats, stale_stats


def extract_mcap_binary_messages(mcap_path):
    """Sequentially scans an mcap for the inbound/outbound ByteArray topics.
    Returns {"inbound": [...], "outbound": [...]}, each a list of dicts with
    timestamp (float seconds, from the message's own std_msgs/Header stamp -
    not mcap log_time, since log_time is recording-host receive time and
    stamp is the driver-assigned message time that's comparable to pcap
    frame timestamps), msg_type (str, from the ByteArray.message_type
    field), and payload_hex (str, from ByteArray.content) - the same shape
    `correlate()` in correlate_j2735_latency.py already expects."""
    schemas = {}
    channels = {}
    decoder_factory = DecoderFactory()
    by_topic = {INBOUND_TOPIC: [], OUTBOUND_TOPIC: []}

    with open(mcap_path, "rb") as f:
        reader = StreamReader(f, record_size_limit=None)
        try:
            for record in reader.records:
                if isinstance(record, Schema):
                    schemas[record.id] = record
                elif isinstance(record, Channel):
                    channels[record.id] = record
                elif isinstance(record, Message):
                    channel = channels.get(record.channel_id)
                    if channel is None or channel.topic not in by_topic:
                        continue
                    schema = schemas.get(channel.schema_id)
                    decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
                    if decoder is None:
                        continue
                    decoded = decoder(record.data)
                    timestamp = decoded.header.stamp.sec + decoded.header.stamp.nanosec * 1e-9
                    msg_type = MCAP_MESSAGE_TYPE_ALIASES.get(decoded.message_type, decoded.message_type)
                    by_topic[channel.topic].append({
                        "timestamp": timestamp,
                        "msg_type": msg_type,
                        "payload_hex": bytes(decoded.content).hex(),
                    })
        except Exception as e:
            print(f"WARNING: {mcap_path}: stopped reading at {type(e).__name__}: {e} "
                  f"(likely an unfinalized/truncated recording) - using partial results", file=sys.stderr)

    return {"inbound": by_topic[INBOUND_TOPIC], "outbound": by_topic[OUTBOUND_TOPIC]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eth0-pcap", required=True, help="Host-facing ethernet capture (e.g. run-N-eth0-*.pcap)")
    ap.add_argument("--mcap", required=True, help="mcap recording covering the same time window")
    ap.add_argument("--drop-threshold-ms", type=float, default=base.DROP_LATENCY_THRESHOLD_MS)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--plot-dir", type=Path, default=None,
                     help="Optional directory to write per-flow latency/drop PNG plots to")
    args = ap.parse_args()
    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    pcap_by_dir, pcap_fail = base.extract_messages(args.eth0_pcap)
    mcap_by_dir = extract_mcap_binary_messages(args.mcap)

    label = args.label or f"{Path(args.eth0_pcap).name} <-> {Path(args.mcap).name}"
    print(f"=== {label} ===")
    pcap_counts = {d: dict(Counter(m["msg_type"] for m in msgs)) for d, msgs in pcap_by_dir.items() if msgs}
    mcap_counts = {d: dict(Counter(m["msg_type"] for m in msgs)) for d, msgs in mcap_by_dir.items() if msgs}
    print(f"eth0 pcap: {pcap_counts} (decode failures: {pcap_fail})")
    print(f"mcap: {mcap_counts}")

    # Inbound flow: message hits the wire (pcap "incoming", or "other" on
    # plain-Ethernet links with no direction bit), then the driver is
    # expected to publish it to ROS shortly after.
    pcap_incoming_pool = pcap_by_dir["incoming"] + pcap_by_dir["other"]
    inbound_result = report_boundary_correlation(
        "Inbound flow (eth0 pcap -> /hardware_interface/comms/inbound_binary_msg)",
        pcap_incoming_pool, mcap_by_dir["inbound"], args.drop_threshold_ms,
    )
    (inbound_latencies, inbound_drops, inbound_stale, inbound_out_of_window,
     inbound_stats, inbound_stale_stats) = inbound_result
    windowed_count_report(
        "Inbound flow (eth0 pcap -> /hardware_interface/comms/inbound_binary_msg)",
        pcap_incoming_pool, mcap_by_dir["inbound"],
    )
    if args.plot_dir:
        plot_flow(
            args.plot_dir / f"{base.slug(label)}_inbound.png",
            f"{label} - inbound flow (eth0 pcap -> inbound_binary_msg)",
            inbound_latencies, inbound_stale, inbound_drops, inbound_out_of_window,
            args.drop_threshold_ms, duplicate_content_types=find_duplicate_content_types(pcap_incoming_pool),
        )

    # Outbound flow: ROS software publishes a transmit request, then the
    # driver is expected to put it on the wire shortly after.
    outbound_result = report_boundary_correlation(
        "Outbound flow (/hardware_interface/comms/outbound_binary_msg -> eth0 pcap)",
        mcap_by_dir["outbound"], pcap_by_dir["outgoing"], args.drop_threshold_ms,
    )
    (outbound_latencies, outbound_drops, outbound_stale, outbound_out_of_window,
     outbound_stats, outbound_stale_stats) = outbound_result
    windowed_count_report(
        "Outbound flow (/hardware_interface/comms/outbound_binary_msg -> eth0 pcap)",
        mcap_by_dir["outbound"], pcap_by_dir["outgoing"],
    )
    if args.plot_dir:
        plot_flow(
            args.plot_dir / f"{base.slug(label)}_outbound.png",
            f"{label} - outbound flow (outbound_binary_msg -> eth0 pcap)",
            outbound_latencies, outbound_stale, outbound_drops, outbound_out_of_window,
            args.drop_threshold_ms, duplicate_content_types=find_duplicate_content_types(mcap_by_dir["outbound"]),
        )

    if args.json_out:
        from collections import Counter as C
        result = {
            "label": label,
            "eth0_pcap": str(args.eth0_pcap),
            "mcap": str(args.mcap),
            "pcap_counts": pcap_counts,
            "mcap_counts": mcap_counts,
            "pcap_decode_failures": pcap_fail,
            "inbound_flow": {
                "matched": len(inbound_latencies),
                "stale": len(inbound_stale),
                "dropped": len(inbound_drops),
                "out_of_window": len(inbound_out_of_window),
                "drops_by_type": dict(C(d[1] for d in inbound_drops)),
                "latency_stats_ms": inbound_stats,
                "stale_latency_stats_ms": inbound_stale_stats,
            },
            "outbound_flow": {
                "matched": len(outbound_latencies),
                "stale": len(outbound_stale),
                "dropped": len(outbound_drops),
                "out_of_window": len(outbound_out_of_window),
                "drops_by_type": dict(C(d[1] for d in outbound_drops)),
                "latency_stats_ms": outbound_stats,
                "stale_latency_stats_ms": outbound_stale_stats,
            },
        }
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"Results saved to: {args.json_out}")


if __name__ == "__main__":
    main()
