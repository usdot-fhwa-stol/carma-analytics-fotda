# PCAP Decoder for SAE J2735 Messages

This script decodes PCAP files containing SAE J2735 messages. The decoded JSON output is saved in a text file with the same name as the original PCAP file. The output is also saved in a csv file which contains the packet number, timestamp, J2735 Message Type, and J2735 UPER Message Payload.

## Supported Platforms
- **Linux**

## Prerequisites

- Python 3
    - pyshark
    - pycrate
- Tshark

Dependencies are automatically installed (if not installed already) when script is run.

## Usage

1. Move your PCAP file containing J2735 messages to the `J2735-PCAP` directory containing the scripts.
2. Execute the script:
```
./pcapDecoder.sh
```
3. Follow the on-screen prompts.
4. Chosen file contents will be decoded and written to the output files in the `decoded` directory. The original PCAP file and text file containing the full PCAP payloads will be moved to the `logs` directory.

## Correlating Latency Between Two Captures

`correlate_j2735_latency.py` compares two pcap captures of the same broadcast stream taken at different points (e.g. a radio-receive interface vs. an ethernet-out interface to a downstream host) and reports per-message latency, match rate, and drops.

Message location is IEEE 1609.2-envelope-aware: it scans each UDP payload for the `unsecuredData` marker (`03 80`) followed by a valid OER length field, then reads the J2735 message ID at the computed content offset — rather than blindly searching for hex markers like `0012`/`0013` anywhere in the payload (that's what `decodeJ2735.py` and an earlier version of this script did). Each message's timestamp is sourced from `tshark -e frame.time_epoch` for the packet it was actually found in.

> [!NOTE]
> The blind-substring approach silently misses messages: a coincidental 2-byte match elsewhere in a packet's payload gets found first and fails validation, while the real marker — only locatable by walking the actual 1609.2 envelope structure — is never reached. Confirmed on a real capture where this caused MAP messages to be missed entirely. Separately, `decodeJ2735.py`'s own CSV/text output assigns timestamps by indexing `pyshark.FileCapture(...)` with the running count of *successfully decoded* messages rather than actual packet position, so timestamps silently drift wrong whenever the capture contains packets that don't decode (background SSH/TCP traffic, etc.) — confirmed compressing reported message timestamps into ~21% of a capture's true duration. Use `correlate_j2735_latency.py` for both message detection and latency, not `decodeJ2735.py`'s own output, whenever a capture may contain non-J2735 traffic or you need MAP messages recognized.

For captures with a Linux-cooked link layer (`rmnet`-style), packets are also classified `incoming` (from the infrastructure/RSU) vs. `outgoing` (the device's own broadcast, e.g. a BSM it transmits itself).

### Three ethernet-facing protocols, auto-detected

Different OBU vendors (and different channels on the same vendor) expose messages on the host-facing interface differently. `extract_messages()` tries all three on every file; a file only produces results from whichever protocol(s) it actually carries:

1. **Raw UDP, 1609.2-enveloped** — the RSU-broadcast-forwarding channel (e.g. Commsignia port 5398). Classified `incoming`/`outgoing`/`other` from the Linux-cooked SLL direction bit, or `other` on plain Ethernet.
2. **MQTT-over-TCP** — Ettifos-vendor OBUs. Payloads are raw MessageFrame bytes with no 1609.2 envelope. Topic naming gives direction: `/ind/...` (OBU→host, an actual received/broadcast indication) → `incoming`; `/req/...` (host→OBU, asking it to transmit) → `outgoing`.
3. **Commsignia "Tx Request" ASCII protocol** — a separate UDP channel (port varies by deployment; detected by content, not port number) the host uses to ask a Commsignia OBU to broadcast a message: plain newline-separated `Version=.../Type=.../PSID=.../Signature=.../Payload=<hex>` text, payload unsigned. Always `outgoing` by definition.

**Why this matters for correlation**: `outgoing` (tx's own broadcast, or an MQTT/Commsignia request) is matched against the *other* file's `outgoing` bucket, not `incoming` — a host's request to broadcast BSM/MobilityOperation isn't the same flow as an RSU's SPAT/MAP/SDSM broadcast, and conflating them produces false "100% drop" readings (confirmed: this happened for BSM before this distinction existed). If SCMS security is enabled, match with `match_mode="prefix"` for the outgoing/request flow — the OBU wraps the host's unsecured request in a signed IEEE 1609.2 `SignedData` envelope before transmitting, so the broadcast payload is longer than, but starts identically to, what the host sent; exact-equality matching silently reports 100% drops in that case too.

> [!NOTE]
> A message that's requested but genuinely never broadcast (e.g. a custom PSID like CARMA's `MobilityOperation` under `0xBFEE`, which some OBU hardware accepts as a request but can't actually transmit) will still show up as `Dropped (no match found)` in the outgoing/request-flow correlation — that's a real, meaningful finding when it's consistent across every run, not a script bug. Cross-check message counts on each side (printed before the correlation section) to tell the two cases apart.

### Stale vs. dropped vs. outside the recording window

A match with latency above `--drop-threshold-ms` (default 200ms — calibrated for periodic safety-broadcast freshness, where a late message has likely been superseded by a newer one) is reported separately as **stale**, not folded into **dropped**. A one-off/request-driven message (like a manually broadcast test message) that took 1.6 seconds to go out is a real, measurable latency worth its own stats, not evidence it was lost — dropped means no matching payload was found at all.

When no match is found, it's further split by whether rx's own capture was actually recording at that moment: a tx message timestamped before rx's first packet or after rx's last packet is reported as **outside rx's recording window**, not **dropped**. Two independent `tcpdump`/`tshark` captures essentially never start and stop at exactly the same instant, so some tx messages near either edge are mechanically impossible for rx to have caught - that's not evidence of loss, it's evidence the two capture windows don't perfectly overlap. In practice this accounts for the overwhelming majority of "no match found" results on real captures - cross-check the `dropped` count specifically (not `dropped + outside window` combined) before treating a result as a real drop.

Usage:
```
python3 correlate_j2735_latency.py --tx-pcap earlier.pcap --rx-pcap later.pcap [--label NAME] [--drop-threshold-ms 200] [--json-out results.json]
```

- `--tx-pcap`: the earlier/source capture point
- `--rx-pcap`: the later/downstream capture point
- Requires `tshark` and `pycrate` (see `requirements.txt`); does not require `pyshark`.

## Correlating Latency Across the ROS<->Ethernet Boundary

`correlate_pcap_mcap.py` correlates an OBU's host-facing `eth0` pcap against an mcap recording of the ROS2 `carma_driver_msgs/msg/ByteArray` topics the comms driver publishes/subscribes to, to check whether messages crossing that pcap<->ROS boundary keep up or get dropped:

- **inbound**: `eth0` pcap → `/hardware_interface/comms/inbound_binary_msg` (message arrives over ethernet from the OBU, then the driver is expected to publish it to ROS)
- **outbound**: `/hardware_interface/comms/outbound_binary_msg` → `eth0` pcap (ROS software asks the driver to transmit, then the driver is expected to put it on the wire)

It reuses `correlate_j2735_latency.py`'s `extract_messages()` unchanged for the pcap side (same three vendor protocols, same 1609.2-envelope-aware parsing) - only the mcap side and the tx/rx matching are new. The `ByteArray.content` field is the raw J2735 MessageFrame bytes with no 1609.2 envelope, which is exactly the on-the-wire format `extract_messages()` already produces, so the two sides match byte-for-byte with no format translation.

> [!NOTE]
> Unlike `correlate_j2735_latency.py`'s `correlate()`, matching here does **not** require `rx_ts >= tx_ts`. Both timestamps come from the same onboard clock, but pcap frame-capture time and ROS message-stamp-assignment time aren't causally ordered at sub-millisecond granularity - confirmed on run-1 Commsignia, where genuinely-matching payloads showed the mcap stamp up to ~0.5ms *before* the pcap timestamp. A generous symmetric ±10s window (`MAX_ABS_MATCH_WINDOW_S`) picks whichever unconsumed candidate is closest in time either direction, rather than enforcing strict causal order.

> [!NOTE]
> mcap files are read with the low-level sequential `StreamReader` rather than `McapReader.iter_messages()`: every capture under `data/` is missing its trailing Footer/magic (the recording process was killed rather than shut down cleanly), which makes the indexed/summary-based reader fail outright. Sequential reading tolerates this the same way `read_classic_pcap` tolerates a truncated trailing packet - it stops at the first unreadable record and uses whatever was read so far, logging a warning rather than erroring out.

### Static/duplicate-payload message types skew this correlation

Matching is entirely payload-content-based (same `msg_type` + identical `payload_hex`) - there's no sequence number or request ID tying a specific ROS publish to a specific wire packet. That's unambiguous for types whose content is unique per instance (BSM, SPAT, SDSM all embed fresh position/time data), but breaks down for types whose payload never changes: a fixed-geometry MAP, or a custom-PSID request carrying a static test string. When every instance of a type is byte-identical, every tx event has *every* rx instance of that type as an equally-valid-looking candidate, and the algorithm falls back to "closest unconsumed one in time" - which is a guess, not a proven causal link.

In practice this shows up as two very different failure signatures, confirmed on real Ettifos captures:

- **A wide, scattered spread** of "matched but stale" latencies for a type → genuine matching ambiguity; treat the latency numbers for that type with caution.
- **A tight, consistent offset** across the *entire* run (e.g. every matched pair lands within a few ms of some constant N-message shift, checked start-to-finish with no jumps) → not ambiguity noise. This is far more likely a one-time pipeline/session warm-up delay: shifting the matched series by a constant message count (equivalently, a constant number of seconds at that type's broadcast rate) collapses the residual to single-digit milliseconds for the whole run, which random duplicate-picking would not produce. Confirmed on Ettifos run-1 (`MobilityOperation`, outbound, ~7s/70-message constant offset) and run-3/run-4 (`MAP`, inbound, ~2-3s/2-3-message constant offset) - present on some runs and completely absent on others, consistent with "how long this particular session's connection took to come up" rather than a fixed protocol timer or per-message network latency.

Use `find_duplicate_content_types()` (in `correlation_plots.py`) to flag which message types in a given capture/mcap pair are static before trusting their latency numbers at face value.

Usage:
```
python3 correlate_pcap_mcap.py --eth0-pcap eth0.pcap --mcap run.mcap [--label NAME] [--drop-threshold-ms 200] [--json-out results.json] [--plot-dir plots/]
```

- `--eth0-pcap`: host-facing ethernet capture (e.g. `run-N-eth0-*.pcap`)
- `--mcap`: mcap recording covering the same time window
- `--plot-dir`: optional directory to write per-flow latency/drop PNGs to (via `correlation_plots.plot_flow`)
- Requires `tshark`, `pycrate`, `mcap`, and `mcap_ros2` (see `requirements.txt`); does not require `pyshark`.

## Aggregating Latency Across Multiple Runs and OBU Vendors

`aggregate_boundary_latency.py` pools `correlate_pcap_mcap.py`'s per-run results across every run for one or more OBU vendor directories, so the combined mean/median/p95/p99/max are computed over the actual pooled sample set rather than approximated from each run's own summary stats.

Runs are discovered automatically within each vendor directory by pairing `run-N-eth0-*.pcap` with whichever `run-N*.mcap` exists alongside it - mcap naming isn't fully consistent across recordings (e.g. one run's mcap carries an extra descriptive suffix, another vendor's run-4 mcap is missing its usual `-<vendor>` suffix), so pairing is done by run number, not exact filename.

Output is three tables: per-run detail, a pooled-per-OBU summary (inbound/outbound each), and pooled latency broken down by message type - the last one flagged with the same duplicate/static-payload caveat described above, so a MAP or custom-PSID-request latency number isn't mistaken for the same kind of measurement as a BSM/SPAT/SDSM one.

Usage:
```
python3 aggregate_boundary_latency.py --vendor Ettifos=/path/to/ettifos --vendor Commsignia=/path/to/commsignia-v2 [--drop-threshold-ms 200]
```

- `--vendor NAME=DIR`: repeatable; a vendor label and the directory containing its `run-N-eth0-*.pcap`/`run-N*.mcap` pairs
- Requires the same dependencies as `correlate_pcap_mcap.py`

### Version
Version 1.0 - June 27, 2025

--------------------------------------------------
**Attribution (alphabetical order):**

- Andrew Fortier - R&D Engineer (Leidos)
- Paul Bourelly - Software Engineer (Leidos)
- William Martin - CAV R&D Engineer (Leidos)
