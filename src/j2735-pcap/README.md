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

### Version
Version 1.0 - June 27, 2025

--------------------------------------------------
**Attribution (alphabetical order):**

- Andrew Fortier - R&D Engineer (Leidos)
- Paul Bourelly - Software Engineer (Leidos)
- William Martin - CAV R&D Engineer (Leidos)
