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

For captures with a Linux-cooked link layer (`rmnet`-style), packets are also classified `incoming` (from the infrastructure/RSU) vs. `outgoing` (the device's own broadcast, e.g. a BSM it transmits itself). Only `incoming` tx messages are matched against the rx capture — `outgoing` messages are a structurally different flow not expected to reappear downstream, and are reported separately so they don't inflate the apparent drop rate.

Usage:
```
python3 correlate_j2735_latency.py --tx-pcap earlier.pcap --rx-pcap later.pcap [--label NAME] [--drop-threshold-ms 200] [--json-out results.json]
```

- `--tx-pcap`: the earlier/source capture point
- `--rx-pcap`: the later/downstream capture point
- A matched message with latency above `--drop-threshold-ms` (default 200ms) is counted as a drop, not a latency sample — it's stale by then.
- Requires `tshark` and `pycrate` (see `requirements.txt`); does not require `pyshark`.

### Version
Version 1.0 - June 27, 2025

--------------------------------------------------
**Attribution (alphabetical order):**

- Andrew Fortier - R&D Engineer (Leidos)
- Paul Bourelly - Software Engineer (Leidos)
- William Martin - CAV R&D Engineer (Leidos)