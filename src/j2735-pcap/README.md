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

### Version
Version 1.0 - June 27, 2025

--------------------------------------------------
**Attribution (alphabetical order):**

- Andrew Fortier - R&D Engineer (Leidos)
- Paul Bourelly - Software Engineer (Leidos)
- William Martin - CAV R&D Engineer (Leidos)