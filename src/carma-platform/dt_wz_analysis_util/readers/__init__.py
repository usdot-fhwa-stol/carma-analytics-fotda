"""Readers for the recording and log formats a verification session produces.

One module per format, each returning plain dicts or numpy arrays. Together they
let the analysis run on a plain Python venv: no ROS 2, no ``tshark``, no
``pycrate``.

======================  ====================================================
``mcap_reader``         CARMA Platform rosbags. Signature-compatible with
                        ``parse_ros2_bags`` so it can stand in where ROS 2 is
                        absent, decoding CDR via ``cdr``.
``cdr``                 CDR decoding driven by the ros2msg schema text rosbag2
                        stores inside the MCAP, so no generated message
                        packages are needed. Used only by ``mcap_reader``.
``pcap_reader``         J2735 messages from a classic pcap. Timestamps come
                        from the record header and matching is on payload
                        bytes, so neither tshark nor an ASN.1 decoder is
                        required.
``tcpdump_text``        OBU captures saved as tcpdump console output rather
                        than pcap: timestamps and lengths, no payload.
``obu_capture``         Dispatches on which of those two an OBU file actually
                        is, since both are named ``.pcap``.
``kafka_log``           kafka-console-consumer dumps, tolerant of both the
                        tab- and pipe-delimited styles in circulation.
``flir_websocket``      The camera's raw websocket stream out of the pc2
                        V2XHub log -- the closest measurement point to the
                        camera itself.
======================  ====================================================

Timestamp normalisation (``timeutil``) and log-line pairing (``pairing``) sit in
the parent package: they are used by the parsers and metrics as well as here,
and neither reads a format.
"""

from . import (
    cdr,
    flir_websocket,
    kafka_log,
    mcap_reader,
    obu_capture,
    pcap_reader,
    tcpdump_text,
)

__all__ = [
    "cdr",
    "flir_websocket",
    "kafka_log",
    "mcap_reader",
    "obu_capture",
    "pcap_reader",
    "tcpdump_text",
]
