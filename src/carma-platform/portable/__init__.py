"""Self-contained readers for MCAP, pcap, tcpdump-text and Kafka console dumps.

The rest of ``src/carma-platform`` reads rosbags through ``rosbag2_py`` and pcaps
through ``tshark`` + ``pycrate``. Neither is available on every machine that has
to look at this data -- an analyst laptop generally has a Python venv and nothing
else -- so each reader here reproduces just the part of that stack the DT-WZ
metrics actually use, depending only on ``pandas``/``numpy``/``mcap``.

The MCAP side is the interesting one: instead of needing the generated
``carma_v2x_msgs``/``carma_driver_msgs`` Python packages, it decodes CDR using
the ``ros2msg`` schema text that rosbag2 already stores inside the file. That
makes the readers self-describing -- a bag records everything needed to read
itself back.

These are *fallbacks*, wired in where the ROS-based path raises ImportError, so
behaviour is unchanged wherever the full stack is installed.
"""

from . import cdr, kafka_log, mcap_backend, pairing, pcap_backend, tcpdump_text, timeutil

__all__ = [
    "cdr",
    "kafka_log",
    "mcap_backend",
    "pairing",
    "pcap_backend",
    "tcpdump_text",
    "timeutil",
]
