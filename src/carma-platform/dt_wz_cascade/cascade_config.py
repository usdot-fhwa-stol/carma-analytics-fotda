"""Stage definitions for the end-to-end SDSM latency cascade.

One row per FLIR camera detection, one column per stage it reached, each an
absolute epoch-millisecond timestamp. Stages are listed in causal order so
consecutive differences are the per-hop latencies.

Two stages differ from earlier sessions and are **not** interchangeable with the
similarly-named ones in those tables:

``t_rsu_broadcast``
    Previously ``t_ota_capture``, taken from a pcap on the *OBU*, so it meant
    "the SDSM arrived over the air". This session captured the *RSU* instead, so
    it now means "the RSU put the SDSM on the air" -- earlier in the chain by one
    propagation hop. End-to-end totals are therefore slightly smaller than the
    2026-09-11/2026-09-14_01 figures for reasons that have nothing to do with the
    system getting faster.

``t_obu_radio_rx``
    New. The OBU's own receive instant, from its tcpdump text capture. Because
    that capture carries no payload, this stage is paired by **time order within
    the run**, not by payload identity like every other stage here. It is
    reliable in aggregate and should not be trusted for any individual row.
"""

from __future__ import annotations

# V2XHub logs the bracketed wall clock in UTC; the SDSS logs local time (UTC-4).
V2XHUB_UTC_OFFSET_H = 0
SDSS_UTC_OFFSET_H = -4

# (column, host, label) in causal order.
STAGES = (
    ("t_flir_detect", "flir", "FLIR detection timestamp (sensor-reported)"),
    ("t_flir_log", "pc2", "FLIRCameraDriverPlugin publishes to TMX bus"),
    ("t_streets_rx", "pc2", "CARMAStreetsPlugin receives SensorDetectedObject"),
    ("t_kafka_produce", "pc2", "CARMAStreetsPlugin produces to Kafka"),
    ("t_kafka_sdo_create", "pc2", "Kafka broker CreateTime (detection topic)"),
    ("t_sdss_consume", "pc2", "sensor_data_sharing_service consumes detection"),
    ("t_sdss_send", "pc2", "sensor_data_sharing_service sends SDSM"),
    ("t_kafka_sdsm_create", "pc2", "Kafka broker CreateTime (SDSM topic)"),
    ("t_streets_sdsm_rx", "pc2", "CARMAStreetsPlugin consumes SDSM"),
    ("t_streets_encode", "pc2", "CARMAStreetsPlugin UPER-encodes SDSM"),
    ("t_pc2_tena_rx", "pc2", "TenaV2XPlugin reads SDSM off TMX bus"),
    ("t_pc2_tena_tx", "pc2", "TenaV2XPlugin sends over TENA"),
    ("t_pc1_tena_rx", "pc1", "TenaV2XPlugin observer receives from TENA"),
    ("t_pc1_bus", "pc1", "SDSM published onto pc1 TMX bus"),
    ("t_immediate_fwd", "pc1", "ImmediateForwardPlugin sends to RSU"),
    ("t_rsu_broadcast", "rsu", "RSU broadcasts SDSM over the air"),
    ("t_obu_radio_rx", "obu", "OBU radio receives SDSM (count-paired)"),
    ("t_ros_inbound", "ros", "inbound_binary_msg (recorder receive)"),
    ("t_ros_j3224", "ros", "incoming_j3224_sdsm (recorder receive)"),
    ("t_ros_fused", "ros", "fused_external_objects (recorder receive)"),
)

STAGE_COLUMNS = tuple(column for column, _host, _label in STAGES)
STAGE_HOST = {column: host for column, host, _label in STAGES}
STAGE_LABEL = {column: label for column, _host, label in STAGES}

# The subset worth plotting by default: one stage per meaningful handoff, so the
# cascade stays readable instead of showing nineteen near-coincident dots.
MILESTONE_STAGES = (
    "t_flir_detect",
    "t_flir_log",
    "t_kafka_sdo_create",
    "t_sdss_consume",
    "t_sdss_send",
    "t_pc2_tena_tx",
    "t_immediate_fwd",
    "t_rsu_broadcast",
    "t_obu_radio_rx",
    "t_ros_inbound",
    "t_ros_fused",
)

STAGE_SHORT_LABEL = {
    "t_flir_detect": "camera detection",
    "t_flir_log": "V2XHub ingest",
    "t_kafka_sdo_create": "onto Kafka",
    "t_sdss_consume": "SDSS receives",
    "t_sdss_send": "SDSS emits",
    "t_pc2_tena_tx": "onto TENA",
    "t_immediate_fwd": "to RSU",
    "t_rsu_broadcast": "broadcast",
    "t_obu_radio_rx": "OBU radio",
    "t_ros_inbound": "vehicle ROS inbound",
    "t_ros_fused": "fused output",
}

# Stages recorded by the pc1 host, whose clock runs slightly behind pc2's.
PC1_STAGES = ("t_pc1_tena_rx", "t_pc1_bus", "t_immediate_fwd")

# Stages joined on the SDSM's UPER payload bytes rather than on detection identity.
UPER_KEYED_STAGES = frozenset(
    {
        "t_streets_encode",
        "t_pc2_tena_rx",
        "t_pc2_tena_tx",
        "t_pc1_tena_rx",
        "t_pc1_bus",
        "t_immediate_fwd",
        "t_rsu_broadcast",
        "t_ros_inbound",
    }
)

# Stage paired by ordinal position within the run rather than by identity.
COUNT_PAIRED_STAGES = frozenset({"t_obu_radio_rx"})

ROS_TOPICS = (
    "/hardware_interface/comms/inbound_binary_msg",
    "/message/incoming_j3224_sdsm",
    "/environment/fused_external_objects",
)

# Tolerance when matching a fused track back to the detection that produced it.
FUSED_MATCH_WINDOW_MS = 600.0
FUSED_MATCH_RADIUS_M = 15.0
