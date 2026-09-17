"""Drop-in replacement for the ``rosbag2_py`` half of ``parse_ros2_bags``.

Same function names, signatures and return shapes, so ``extract_mcap_data`` and
``open_bagfile`` can be swapped in wherever ROS 2 is not installed. Messages are
decoded by ``portable.cdr`` using the ``ros2msg`` schema text rosbag2 stores
inside the MCAP, so no generated message packages are needed either.

One deliberate difference from the ROS implementation: the recording origin is
taken from the MCAP summary rather than by calling ``read_next()``. The ROS
version consumes the bag's first message to learn the start time and never
replays it, which silently drops that message if it happens to be on a topic
being extracted. Reading the summary gets the same number without the side
effect.

Decoded messages are wrapped so field extractors written against ROS message
objects -- ``lambda msg: msg.state``, ``lambda msg: msg.header.stamp`` -- work
unchanged against the decoded dicts.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
from mcap.reader import make_reader

from . import cdr

# ByteArray.message_type values, mapped to the short names the pcap side uses.
MCAP_TYPE_ALIASES = {"SensorDataSharingMessage": "SDSM"}

INBOUND_BINARY_TOPIC = "/hardware_interface/comms/inbound_binary_msg"
OUTBOUND_BINARY_TOPIC = "/hardware_interface/comms/outbound_binary_msg"


class Message:
    """Attribute access over a decoded CDR dict, recursively.

    Field extractors in this repo are written as ``lambda msg: msg.header.stamp``
    against ROS message objects. Wrapping keeps those callers working rather than
    requiring every one to be rewritten for dict access.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise AttributeError(name)
        return _wrap(data[name])

    def __getitem__(self, key):
        return _wrap(object.__getattribute__(self, "_data")[key])

    def __contains__(self, key):
        return key in object.__getattribute__(self, "_data")

    def get(self, key, default=None):
        data = object.__getattribute__(self, "_data")
        return _wrap(data[key]) if key in data else default

    def to_dict(self) -> dict:
        return object.__getattribute__(self, "_data")

    def __repr__(self):
        return f"Message({object.__getattribute__(self, '_data')!r})"


def _wrap(value):
    if isinstance(value, dict):
        return Message(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


class SequentialReader:
    """Minimal stand-in for ``rosbag2_py.SequentialReader``.

    Implements only ``has_next`` / ``read_next`` / ``set_filter``, which is all
    ``parse_ros2_bags`` uses. ``read_next`` returns the *decoded and wrapped*
    message rather than raw bytes, since decoding here needs the schema table
    that only this reader holds.
    """

    def __init__(self, path, topics=None):
        self._handle = open(path, "rb")
        self._reader = make_reader(self._handle)
        summary = self._reader.get_summary()
        if summary is None:
            raise ValueError(
                f"{path}: no MCAP summary; the file is likely truncated. "
                f"Use the recovered copy of this bag if one exists."
            )
        self._summary = summary
        self.type_map = {
            channel.topic: summary.schemas[channel.schema_id].name
            for channel in summary.channels.values()
        }
        self._topics = set(topics) if topics else None
        self._iter = None
        self._pending = None
        self._decoders = None

    def set_filter(self, topics):
        self._topics = set(topics)
        self._iter = None
        self._pending = None

    @property
    def start_time_ns(self) -> int:
        statistics = self._summary.statistics
        if statistics is not None and statistics.message_start_time:
            return int(statistics.message_start_time)
        for _schema, _channel, message in self._reader.iter_messages():
            return int(message.log_time)
        raise ValueError("No valid timestamps found in the ROS2 bag")

    def _ensure_iter(self):
        if self._iter is not None:
            return
        topics = sorted(self._topics) if self._topics else None
        self._decoders = cdr.make_decoders(self._summary, set(topics) if topics else set(self.type_map))
        self._iter = self._reader.iter_messages(topics=topics)

    def has_next(self) -> bool:
        self._ensure_iter()
        if self._pending is not None:
            return True
        for _schema, channel, message in self._iter:
            decoder = self._decoders.get(channel.id)
            if decoder is None:
                continue
            self._pending = (channel.topic, Message(decoder[1].decode(message.data)), int(message.log_time))
            return True
        return False

    def read_next(self):
        if not self.has_next():
            raise StopIteration("No more messages")
        pending, self._pending = self._pending, None
        return pending

    def close(self):
        self._handle.close()

    def __del__(self):
        try:
            self._handle.close()
        except Exception:
            pass


def open_bagfile(path, topics=[], serialization_format="cdr", storage_id="mcap"):
    """Open an MCAP and return ``(reader, type_map, earliest_time_ns)``.

    Signature-compatible with ``parse_ros2_bags.open_bagfile``. The
    ``serialization_format``/``storage_id`` arguments are accepted and ignored;
    this backend reads MCAP with CDR only.
    """
    reader = SequentialReader(path, topics=topics or None)
    return reader, reader.type_map, reader.start_time_ns


def check_mcap_file_existence(mcap_path):
    if not os.path.exists(mcap_path):
        raise ValueError(f"MCAP file {mcap_path} does not exist")


def initialize_field_extractors(topics, field_extractors):
    if field_extractors is None:
        return {topic: lambda msg: msg for topic in topics}
    missing_extractors = set(topics) - set(field_extractors.keys())
    if missing_extractors:
        raise ValueError(f"Missing field extractors for topics: {missing_extractors}")
    return field_extractors


def check_missing_topics(topics, type_map):
    missing_topics = set(topics) - set(type_map.keys())
    if missing_topics:
        raise ValueError(f"Topics not found in MCAP file: {missing_topics}")


def read_messages(reader, topics, type_map, field_extractors):
    """Read and extract per-topic values; mirrors ``parse_ros2_bags.read_messages``."""
    data = {topic: {"values": [], "timestamps": []} for topic in topics}
    while reader.has_next():
        topic, message, timestamp = reader.read_next()
        if topic not in topics:
            continue
        try:
            data[topic]["values"].append(field_extractors[topic](message))
            data[topic]["timestamps"].append(timestamp)
        except Exception as error:
            print(f"Warning: Failed to extract data from message on topic {topic}: {error}")
    return data


def filter_data_with_start_and_end_time(
    data, topics, global_start_time, start_time, end_time, use_relative_time=True
):
    """Window each topic; mirrors ``parse_ros2_bags`` including its time bases.

    With ``use_relative_time`` the timestamps come back as seconds since the
    recording start. Without it they stay absolute nanoseconds and the bounds --
    which callers still pass as relative seconds -- are converted up to absolute.
    """
    result = {}
    for topic, topic_data in data.items():
        timestamps = np.array(topic_data["timestamps"], dtype=float)
        values = np.array(topic_data["values"], dtype=object)

        if use_relative_time:
            timestamps = (timestamps - global_start_time) / 1e9
            low, high = start_time, end_time
        else:
            low = None if start_time is None else start_time * 1e9 + global_start_time
            high = None if end_time is None else end_time * 1e9 + global_start_time

        if low is not None or high is not None:
            mask = np.ones_like(timestamps, dtype=bool)
            if low is not None:
                mask &= timestamps >= low
            if high is not None:
                mask &= timestamps <= high
            timestamps = timestamps[mask]
            values = values[mask]
            if len(timestamps) == 0:
                raise ValueError(f"No data found for topic {topic} in specified time range")

        result[topic] = (timestamps, values)
    return result


def extract_mcap_data(
    mcap_path, topics, start_time=None, end_time=None, field_extractors=None, use_relative_time=True
):
    """Extract per-topic ``(timestamps, values)``; mirrors ``parse_ros2_bags``."""
    check_mcap_file_existence(mcap_path)
    field_extractors = initialize_field_extractors(topics, field_extractors)

    reader, type_map, global_start_time = open_bagfile(str(mcap_path), topics=topics)
    check_missing_topics(topics, type_map)

    data = read_messages(reader, topics, type_map, field_extractors)
    empty_topics = [topic for topic, topic_data in data.items() if not topic_data["values"]]
    if empty_topics:
        raise ValueError(f"No valid messages found for topics: {empty_topics}")

    return filter_data_with_start_and_end_time(
        data, topics, global_start_time, start_time, end_time, use_relative_time
    )


def extract_mcap_binary_messages(mcap_path) -> Dict[str, List[Dict]]:
    """ByteArray traffic as ``{"inbound": [...], "outbound": [...]}``.

    Replaces ``correlate_pcap_mcap.extract_mcap_binary_messages``. Each entry is
    ``{timestamp, msg_type, payload_hex}`` with ``timestamp`` in epoch seconds
    taken from the message's own ``header.stamp`` -- the driver's receive
    instant, which is what correlates against a pcap -- falling back to the
    recorder's log time when the header is unset.
    """
    result: Dict[str, List[Dict]] = {"inbound": [], "outbound": []}
    topics = [INBOUND_BINARY_TOPIC, OUTBOUND_BINARY_TOPIC]

    with open(mcap_path, "rb") as handle:
        reader = make_reader(handle)
        summary = reader.get_summary()
        if summary is None:
            raise ValueError(f"{mcap_path}: no MCAP summary; the file is likely truncated")
        present = {channel.topic for channel in summary.channels.values()}
        wanted = [topic for topic in topics if topic in present]
        if not wanted:
            return result
        decoders = cdr.make_decoders(summary, set(wanted))

        for _schema, channel, message in reader.iter_messages(topics=wanted):
            decoder = decoders.get(channel.id)
            if decoder is None:
                continue
            value = decoder[1].decode(message.data)
            stamp = (value.get("header") or {}).get("stamp") or {}
            seconds = stamp.get("sec")
            if seconds:
                timestamp = float(seconds) + float(stamp.get("nanosec", 0)) / 1e9
            else:
                timestamp = message.log_time / 1e9
            msg_type = value.get("message_type", "")
            content = value.get("content", b"")
            result["inbound" if channel.topic == INBOUND_BINARY_TOPIC else "outbound"].append(
                {
                    "timestamp": timestamp,
                    "msg_type": MCAP_TYPE_ALIASES.get(msg_type, msg_type),
                    "payload_hex": content.hex() if isinstance(content, (bytes, bytearray)) else "",
                }
            )
    return result


def topic_message_counts(mcap_path) -> Dict[str, int]:
    """Per-topic message counts from the summary, without decoding anything."""
    with open(mcap_path, "rb") as handle:
        summary = make_reader(handle).get_summary()
        if summary is None:
            raise ValueError(f"{mcap_path}: no MCAP summary; the file is likely truncated")
        statistics = summary.statistics
        return {
            channel.topic: (
                statistics.channel_message_counts.get(channel.id, 0) if statistics else 0
            )
            for channel in summary.channels.values()
        }
