#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_TOPIC_BY_METRIC = {
    "CDAS-8": "/clock",
    "CDAS-9": "/localization/current_pose",
    "CDAS-10": "/hardware_interface/controller/robot_status",
    "CDAS-11": "/rosout",
    "CDAS-13": "/rosout",
    "CDAS-14": "/rosout",
}

DEFAULT_CDAS_11_PLUGINS = [
    "/guidance/plugins/pure_pursuit_wrapper",
    "/guidance/plugins/stop_and_wait_plugin",
    "/guidance/plugins/inlanecruising_plugin",
    "/guidance/plugins/cooperative_lanechange",
    "/guidance/plugins/route_following_plugin",
]


@dataclass
class TopicSample:
    index: int
    receive_timestamp_ns: int
    extracted_time_ns: Optional[int]


@dataclass
class RosoutEntry:
    index: int
    receive_timestamp_ns: int
    stamp_ns: Optional[int]
    name: str
    level: int
    msg: str


@dataclass
class MetricResult:
    metric_id: str
    topic: str
    status: str
    summary: str
    details: List[str]


def import_rosbag_tools():
    try:
        import rosbag2_py
    except ImportError as exc:
        raise RuntimeError(
            "Could not import rosbag2_py. Source ROS 2 first, for example:\n"
            "  source /opt/ros/humble/setup.bash"
        ) from exc

    return rosbag2_py


def import_deserialization_tools():
    try:
        from rosidl_runtime_py.utilities import get_message
        from rclpy.serialization import deserialize_message
    except ImportError as exc:
        raise RuntimeError(
            "Could not import ROS 2 deserialization tools. Source ROS 2 first, for example:\n"
            "  source /opt/ros/humble/setup.bash"
        ) from exc

    return get_message, deserialize_message


def open_reader(bag_path: Path, storage_id: str):
    rosbag2_py = import_rosbag_tools()

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="",
        output_serialization_format="",
    )

    reader.open(storage_options, converter_options)
    return reader, rosbag2_py


def get_topic_type_map(reader) -> Dict[str, str]:
    return {item.name: item.type for item in reader.get_all_topics_and_types()}


def list_topics(bag_path: Path, storage_id: str) -> int:
    reader, _ = open_reader(bag_path, storage_id)
    topic_type_map = get_topic_type_map(reader)

    if not topic_type_map:
        print("No topics found.")
        return 1

    print("Topics found:")
    for topic in sorted(topic_type_map):
        print(f"  {topic}\t{topic_type_map[topic]}")
    return 0


def set_topic_filter_if_available(reader, rosbag2_py, topic: str) -> None:
    try:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    except Exception:
        pass


def time_msg_to_ns(time_msg) -> Optional[int]:
    sec = getattr(time_msg, "sec", None)
    nanosec = getattr(time_msg, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)


def extract_clock_time_ns(msg) -> Optional[int]:
    if hasattr(msg, "clock"):
        return time_msg_to_ns(msg.clock)
    return time_msg_to_ns(msg)


def read_topic_samples(
    bag_path: Path,
    storage_id: str,
    topic: str,
    max_messages: Optional[int],
    deserialize: bool,
    extract_clock_time: bool,
) -> tuple[List[TopicSample], str]:
    reader, rosbag2_py = open_reader(bag_path, storage_id)
    topic_type_map = get_topic_type_map(reader)

    if topic not in topic_type_map:
        available = "\n".join(
            f"  {name}\t{topic_type_map[name]}" for name in sorted(topic_type_map)
        )
        raise ValueError(f"Topic not found: {topic}\nAvailable topics:\n{available}")

    topic_type = topic_type_map[topic]
    set_topic_filter_if_available(reader, rosbag2_py, topic)

    msg_class = None
    deserialize_message = None

    if deserialize:
        get_message, deserialize_message = import_deserialization_tools()
        msg_class = get_message(topic_type)

    samples: List[TopicSample] = []

    while reader.has_next():
        read_topic_name, serialized_msg, receive_timestamp_ns = reader.read_next()
        if read_topic_name != topic:
            continue

        extracted_time_ns = None

        if deserialize and extract_clock_time:
            msg = deserialize_message(serialized_msg, msg_class)
            extracted_time_ns = extract_clock_time_ns(msg)

        samples.append(
            TopicSample(
                index=len(samples) + 1,
                receive_timestamp_ns=int(receive_timestamp_ns),
                extracted_time_ns=extracted_time_ns,
            )
        )

        if max_messages is not None and len(samples) >= max_messages:
            break

    return samples, topic_type


def read_rosout_entries(
    bag_path: Path,
    storage_id: str,
    topic: str,
    max_messages: Optional[int],
) -> tuple[List[RosoutEntry], str]:
    reader, rosbag2_py = open_reader(bag_path, storage_id)
    topic_type_map = get_topic_type_map(reader)

    if topic not in topic_type_map:
        available = "\n".join(
            f"  {name}\t{topic_type_map[name]}" for name in sorted(topic_type_map)
        )
        raise ValueError(f"Topic not found: {topic}\nAvailable topics:\n{available}")

    topic_type = topic_type_map[topic]
    get_message, deserialize_message = import_deserialization_tools()
    msg_class = get_message(topic_type)

    set_topic_filter_if_available(reader, rosbag2_py, topic)

    entries: List[RosoutEntry] = []

    while reader.has_next():
        read_topic_name, serialized_msg, receive_timestamp_ns = reader.read_next()
        if read_topic_name != topic:
            continue

        msg = deserialize_message(serialized_msg, msg_class)

        stamp_ns = None
        if hasattr(msg, "stamp"):
            stamp_ns = time_msg_to_ns(msg.stamp)

        entries.append(
            RosoutEntry(
                index=len(entries) + 1,
                receive_timestamp_ns=int(receive_timestamp_ns),
                stamp_ns=stamp_ns,
                name=str(getattr(msg, "name", "")),
                level=int(getattr(msg, "level", 0)),
                msg=str(getattr(msg, "msg", "")),
            )
        )

        if max_messages is not None and len(entries) >= max_messages:
            break

    return entries, topic_type


def write_rosout_csv(csv_path: Path, entries: List[RosoutEntry]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "receive_timestamp_ns",
                "receive_timestamp_sec",
                "stamp_ns",
                "stamp_sec",
                "name",
                "level",
                "msg",
            ],
        )
        writer.writeheader()

        for entry in entries:
            writer.writerow({
                "index": entry.index,
                "receive_timestamp_ns": entry.receive_timestamp_ns,
                "receive_timestamp_sec": entry.receive_timestamp_ns / 1_000_000_000.0,
                "stamp_ns": entry.stamp_ns if entry.stamp_ns is not None else "",
                "stamp_sec": (
                    entry.stamp_ns / 1_000_000_000.0
                    if entry.stamp_ns is not None
                    else ""
                ),
                "name": entry.name,
                "level": entry.level,
                "msg": entry.msg,
            })


def write_topic_csv(csv_path: Path, samples: List[TopicSample], include_extracted_time: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "receive_timestamp_ns",
            "receive_timestamp_sec",
            "receive_delta_sec",
        ]

        if include_extracted_time:
            fieldnames.extend([
                "extracted_time_ns",
                "extracted_time_sec",
                "extracted_delta_sec",
            ])

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        prev_receive_ns = None
        prev_extracted_ns = None

        for sample in samples:
            row = {
                "index": sample.index,
                "receive_timestamp_ns": sample.receive_timestamp_ns,
                "receive_timestamp_sec": sample.receive_timestamp_ns / 1_000_000_000.0,
                "receive_delta_sec": "",
            }

            if prev_receive_ns is not None:
                row["receive_delta_sec"] = (
                    f"{(sample.receive_timestamp_ns - prev_receive_ns) / 1_000_000_000.0:.9f}"
                )

            if include_extracted_time:
                row["extracted_time_ns"] = (
                    sample.extracted_time_ns if sample.extracted_time_ns is not None else ""
                )
                row["extracted_time_sec"] = (
                    sample.extracted_time_ns / 1_000_000_000.0
                    if sample.extracted_time_ns is not None
                    else ""
                )
                row["extracted_delta_sec"] = ""

                if prev_extracted_ns is not None and sample.extracted_time_ns is not None:
                    row["extracted_delta_sec"] = (
                        f"{(sample.extracted_time_ns - prev_extracted_ns) / 1_000_000_000.0:.9f}"
                    )

            writer.writerow(row)

            prev_receive_ns = sample.receive_timestamp_ns
            if sample.extracted_time_ns is not None:
                prev_extracted_ns = sample.extracted_time_ns


def evaluate_clock_metric(
    bag_path: Path,
    storage_id: str,
    topic: str,
    threshold_sec: float,
    tolerance_sec: float,
    max_messages: Optional[int],
    csv_dir: Optional[Path],
) -> MetricResult:
    metric_id = "CDAS-8"
    details: List[str] = []

    try:
        samples, topic_type = read_topic_samples(
            bag_path=bag_path,
            storage_id=storage_id,
            topic=topic,
            max_messages=max_messages,
            deserialize=True,
            extract_clock_time=True,
        )
    except Exception as exc:
        return MetricResult(metric_id, topic, "FAIL", f"Failed to read {topic}: {exc}", details)

    threshold_ns = int(round(threshold_sec * 1_000_000_000))
    tolerance_ns = int(round(tolerance_sec * 1_000_000_000))
    allowed_ns = threshold_ns + tolerance_ns

    details.append(f"topic type: {topic_type}")
    details.append(f"message count: {len(samples)}")
    details.append(f"threshold sec: {threshold_sec}")
    details.append(f"tolerance sec: {tolerance_sec}")
    details.append(f"allowed interval ns: {allowed_ns}")

    if csv_dir:
        csv_path = csv_dir / "cdas_8_clock.csv"
        write_topic_csv(csv_path, samples, include_extracted_time=True)
        details.append(f"csv written: {csv_path}")

    if len(samples) == 0:
        return MetricResult(metric_id, topic, "FAIL", f"No messages found on {topic}.", details)

    if len(samples) == 1:
        return MetricResult(metric_id, topic, "FAIL", f"Only one message found on {topic}; cannot verify interval.", details)

    clock_times = [sample.extracted_time_ns for sample in samples if sample.extracted_time_ns is not None]

    if len(clock_times) != len(samples):
        details.append(f"messages without extractable /clock time: {len(samples) - len(clock_times)}")
        return MetricResult(metric_id, topic, "FAIL", "Could not extract simulated clock time from every /clock message.", details)

    deltas_ns = [clock_times[i] - clock_times[i - 1] for i in range(1, len(clock_times))]
    non_increasing = [delta for delta in deltas_ns if delta <= 0]
    over_threshold = [delta for delta in deltas_ns if delta > allowed_ns]

    details.append(f"first /clock time ns: {clock_times[0]}")
    details.append(f"last /clock time ns: {clock_times[-1]}")
    details.append(f"interval count: {len(deltas_ns)}")
    details.append(f"max interval sec: {max(deltas_ns) / 1_000_000_000.0:.9f}")
    details.append(f"average interval sec: {sum(deltas_ns) / len(deltas_ns) / 1_000_000_000.0:.9f}")
    details.append(f"intervals over threshold count: {len(over_threshold)}")
    details.append(f"non-increasing intervals count: {len(non_increasing)}")

    if non_increasing:
        return MetricResult(metric_id, topic, "FAIL", "Simulated /clock time did not strictly increase.", details)

    if over_threshold:
        return MetricResult(
            metric_id,
            topic,
            "FAIL",
            f"Simulated /clock interval exceeded {threshold_sec:.3f} sec {len(over_threshold)} time(s).",
            details,
        )

    return MetricResult(
        metric_id,
        topic,
        "PASS",
        f"Simulated /clock time advances within {threshold_sec:.3f} sec.",
        details,
    )


def evaluate_topic_publish_rate_metric(
    metric_id: str,
    bag_path: Path,
    storage_id: str,
    topic: str,
    threshold_sec: float,
    tolerance_sec: float,
    max_messages: Optional[int],
    csv_dir: Optional[Path],
) -> MetricResult:
    details: List[str] = []

    try:
        samples, topic_type = read_topic_samples(
            bag_path=bag_path,
            storage_id=storage_id,
            topic=topic,
            max_messages=max_messages,
            deserialize=False,
            extract_clock_time=False,
        )
    except Exception as exc:
        return MetricResult(metric_id, topic, "FAIL", f"Failed to read {topic}: {exc}", details)

    threshold_ns = int(round(threshold_sec * 1_000_000_000))
    tolerance_ns = int(round(tolerance_sec * 1_000_000_000))
    allowed_ns = threshold_ns + tolerance_ns

    details.append(f"topic type: {topic_type}")
    details.append(f"message count: {len(samples)}")
    details.append(f"threshold sec: {threshold_sec}")
    details.append(f"tolerance sec: {tolerance_sec}")
    details.append(f"allowed interval ns: {allowed_ns}")

    if csv_dir:
        safe_metric = metric_id.lower().replace("-", "_")
        safe_topic = topic.strip("/").replace("/", "_") or "root"
        csv_path = csv_dir / f"{safe_metric}_{safe_topic}.csv"
        write_topic_csv(csv_path, samples, include_extracted_time=False)
        details.append(f"csv written: {csv_path}")

    if len(samples) == 0:
        return MetricResult(metric_id, topic, "FAIL", f"No messages found on {topic}.", details)

    if len(samples) == 1:
        return MetricResult(metric_id, topic, "FAIL", f"Only one message found on {topic}; cannot verify publish interval.", details)

    receive_times = [sample.receive_timestamp_ns for sample in samples]
    deltas_ns = [receive_times[i] - receive_times[i - 1] for i in range(1, len(receive_times))]
    non_increasing = [delta for delta in deltas_ns if delta <= 0]
    over_threshold = [delta for delta in deltas_ns if delta > allowed_ns]

    details.append(f"first receive timestamp ns: {receive_times[0]}")
    details.append(f"last receive timestamp ns: {receive_times[-1]}")
    details.append(f"interval count: {len(deltas_ns)}")
    details.append(f"max interval sec: {max(deltas_ns) / 1_000_000_000.0:.9f}")
    details.append(f"average interval sec: {sum(deltas_ns) / len(deltas_ns) / 1_000_000_000.0:.9f}")
    details.append(f"intervals over threshold count: {len(over_threshold)}")
    details.append(f"non-increasing intervals count: {len(non_increasing)}")

    if non_increasing:
        return MetricResult(metric_id, topic, "FAIL", "Bag receive timestamps for this topic are not strictly increasing.", details)

    if over_threshold:
        return MetricResult(
            metric_id,
            topic,
            "FAIL",
            f"{topic} publish interval exceeded {threshold_sec:.3f} sec {len(over_threshold)} time(s).",
            details,
        )

    return MetricResult(
        metric_id,
        topic,
        "PASS",
        f"{topic} publishes within {threshold_sec:.3f} sec.",
        details,
    )


def evaluate_cdas_11(
    bag_path: Path,
    storage_id: str,
    topic: str,
    expected_plugins: List[str],
    max_messages: Optional[int],
    csv_dir: Optional[Path],
) -> MetricResult:
    metric_id = "CDAS-11"
    details: List[str] = []

    try:
        entries, topic_type = read_rosout_entries(
            bag_path=bag_path,
            storage_id=storage_id,
            topic=topic,
            max_messages=max_messages,
        )
    except Exception as exc:
        return MetricResult(metric_id, topic, "FAIL", f"Failed to read {topic}: {exc}", details)

    if csv_dir:
        csv_path = csv_dir / "cdas_11_rosout.csv"
        write_rosout_csv(csv_path, entries)
        details.append(f"csv written: {csv_path}")

    details.append(f"topic type: {topic_type}")
    details.append(f"rosout message count read: {len(entries)}")
    details.append(f"expected plugin count: {len(expected_plugins)}")

    plugin_hits: Dict[str, int] = {plugin: 0 for plugin in expected_plugins}
    route_success_count = 0
    guidance_engaged_count = 0
    arbitrator_engaged_count = 0

    plugin_pattern_by_name = {
        plugin: re.compile(re.escape(plugin) + r".*has been activated", re.IGNORECASE)
        for plugin in expected_plugins
    }
    route_pattern = re.compile(r"Call to set_active_route succeeded", re.IGNORECASE)
    guidance_pattern = re.compile(r"\bGuidance engaged\b", re.IGNORECASE)
    arbitrator_pattern = re.compile(r"Guidance has been engaged", re.IGNORECASE)

    for entry in entries:
        for plugin, pattern in plugin_pattern_by_name.items():
            if pattern.search(entry.msg):
                plugin_hits[plugin] += 1

        if route_pattern.search(entry.msg):
            route_success_count += 1

        if guidance_pattern.search(entry.msg):
            guidance_engaged_count += 1

        if arbitrator_pattern.search(entry.msg):
            arbitrator_engaged_count += 1

    missing_plugins = [plugin for plugin, count in plugin_hits.items() if count == 0]

    for plugin, count in plugin_hits.items():
        details.append(f"plugin activation count for {plugin}: {count}")

    details.append(f"set_active_route success count: {route_success_count}")
    details.append(f"carma_carla_guidance engaged count: {guidance_engaged_count}")
    details.append(f"guidance arbitrator engaged count: {arbitrator_engaged_count}")

    if missing_plugins:
        details.append(f"missing plugin activations: {missing_plugins}")

    if missing_plugins or route_success_count == 0 or guidance_engaged_count == 0 or arbitrator_engaged_count == 0:
        missing_parts = []
        if missing_plugins:
            missing_parts.append("plugin activation")
        if route_success_count == 0:
            missing_parts.append("set_active_route success")
        if guidance_engaged_count == 0:
            missing_parts.append("Guidance engaged")
        if arbitrator_engaged_count == 0:
            missing_parts.append("Guidance has been engaged")
        return MetricResult(
            metric_id,
            topic,
            "FAIL",
            "Missing CDAS-11 evidence: " + ", ".join(missing_parts),
            details,
        )

    return MetricResult(
        metric_id,
        topic,
        "PASS",
        "Expected plugins activated, route set successfully, and guidance engaged.",
        details,
    )


def evaluate_cdas_13(
    bag_path: Path,
    storage_id: str,
    topic: str,
    vehicle_name: Optional[str],
    max_messages: Optional[int],
    csv_dir: Optional[Path],
) -> MetricResult:
    metric_id = "CDAS-13"
    details: List[str] = []

    try:
        entries, topic_type = read_rosout_entries(
            bag_path=bag_path,
            storage_id=storage_id,
            topic=topic,
            max_messages=max_messages,
        )
    except Exception as exc:
        return MetricResult(metric_id, topic, "FAIL", f"Failed to read {topic}: {exc}", details)

    if csv_dir:
        csv_path = csv_dir / "cdas_13_rosout.csv"
        write_rosout_csv(csv_path, entries)
        details.append(f"csv written: {csv_path}")

    details.append(f"topic type: {topic_type}")
    details.append(f"rosout message count read: {len(entries)}")
    details.append(f"vehicle name: {vehicle_name or 'any vehicle'}")

    spawn_point_pattern = re.compile(r"\[Spawner\]\s+Received spawn_point parameter:", re.IGNORECASE)
    spawned_vehicle_pattern = re.compile(
        r"\[Spawner\]\s+Spawned vehicle\s+'?(?P<vehicle>[^'\s]+)'?",
        re.IGNORECASE,
    )

    spawn_point_count = 0
    spawned_vehicle_names: List[str] = []

    for entry in entries:
        if spawn_point_pattern.search(entry.msg):
            spawn_point_count += 1
        spawned_match = spawned_vehicle_pattern.search(entry.msg)
        if spawned_match:
            spawned_name = spawned_match.group("vehicle")
            if vehicle_name is None or spawned_name == vehicle_name:
                spawned_vehicle_names.append(spawned_name)

    spawned_vehicle_count = len(spawned_vehicle_names)

    details.append(f"spawner spawn_point parameter count: {spawn_point_count}")
    details.append(
        f"spawner spawned vehicle count for {vehicle_name or 'any vehicle'}: {spawned_vehicle_count}"
    )
    if spawned_vehicle_names:
        details.append(f"spawned vehicle names: {sorted(set(spawned_vehicle_names))}")

    if spawn_point_count == 0 or spawned_vehicle_count == 0:
        missing_parts = []
        if spawn_point_count == 0:
            missing_parts.append("spawn_point parameter")
        if spawned_vehicle_count == 0:
            missing_parts.append(f"spawned vehicle {vehicle_name or 'any vehicle'}")
        return MetricResult(
            metric_id,
            topic,
            "FAIL",
            "Missing CDAS-13 evidence: " + ", ".join(missing_parts),
            details,
        )

    return MetricResult(
        metric_id,
        topic,
        "PASS",
        f"Spawner received spawn_point parameter and spawned {vehicle_name or 'a vehicle'}.",
        details,
    )


def evaluate_cdas_14(
    bag_path: Path,
    storage_id: str,
    topic: str,
    max_messages: Optional[int],
    csv_dir: Optional[Path],
) -> MetricResult:
    metric_id = "CDAS-14"
    details: List[str] = []

    try:
        entries, topic_type = read_rosout_entries(
            bag_path=bag_path,
            storage_id=storage_id,
            topic=topic,
            max_messages=max_messages,
        )
    except Exception as exc:
        return MetricResult(metric_id, topic, "FAIL", f"Failed to read {topic}: {exc}", details)

    if csv_dir:
        csv_path = csv_dir / "cdas_14_rosout.csv"
        write_rosout_csv(csv_path, entries)
        details.append(f"csv written: {csv_path}")

    details.append(f"topic type: {topic_type}")
    details.append(f"rosout message count read: {len(entries)}")

    command_pattern = re.compile(
        r"PUBLISHING to CARLA:\s*throttle=(?P<throttle>[-+0-9.eE]+),\s*"
        r"brake=(?P<brake>[-+0-9.eE]+),\s*"
        r"steer=(?P<steer>[-+0-9.eE]+)",
        re.IGNORECASE,
    )

    command_count = 0
    ackermann_named_count = 0
    throttle_values: List[float] = []
    brake_values: List[float] = []
    steer_values: List[float] = []

    for entry in entries:
        match = command_pattern.search(entry.msg)
        if not match:
            continue

        command_count += 1

        if "ackermann_control_node" in entry.name or "ackermann_control_node" in entry.msg:
            ackermann_named_count += 1

        try:
            throttle_values.append(float(match.group("throttle")))
            brake_values.append(float(match.group("brake")))
            steer_values.append(float(match.group("steer")))
        except ValueError:
            pass

    details.append(f"ackermann PUBLISHING to CARLA command count: {command_count}")
    details.append(f"commands associated with ackermann_control_node count: {ackermann_named_count}")

    if throttle_values:
        details.append(f"throttle min/max: {min(throttle_values):.6f} / {max(throttle_values):.6f}")
    if brake_values:
        details.append(f"brake min/max: {min(brake_values):.6f} / {max(brake_values):.6f}")
    if steer_values:
        details.append(f"steer min/max: {min(steer_values):.6f} / {max(steer_values):.6f}")

    if command_count == 0:
        return MetricResult(
            metric_id,
            topic,
            "FAIL",
            "No ackermann control logs publishing throttle/brake/steer to CARLA were found.",
            details,
        )

    if ackermann_named_count == 0:
        return MetricResult(
            metric_id,
            topic,
            "PARTIAL",
            "PUBLISHING to CARLA command logs found, but rosout name did not confirm ackermann_control_node.",
            details,
        )

    return MetricResult(
        metric_id,
        topic,
        "PASS",
        "ackermann_control_node published throttle/brake/steer commands to CARLA.",
        details,
    )


def print_result(result: MetricResult) -> None:
    print("=" * 80)
    print(f"{result.metric_id} result: {result.status}")
    print(f"topic: {result.topic}")
    print(result.summary)
    print("-" * 80)
    for detail in result.details:
        print(detail)


def print_summary(results: List[MetricResult]) -> None:
    print("=" * 80)
    print("CDAS MCAP metric summary")
    print("-" * 80)
    for result in results:
        print(f"{result.metric_id}\t{result.status}\t{result.topic}\t{result.summary}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ROS 2 MCAP bag topics for CDAS-8, CDAS-9, CDAS-10, "
            "CDAS-11, CDAS-13, and CDAS-14. By default, runs all implemented metrics."
        )
    )
    parser.add_argument(
        "bag_path",
        type=Path,
        help="Path to ROS 2 bag directory or direct .mcap file.",
    )
    parser.add_argument(
        "--metric",
        choices=["CDAS-8", "CDAS-9", "CDAS-10", "CDAS-11", "CDAS-13", "CDAS-14", "all"],
        default="all",
        help="Metric to run. Default: all.",
    )
    parser.add_argument(
        "--threshold-sec",
        type=float,
        default=0.1,
        help="Maximum allowed topic interval in seconds. Default: 0.1",
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=1e-6,
        help="Tolerance added to threshold. Default: 1e-6 sec.",
    )
    parser.add_argument(
        "--storage-id",
        default="mcap",
        help="rosbag2 storage plugin id. Default: mcap",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="List topics and exit instead of running metrics.",
    )
    parser.add_argument(
        "--clock-topic",
        default=DEFAULT_TOPIC_BY_METRIC["CDAS-8"],
        help="Topic for CDAS-8. Default: /clock",
    )
    parser.add_argument(
        "--current-pose-topic",
        default=DEFAULT_TOPIC_BY_METRIC["CDAS-9"],
        help="Topic for CDAS-9. Default: /localization/current_pose",
    )
    parser.add_argument(
        "--robot-status-topic",
        default=DEFAULT_TOPIC_BY_METRIC["CDAS-10"],
        help="Topic for CDAS-10. Default: /hardware_interface/controller/robot_status",
    )
    parser.add_argument(
        "--vehicle-name",
        default=None,
        help=(
            "Optional vehicle name for CDAS-13. If omitted, any vehicle spawned "
            "by the CARLA spawner satisfies the check."
        ),
    )
    parser.add_argument(
        "--rosout-topic",
        default="/rosout",
        help="ROS log topic for CDAS-11/CDAS-13/CDAS-14. Default: /rosout",
    )
    parser.add_argument(
        "--expected-plugin",
        action="append",
        default=None,
        help=(
            "Expected plugin for CDAS-11. Can be repeated. "
            "If omitted, the default CARMA-CARLA plugin list is used."
        ),
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Optional maximum number of matching topic messages to read per metric.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Optional directory for per-topic CSV output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bag_path = args.bag_path.expanduser()

    if not bag_path.exists():
        print(f"Bag path does not exist: {bag_path}", file=sys.stderr)
        return 2

    if args.list_topics:
        return list_topics(bag_path, args.storage_id)

    metric_ids = (
        ["CDAS-8", "CDAS-9", "CDAS-10", "CDAS-11", "CDAS-13", "CDAS-14"]
        if args.metric == "all"
        else [args.metric]
    )

    expected_plugins = args.expected_plugin if args.expected_plugin else DEFAULT_CDAS_11_PLUGINS

    results: List[MetricResult] = []

    for metric_id in metric_ids:
        if metric_id == "CDAS-8":
            result = evaluate_clock_metric(
                bag_path=bag_path,
                storage_id=args.storage_id,
                topic=args.clock_topic,
                threshold_sec=args.threshold_sec,
                tolerance_sec=args.tolerance_sec,
                max_messages=args.max_messages,
                csv_dir=args.csv_dir,
            )
        elif metric_id == "CDAS-9":
            result = evaluate_topic_publish_rate_metric(
                metric_id="CDAS-9",
                bag_path=bag_path,
                storage_id=args.storage_id,
                topic=args.current_pose_topic,
                threshold_sec=args.threshold_sec,
                tolerance_sec=args.tolerance_sec,
                max_messages=args.max_messages,
                csv_dir=args.csv_dir,
            )
        elif metric_id == "CDAS-10":
            result = evaluate_topic_publish_rate_metric(
                metric_id="CDAS-10",
                bag_path=bag_path,
                storage_id=args.storage_id,
                topic=args.robot_status_topic,
                threshold_sec=args.threshold_sec,
                tolerance_sec=args.tolerance_sec,
                max_messages=args.max_messages,
                csv_dir=args.csv_dir,
            )
        elif metric_id == "CDAS-11":
            result = evaluate_cdas_11(
                bag_path=bag_path,
                storage_id=args.storage_id,
                topic=args.rosout_topic,
                expected_plugins=expected_plugins,
                max_messages=args.max_messages,
                csv_dir=args.csv_dir,
            )
        elif metric_id == "CDAS-13":
            result = evaluate_cdas_13(
                bag_path=bag_path,
                storage_id=args.storage_id,
                topic=args.rosout_topic,
                vehicle_name=args.vehicle_name,
                max_messages=args.max_messages,
                csv_dir=args.csv_dir,
            )
        elif metric_id == "CDAS-14":
            result = evaluate_cdas_14(
                bag_path=bag_path,
                storage_id=args.storage_id,
                topic=args.rosout_topic,
                max_messages=args.max_messages,
                csv_dir=args.csv_dir,
            )
        else:
            raise RuntimeError(f"Unsupported metric: {metric_id}")

        print_result(result)
        results.append(result)

    if len(results) > 1:
        print_summary(results)

    if any(result.status == "FAIL" for result in results):
        return 2
    if any(result.status == "PARTIAL" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
