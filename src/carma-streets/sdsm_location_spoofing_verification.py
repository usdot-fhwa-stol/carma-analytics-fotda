import argparse
import csv
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from parse_kafka_logs import KafkaLogMessageType, parse_kafka_logs_as_type

# What this check is called in the report. A caller running it as part of a
# wider suite passes its own test code; run on its own the script uses this.
DEFAULT_METRIC_LABEL = "SDSM Location Spoofing Verification"

DEFAULT_MAX_MEAN_POSITION_ERROR_M = 0.2
DEFAULT_MAX_MEAN_HEADING_ERROR_DEG = 1.0
DEFAULT_MIN_HEADING_SPEED_MPS = 0.1
DEFAULT_MATCH_TOLERANCE_MS = 5
# projString lat_0/lon_0 are written with 10 decimals, so an exact config match agrees far below 1e-7 deg (~1 cm)
REFERENCE_MATCH_TOLERANCE_DEG = 1e-7
INCOMING_SDSM_TOPIC = "/message/incoming_sdsm"
PLOT_NAME = "sdsm_location_spoofing_verification.png"
CSV_NAME = "sdsm_location_spoofing_verification.csv"

# J2735 / J3224 JSON units used on the SDSM Kafka topic
SDSM_LAT_LON_UNITS_DEG = 1e-7
SDSM_OFFSET_UNITS_M = 0.1
SDSM_HEADING_UNITS_DEG = 0.0125
SDSM_HEADING_UNAVAILABLE = 28800

SOURCE_COLORS = {"kafka": "#2a78d6", "mcap": "#eb6834"}
EXPECTED_COLOR = "#52514e"

# WGS84 ellipsoid
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def geodetic_to_ecef(lat_deg, lon_deg) -> np.ndarray:
    """Convert WGS84 latitude/longitude (on the ellipsoid surface) to ECEF meters."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([n * np.cos(lat) * np.cos(lon), n * np.cos(lat) * np.sin(lon), n * (1 - WGS84_E2) * np.sin(lat)])


def enu_basis(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """Rows are the east, north and up unit vectors, in ECEF, at the given origin."""
    lat0, lon0 = np.radians(lat0_deg), np.radians(lon0_deg)
    return np.array([
        [-np.sin(lon0), np.cos(lon0), 0.0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
    ])


def geodetic_to_enu(lat_deg, lon_deg, lat0_deg: float, lon0_deg: float) -> tuple:
    """Local east/north offsets in meters of lat/lon from the origin lat0/lon0."""
    delta = geodetic_to_ecef(lat_deg, lon_deg) - geodetic_to_ecef(lat0_deg, lon0_deg)[:, None]
    east, north, _ = enu_basis(lat0_deg, lon0_deg) @ delta.reshape(3, -1)
    return east, north


def enu_to_geodetic(east, north, lat0_deg: float, lon0_deg: float) -> tuple:
    """Latitude/longitude of local east/north offsets in meters from the origin lat0/lon0."""
    east, north = np.atleast_1d(east).astype(float), np.atleast_1d(north).astype(float)
    x, y, z = geodetic_to_ecef(lat0_deg, lon0_deg)[:, None] + enu_basis(lat0_deg, lon0_deg)[:2].T @ np.vstack((east, north))
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(5):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + WGS84_E2 * n * np.sin(lat), p)
    return np.degrees(lat), np.degrees(lon)


def compass_heading_deg(east, north) -> np.ndarray:
    """Heading of an east/north vector in degrees clockwise from north, in [0, 360)."""
    return np.degrees(np.arctan2(east, north)) % 360


def wrap_deg(angle) -> np.ndarray:
    """Wrap an angle in degrees to [-180, 180)."""
    return (np.asarray(angle) + 180) % 360 - 180


def ddatetime_to_epoch_ms(year, month, day, hour, minute, millisecond, offset_minute) -> int:
    """Convert a J2735 DDateTime to epoch milliseconds. offset_minute is the local offset from UTC."""
    local = datetime(year, month, day, hour, minute, tzinfo=timezone.utc) + timedelta(milliseconds=millisecond)
    return round((local - timedelta(minutes=offset_minute)).timestamp() * 1000)


def parse_projection_origin(proj_string: str) -> tuple:
    """Read the lat_0/lon_0 origin of a detected object's projection string."""
    match = re.search(r"\+lat_0=([-\d.]+).*?\+lon_0=([-\d.]+)", proj_string)
    if match is None:
        raise ValueError(f"No lat_0/lon_0 in projection string {proj_string}")
    return float(match.group(1)), float(match.group(2))


def load_detections(detection_log_path: Path) -> list:
    """Read a detected object Kafka log into a list of detections.

    Each detection's position and velocity are cartesian (east, north) in the frame of its projection
    string, whose lat_0/lon_0 origin is the reference location FLIRCameraDriver reports detections at.

    Args:
        detection_log_path (Path): Path to the detected object (v2xhub_sim_sensor_detected_object) Kafka log

    Returns:
        list: One dict per detection
    """
    if not detection_log_path.is_file():
        raise FileNotFoundError(f"Detection log {detection_log_path} does not exist")
    detections = []
    for msg in parse_kafka_logs_as_type(detection_log_path, KafkaLogMessageType.DetectedObject):
        json_message = msg.json_message
        origin_lat, origin_lon = parse_projection_origin(json_message["projString"])
        detections.append({
            "object_id": json_message["objectId"],
            "time_ms": json_message["timestamp"],
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "east": json_message["position"]["x"],
            "north": json_message["position"]["y"],
            "velocity_east": json_message["velocity"]["x"],
            "velocity_north": json_message["velocity"]["y"],
            "true_lat": json_message["wgs84Position"]["latitude"],
            "true_lon": json_message["wgs84Position"]["longitude"],
        })
    return detections


def load_kafka_sdsm_objects(sdsm_log_path: Path) -> list:
    """Read the detected objects of every SDSM in an SDSM Kafka log, converted to SI units.

    Args:
        sdsm_log_path (Path): Path to the SDSM Kafka topic log

    Returns:
        list: One dict per SDSM detected object
    """
    sdsm_objects = []
    for msg in parse_kafka_logs_as_type(sdsm_log_path, KafkaLogMessageType.SDSM):
        json_message = msg.json_message
        time_stamp = json_message["sdsm_time_stamp"]
        sdsm_time_ms = ddatetime_to_epoch_ms(
            time_stamp["year"], time_stamp["month"], time_stamp["day"], time_stamp["hour"],
            time_stamp["minute"], time_stamp["second"], time_stamp.get("offset", 0),
        )
        for detected_object in json_message["objects"]:
            common_data = detected_object["detected_object_data"]["detected_object_common_data"]
            heading = common_data["heading"]
            sdsm_objects.append({
                "source": "kafka",
                "sdsm_time_ms": sdsm_time_ms,
                "measurement_time_ms": common_data["measurement_time"],
                "object_id": common_data["object_id"],
                "ref_lat": json_message["ref_pos"]["lat"] * SDSM_LAT_LON_UNITS_DEG,
                "ref_lon": json_message["ref_pos"]["long"] * SDSM_LAT_LON_UNITS_DEG,
                "offset_x": common_data["pos"]["offset_x"] * SDSM_OFFSET_UNITS_M,
                "offset_y": common_data["pos"]["offset_y"] * SDSM_OFFSET_UNITS_M,
                "heading_deg": None if heading == SDSM_HEADING_UNAVAILABLE else heading * SDSM_HEADING_UNITS_DEG,
            })
    return sdsm_objects


def load_mcap_sdsm_objects(mcap_path: Path) -> list:
    """Read the detected objects of every SDSM CARMA Platform received (/message/incoming_sdsm) in an mcap.

    Requires ROS 2 and carma_v2x_msgs to be sourced.

    Args:
        mcap_path (Path): Path to a CARMA Platform mcap

    Returns:
        list: One dict per SDSM detected object, in the same units as load_kafka_sdsm_objects
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(mcap_path), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if INCOMING_SDSM_TOPIC not in topic_types:
        raise ValueError(f"{mcap_path} has no {INCOMING_SDSM_TOPIC} topic")
    sdsm_type = get_message(topic_types[INCOMING_SDSM_TOPIC])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[INCOMING_SDSM_TOPIC]))

    sdsm_objects = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        sdsm = deserialize_message(data, sdsm_type)
        time_stamp = sdsm.sdsm_time_stamp
        sdsm_time_ms = ddatetime_to_epoch_ms(
            time_stamp.year.year, time_stamp.month.month, time_stamp.day.day, time_stamp.hour.hour,
            time_stamp.minute.minute, time_stamp.second.millisecond, time_stamp.offset.offset_minute,
        )
        for detected_object in sdsm.objects.detected_object_data:
            common_data = detected_object.detected_object_common_data
            sdsm_objects.append({
                "source": "mcap",
                "sdsm_time_ms": sdsm_time_ms,
                "measurement_time_ms": common_data.measurement_time.measurement_time_offset * 1000,
                "object_id": common_data.detected_id.object_id,
                "ref_lat": sdsm.ref_pos.latitude,
                "ref_lon": sdsm.ref_pos.longitude,
                "offset_x": common_data.pos.offset_x.object_distance,
                "offset_y": common_data.pos.offset_y.object_distance,
                "heading_deg": None if common_data.heading.unavailable else common_data.heading.heading,
            })
    print(f"Extracted {len(sdsm_objects)} SDSM objects from {mcap_path}.")
    return sdsm_objects


def match_sdsm_objects(sdsm_objects: list, detections: list, tolerance_ms: float) -> tuple:
    """Pair each SDSM object with the detection it was generated from.

    The measurement time offset is how long before the SDSM timestamp the object was detected, so the
    source detection has the same object ID and a timestamp of sdsm_time - measurement_time.

    Returns:
        tuple: (list of (sdsm_object, detection) pairs, count of SDSM objects without a source detection)
    """
    detections_by_id = {}
    for detection in detections:
        detections_by_id.setdefault(detection["object_id"], []).append(detection)
    times_by_id = {}
    for object_id, object_detections in detections_by_id.items():
        object_detections.sort(key=lambda detection: detection["time_ms"])
        times_by_id[object_id] = np.array([detection["time_ms"] for detection in object_detections])

    pairs = []
    unmatched = 0
    for sdsm_object in sdsm_objects:
        times = times_by_id.get(sdsm_object["object_id"])
        if times is None:
            unmatched += 1
            continue
        detection_time_ms = sdsm_object["sdsm_time_ms"] - sdsm_object["measurement_time_ms"]
        idx = np.searchsorted(times, detection_time_ms)
        nearest = min((i for i in (idx - 1, idx) if 0 <= i < len(times)), key=lambda i: abs(times[i] - detection_time_ms))
        if abs(times[nearest] - detection_time_ms) <= tolerance_ms:
            pairs.append((sdsm_object, detections_by_id[sdsm_object["object_id"]][nearest]))
        else:
            unmatched += 1
    return pairs, unmatched


def fit_driver_rotation_deg(detections: list, ref_lat: float, ref_lon: float) -> tuple:
    """Fit the rigid rotation FLIRCameraDriver applied between each detection's true location
    (wgs84Position, as seen from the camera's true location) and its reported position at the reference.

    Returns:
        tuple: (clockwise rotation in degrees, mean fit residual in meters), or (None, None) if the
        pedestrian did not move enough to fit a rotation
    """
    reported = np.array([[detection["east"], detection["north"]] for detection in detections])
    true_east, true_north = geodetic_to_enu(
        np.array([detection["true_lat"] for detection in detections]),
        np.array([detection["true_lon"] for detection in detections]),
        ref_lat, ref_lon,
    )
    true = np.column_stack((true_east, true_north))
    reported_centered, true_centered = reported - reported.mean(0), true - true.mean(0)
    if np.linalg.norm(true_centered, axis=1).max() < 1.0:
        return None, None
    u, _, vt = np.linalg.svd(true_centered.T @ reported_centered)
    rotation = (u @ vt).T
    residual = np.linalg.norm(reported_centered - true_centered @ rotation.T, axis=1).mean()
    counterclockwise_deg = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return float(wrap_deg(-counterclockwise_deg)), float(residual)


def verify_location_spoofing(
    detection_log_path: Path,
    ref_lat: float,
    ref_lon: float,
    sdsm_log_path: Path = None,
    mcap_path: Path = None,
    max_mean_position_error_m: float = DEFAULT_MAX_MEAN_POSITION_ERROR_M,
    max_mean_heading_error_deg: float = DEFAULT_MAX_MEAN_HEADING_ERROR_DEG,
    min_heading_speed_mps: float = DEFAULT_MIN_HEADING_SPEED_MPS,
    match_tolerance_ms: float = DEFAULT_MATCH_TOLERANCE_MS,
    plots_dir: Path = None,
    metric_label: str = DEFAULT_METRIC_LABEL,
) -> dict:
    """Verify SDSMs place a location-spoofed pedestrian where FLIRCameraDriver reported it.

    FLIRCameraDriver discards the camera's true location and reports each detection's cartesian offsets
    (already rotated to the configured reference heading) from the configured remote reference
    lat/lon, which it writes as the lat_0/lon_0 of the detection's projection string. Each SDSM object's
    location, ref_pos plus its (offset_x north, offset_y east) offsets, should equal the reference lat/lon
    plus the source detection's offsets, and its heading should equal the detection's velocity heading.

    Args:
        detection_log_path (Path): Path to the detected object (v2xhub_sim_sensor_detected_object) Kafka log
        ref_lat (float): Remote reference latitude configured in FLIRCameraDriver
        ref_lon (float): Remote reference longitude configured in FLIRCameraDriver
        sdsm_log_path (Path): Path to the SDSM Kafka log. At least one of sdsm_log_path and mcap_path is required.
        mcap_path (Path): Path to a CARMA Platform mcap recording /message/incoming_sdsm
        max_mean_position_error_m (float): Pass threshold on mean SDSM position error. Default is 0.2 m.
        max_mean_heading_error_deg (float): Pass threshold on mean absolute SDSM heading error. Default is 1 degree.
        min_heading_speed_mps (float): Detections slower than this have no meaningful heading and are left
            out of the heading error. Default is 0.1 m/s.
        match_tolerance_ms (float): Allowed difference between sdsm_time - measurement_time and the source
            detection's timestamp. Default is 5 ms.
        plots_dir (Path): Directory to save the plot and per-object csv to. If not given, the plot is shown.

    Returns:
        dict: Pass/fail and error statistics per SDSM source, and the per-object rows
    """
    if sdsm_log_path is None and mcap_path is None:
        raise ValueError("Need an SDSM Kafka log and/or an mcap to read SDSMs from")

    def at_reference(detection):
        return (abs(detection["origin_lat"] - ref_lat) < REFERENCE_MATCH_TOLERANCE_DEG
                and abs(detection["origin_lon"] - ref_lon) < REFERENCE_MATCH_TOLERANCE_DEG)

    all_detections = load_detections(Path(detection_log_path))
    detections = [detection for detection in all_detections if at_reference(detection)]
    if not detections:
        origins = sorted({(detection["origin_lat"], detection["origin_lon"]) for detection in all_detections})
        raise ValueError(
            f"No detections are reported at reference {ref_lat}, {ref_lon}. Detection projection origins "
            f"found in {detection_log_path}: {origins}"
        )
    print(
        f"{len(detections)}/{len(all_detections)} detections are reported at reference {ref_lat}, {ref_lon}. "
        "Other detections (runs with a different reference) are ignored."
    )

    sdsm_objects = []
    if sdsm_log_path is not None:
        sdsm_objects += load_kafka_sdsm_objects(Path(sdsm_log_path))
    if mcap_path is not None:
        sdsm_objects += load_mcap_sdsm_objects(Path(mcap_path))

    rows = []
    results = {}
    for source in dict.fromkeys(sdsm_object["source"] for sdsm_object in sdsm_objects):
        source_objects = [sdsm_object for sdsm_object in sdsm_objects if sdsm_object["source"] == source]
        all_pairs, unmatched = match_sdsm_objects(source_objects, all_detections, match_tolerance_ms)
        # SDSM objects generated from another run's detections (a different reference) are not verified
        pairs = [pair for pair in all_pairs if at_reference(pair[1])]
        other_run = len(all_pairs) - len(pairs)
        if not pairs:
            print(f"WARNING: No {source} SDSM objects match a detection reported at the reference.")
            continue

        for sdsm_object, detection in pairs:
            expected_lat, expected_lon = enu_to_geodetic(detection["east"], detection["north"], ref_lat, ref_lon)
            # SDSM offsets are NED: offset_x is north and offset_y is east
            sdsm_lat, sdsm_lon = enu_to_geodetic(
                sdsm_object["offset_y"], sdsm_object["offset_x"], sdsm_object["ref_lat"], sdsm_object["ref_lon"]
            )
            error_east, error_north = geodetic_to_enu(sdsm_lat, sdsm_lon, expected_lat[0], expected_lon[0])
            speed = np.hypot(detection["velocity_east"], detection["velocity_north"])
            expected_heading = compass_heading_deg(detection["velocity_east"], detection["velocity_north"])
            heading_error = (
                float(wrap_deg(sdsm_object["heading_deg"] - expected_heading))
                if sdsm_object["heading_deg"] is not None and speed >= min_heading_speed_mps else None
            )
            rows.append({
                "Source": source,
                "SDSM Time (ms)": sdsm_object["sdsm_time_ms"],
                "Detection Time (ms)": detection["time_ms"],
                "Object ID": detection["object_id"],
                "Expected Latitude": round(float(expected_lat[0]), 9),
                "Expected Longitude": round(float(expected_lon[0]), 9),
                "SDSM Latitude": round(float(sdsm_lat[0]), 9),
                "SDSM Longitude": round(float(sdsm_lon[0]), 9),
                "Expected East (m)": round(detection["east"], 3),
                "Expected North (m)": round(detection["north"], 3),
                "Error East (m)": round(float(error_east[0]), 4),
                "Error North (m)": round(float(error_north[0]), 4),
                "Position Error (m)": round(float(np.hypot(error_east[0], error_north[0])), 4),
                "Speed (m/s)": round(float(speed), 3),
                "Expected Heading (deg)": round(float(expected_heading), 3),
                "SDSM Heading (deg)": sdsm_object["heading_deg"],
                "Heading Error (deg)": None if heading_error is None else round(heading_error, 4),
            })

        source_rows = [row for row in rows if row["Source"] == source]
        position_errors = np.array([row["Position Error (m)"] for row in source_rows])
        heading_errors = np.abs([row["Heading Error (deg)"] for row in source_rows if row["Heading Error (deg)"] is not None])
        mean_position_error = float(position_errors.mean())
        mean_heading_error = float(heading_errors.mean()) if len(heading_errors) else None
        position_pass = mean_position_error < max_mean_position_error_m
        heading_pass = mean_heading_error is not None and mean_heading_error < max_mean_heading_error_deg
        results[source] = {
            "verified_objects": len(source_rows),
            "unmatched_objects": unmatched,
            "other_reference_objects": other_run,
            "mean_position_error_m": mean_position_error,
            "p95_position_error_m": float(np.percentile(position_errors, 95)),
            "max_position_error_m": float(position_errors.max()),
            "mean_bias_east_m": float(np.mean([row["Error East (m)"] for row in source_rows])),
            "mean_bias_north_m": float(np.mean([row["Error North (m)"] for row in source_rows])),
            "heading_objects": len(heading_errors),
            "mean_heading_error_deg": mean_heading_error,
            "max_heading_error_deg": float(heading_errors.max()) if len(heading_errors) else None,
            "position_pass": position_pass,
            "heading_pass": heading_pass,
            "pass": position_pass and heading_pass,
        }

    if not results:
        raise ValueError("No SDSM objects match a detection reported at the reference; nothing to verify")

    rotation_deg, rotation_residual_m = fit_driver_rotation_deg(detections, ref_lat, ref_lon)

    print(f"\n=== {metric_label} ===")
    print(f"Reference: {ref_lat}, {ref_lon}")
    if rotation_deg is not None:
        print(
            f"FLIRCameraDriver rotated detections {rotation_deg:+.2f} deg (clockwise) from the camera's true frame "
            f"to the reference frame (rigid fit residual {rotation_residual_m:.3f} m)"
        )
    for source, result in results.items():
        mean_heading = result["mean_heading_error_deg"]
        print(f"\n[{source} SDSM] {result['verified_objects']} objects verified, "
              f"{result['unmatched_objects']} without a source detection, "
              f"{result['other_reference_objects']} from runs at other references (ignored)")
        print(f"  Position error: mean {result['mean_position_error_m']:.3f} m, "
              f"p95 {result['p95_position_error_m']:.3f} m, max {result['max_position_error_m']:.3f} m "
              f"(mean bias east {result['mean_bias_east_m']:+.3f} m, north {result['mean_bias_north_m']:+.3f} m) "
              f"-> {'PASS' if result['position_pass'] else 'FAIL'} (< {max_mean_position_error_m:g} m)")
        print(f"  Heading error:  mean {'n/a' if mean_heading is None else f'{mean_heading:.3f}'} deg over "
              f"{result['heading_objects']} objects moving >= {min_heading_speed_mps:g} m/s "
              f"-> {'PASS' if result['heading_pass'] else 'FAIL'} (< {max_mean_heading_error_deg:g} deg)")
    overall_pass = all(result["pass"] for result in results.values())
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")

    plot_verification(rows, results, max_mean_position_error_m,
                      max_mean_heading_error_deg, plots_dir, metric_label)

    return {"pass": overall_pass, "rotation_deg": rotation_deg, "sources": results, "rows": rows}


HISTOGRAM_BINS = 20
# The heading error is quantised to the 0.0125 deg SDSM encoding step, so its
# histogram uses a fixed bin width rather than a fixed bin count.
HEADING_BIN_WIDTH_DEG = 0.001


def _frame_axes(axis, x_values, y_values, margin: float = 1.1):
    """Set each axis limit to ``margin`` times that axis's own data range.

    The two axes are scaled independently, so the panel frames the walked area
    tightly. That means a metre east and a metre north are not the same length on
    the page; the panel shows where the pedestrian went, and the position error
    it is checked against is quantified in its own histogram.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if x_values.size == 0 or y_values.size == 0:
        return
    for setter, values in ((axis.set_xlim, x_values), (axis.set_ylim, y_values)):
        middle = (values.max() + values.min()) / 2.0
        half = ((values.max() - values.min()) / 2.0 or 1.0) * margin
        setter(middle - half, middle + half)


def _fixed_width_bins(values, width: float):
    """Bin edges of exactly ``width``, covering the data and aligned to multiples of it.

    Used for the heading error, whose values are quantised to the 0.0125 deg
    encoding step of the SDSM heading field. A fixed bin count rescales with the
    data and hides that structure; a fixed width keeps every plot on the same
    scale, so a change in the spread is visible between runs.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([0.0, width])
    low = np.floor(values.min() / width) * width
    high = np.ceil(values.max() / width) * width
    if high <= low:
        high = low + width
    # +1.5 so the final edge is always included despite floating-point drift.
    return np.arange(low, high + width * 1.5, width)


def _threshold_marker(axis, values, threshold, label, signed=False):
    """Draw a threshold line, but only when it does not swamp the data.

    The heading error is within +/-0.013 deg against a +/-1 deg limit. Drawing
    that line forces an axis 75x wider than the data and flattens the histogram
    into an invisible sliver, so when the threshold is far outside the data the
    limit is stated in the legend instead of plotted.
    """
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min() if values.size else 0.0
    reach = max(abs(values.max()), abs(values.min())) if values.size else 0.0
    if threshold <= max(reach, span) * 1.5:
        for sign in ((1, -1) if signed else (1,)):
            axis.axvline(sign * threshold, color="black", linestyle="--", linewidth=1,
                         label=label if sign == 1 else None)
    else:
        # Off-scale: keep the criterion visible without destroying the scale.
        axis.plot([], [], linestyle="--", color="black", linewidth=1,
                  label=f"{label} — off scale")


def plot_verification(rows: list, results: dict, max_mean_position_error_m: float,
                      max_mean_heading_error_deg: float, plots_dir: Path = None,
                      metric_label: str = DEFAULT_METRIC_LABEL):
    """Plot SDSM vs expected object locations in the reference frame, and the error distributions.

    The errors are shown as histograms rather than against time. These runs are
    minutes apart, so a time axis spends most of its width on the gaps between
    runs and compresses every run into a narrow band; the distribution is what
    the pass criteria are actually stated against.
    """
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.4])
    map_ax = fig.add_subplot(grid[:, 0])
    position_ax = fig.add_subplot(grid[0, 1])
    heading_ax = fig.add_subplot(grid[1, 1])

    expected = np.array([[row["Expected East (m)"], row["Expected North (m)"]] for row in rows])
    map_ax.scatter(expected[:, 0], expected[:, 1], s=28, marker="x", color=EXPECTED_COLOR,
                   linewidths=0.8, label="Expected (reference + detection offsets)")
    position_errors, heading_errors, labels = [], [], []
    for source in results:
        source_rows = [row for row in rows if row["Source"] == source]
        sdsm_east = [row["Expected East (m)"] + row["Error East (m)"] for row in source_rows]
        sdsm_north = [row["Expected North (m)"] + row["Error North (m)"] for row in source_rows]
        color = SOURCE_COLORS[source]
        map_ax.scatter(sdsm_east, sdsm_north, s=8, color=color, label=f"{source} SDSM")

        position_errors.append(([row["Position Error (m)"] for row in source_rows], color,
                                f"{source} SDSM (mean {results[source]['mean_position_error_m']:.3f} m)"))
        source_headings = [row["Heading Error (deg)"] for row in source_rows
                           if row["Heading Error (deg)"] is not None]
        if source_headings:
            heading_errors.append((source_headings, color,
                                   f"{source} SDSM (mean |error| "
                                   f"{results[source]['mean_heading_error_deg']:.3f} deg)"))
        labels.append(source)

    map_ax.set_title("Pedestrian Location in Reference Frame")
    map_ax.set_xlabel("East of Reference (m)")
    map_ax.set_ylabel("North of Reference (m)")
    # Frame the data, not the origin. The reference sits at (0, 0) by
    # construction and the pedestrian never walks over it, so including it
    # leaves most of the panel empty.
    all_east = np.r_[expected[:, 0], [row["Expected East (m)"] + row["Error East (m)"] for row in rows]]
    all_north = np.r_[expected[:, 1], [row["Expected North (m)"] + row["Error North (m)"] for row in rows]]
    _frame_axes(map_ax, all_east, all_north, margin=1.1)
    map_ax.grid(True, alpha=0.3)
    map_ax.legend(fontsize=8)

    for values, color, label in position_errors:
        position_ax.hist(values, bins=HISTOGRAM_BINS, color=color, alpha=0.8,
                         edgecolor="white", linewidth=0.5, label=label)
    _threshold_marker(position_ax, [v for values, _c, _l in position_errors for v in values],
                      max_mean_position_error_m,
                      f"Mean threshold ({max_mean_position_error_m:g} m)")
    position_ax.set_title("SDSM Position Error")
    position_ax.set_xlabel("Position Error (m)")
    position_ax.set_ylabel("SDSM objects")
    position_ax.grid(True, axis="y", alpha=0.3)
    position_ax.legend(fontsize=8)

    heading_bins = _fixed_width_bins(
        [v for values, _c, _l in heading_errors for v in values] or [0.0],
        HEADING_BIN_WIDTH_DEG,
    )
    for values, color, label in heading_errors:
        heading_ax.hist(values, bins=heading_bins, color=color, alpha=0.8,
                        edgecolor="white", linewidth=0.5, label=label)
    _threshold_marker(heading_ax, [v for values, _c, _l in heading_errors for v in values] or [0.0],
                      max_mean_heading_error_deg,
                      f"Mean threshold (±{max_mean_heading_error_deg:g} deg)", signed=True)
    heading_ax.set_title("SDSM Heading Error (moving detections)")
    heading_ax.set_xlabel("Heading Error (deg), {:g} deg bins".format(HEADING_BIN_WIDTH_DEG))
    heading_ax.set_ylabel("SDSM objects")
    heading_ax.grid(True, axis="y", alpha=0.3)
    heading_ax.legend(fontsize=8)

    for axis in (map_ax, position_ax, heading_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)

    overall = "PASS" if all(result["pass"] for result in results.values()) else "FAIL"
    fig.suptitle(f"{metric_label}: {overall}")
    fig.tight_layout()

    if plots_dir:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / PLOT_NAME, dpi=300)
        with open(plots_dir / CSV_NAME, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        print(f"Plot and per-object csv saved to: {plots_dir}")
        plt.close(fig)
    else:
        plt.show()


def find_sdsm_log(kafka_log_dir: Path) -> Path:
    """Find the non-empty SDSM Kafka log in a directory, or None if there is none."""
    sdsm_logs = [
        log for log in kafka_log_dir.glob(f"*{KafkaLogMessageType.SDSM.value}*.log") if log.stat().st_size > 0
    ]
    if len(sdsm_logs) > 1:
        raise ValueError(f"Found several SDSM logs in {kafka_log_dir}: {sdsm_logs}. Select one with --sdsm-log.")
    return sdsm_logs[0] if sdsm_logs else None


def main():
    parser = argparse.ArgumentParser(
        description="Verify SDSMs place a location-spoofed pedestrian at the remote reference location "
        "configured in FLIRCameraDriver, by comparing each SDSM object's location and heading against its source "
        "detection on the detected object Kafka topic."
    )
    parser.add_argument(
        "--kafka-log-dir", help="Directory containing Kafka Log files.", type=Path, required=True
    )
    parser.add_argument(
        "--ref-lat", help="Remote reference latitude configured in FLIRCameraDriver.", type=float, required=True
    )
    parser.add_argument(
        "--ref-lon", help="Remote reference longitude configured in FLIRCameraDriver.", type=float, required=True
    )
    parser.add_argument(
        "--sdsm-log",
        help="SDSM Kafka log to verify. Default is the non-empty *sdsm*.log in --kafka-log-dir.",
        type=Path,
    )
    parser.add_argument(
        "--mcap",
        help=f"CARMA Platform mcap whose {INCOMING_SDSM_TOPIC} SDSMs are also verified. Requires ROS 2 and "
        "carma_v2x_msgs to be sourced.",
        type=Path,
    )
    parser.add_argument(
        "--plots-dir", help="Directory to save generated plot and per-object csv.", type=Path, required=True
    )
    parser.add_argument(
        "--max-mean-position-error", help="Pass threshold on mean SDSM position error, in meters.",
        type=float, default=DEFAULT_MAX_MEAN_POSITION_ERROR_M,
    )
    parser.add_argument(
        "--max-mean-heading-error", help="Pass threshold on mean absolute SDSM heading error, in degrees.",
        type=float, default=DEFAULT_MAX_MEAN_HEADING_ERROR_DEG,
    )
    parser.add_argument(
        "--min-heading-speed",
        help="Detections slower than this (m/s) have no meaningful heading and are left out of the heading error.",
        type=float, default=DEFAULT_MIN_HEADING_SPEED_MPS,
    )
    parser.add_argument(
        "--match-tolerance-ms",
        help="Allowed difference between an SDSM object's measurement time and its source detection's timestamp.",
        type=float, default=DEFAULT_MATCH_TOLERANCE_MS,
    )
    args = parser.parse_args()

    detection_logs = list(args.kafka_log_dir.glob(f"*{KafkaLogMessageType.DetectedObject.value}*.log"))
    if len(detection_logs) != 1:
        print(f"ERROR: Expected one detected object log in {args.kafka_log_dir}, found {detection_logs}")
        sys.exit(2)
    sdsm_log = args.sdsm_log or find_sdsm_log(args.kafka_log_dir)
    if sdsm_log is None and args.mcap is None:
        print(f"ERROR: No non-empty SDSM log in {args.kafka_log_dir}; pass --sdsm-log or --mcap")
        sys.exit(2)

    try:
        result = verify_location_spoofing(
            detection_logs[0],
            args.ref_lat,
            args.ref_lon,
            sdsm_log_path=sdsm_log,
            mcap_path=args.mcap,
            max_mean_position_error_m=args.max_mean_position_error,
            max_mean_heading_error_deg=args.max_mean_heading_error,
            min_heading_speed_mps=args.min_heading_speed,
            match_tolerance_ms=args.match_tolerance_ms,
            plots_dir=args.plots_dir,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(2)
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
