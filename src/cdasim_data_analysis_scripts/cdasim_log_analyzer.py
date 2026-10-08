#!/usr/bin/env python3

import argparse
import re
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_BASE = "/opt/carma-simulation/logs"
ANALYSIS_OUTPUT_BASE = Path("cdasim_analysis_log")
ANALYSIS_LOG = Path(f"Cdasim_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

CDAS_PASS_RATE = 0.95
CDAS_EVENT_PREFIX = "CDAS_EVENT"


def is_cdas_event(line: str, event_name: str) -> bool:
    """Return True when line contains the named structured CDAS event."""
    return re.search(
        rf"\b{CDAS_EVENT_PREFIX}\s+event={re.escape(event_name)}(?:\s|$)",
        line,
    ) is not None


def cdas_event_field(line: str, field_name: str):
    """Read a whitespace-delimited scalar field from a structured CDAS event."""
    match = re.search(rf"\b{re.escape(field_name)}=(?P<value>\S+)", line)
    return match.group("value") if match else None


def count_cdas_events(log_lines, event_name: str) -> int:
    return sum(1 for line in log_lines if is_cdas_event(line, event_name))


def write_analysis(message: str) -> None:
    # The report should only contain CDAS-related lines; everything else is suppressed.
    if "cdas" not in message.lower():
        return
    with open(ANALYSIS_LOG, "a", encoding="utf-8") as out:
        out.write(message + "\n")


def sanitize_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def build_analysis_log_path(log_dir: Path) -> Path:
    today = datetime.now().strftime("%Y%m%d")
    analyzed_folder = sanitize_path_part(log_dir.name)
    parent_folder = sanitize_path_part(log_dir.parent.name)
    output_folder = ANALYSIS_OUTPUT_BASE / f"{today}_{parent_folder}-{analyzed_folder}"

    output_folder.mkdir(parents=True, exist_ok=True)

    return output_folder / f"Cdasim_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def read_log_lines(log_file_path: str):
    path = Path(log_file_path)
    if not path.exists():
        write_analysis(f"log missing: {log_file_path}")
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def parse_log_timestamp(line: str):
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
    if not match:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def count_lines(log_lines, phrase: str) -> int:
    return sum(1 for line in log_lines if phrase in line)


def count_matching_lines(log_lines, pattern: str) -> int:
    regex = re.compile(pattern, re.IGNORECASE)
    return sum(1 for line in log_lines if regex.search(line))


def Check_Vehicle_Spawn(carla_log_lines, vehicle_name: str) -> bool:
    structured_spawn_pattern = re.compile(
        rf"\bCDAS_EVENT\s+event=(?:carla_sumo_vehicle_spawned|carla_external_vehicle_added)\b.*?"
        rf"(?:vehicle_id|actor_id)={re.escape(vehicle_name)}(?:\s|$)"
    )
    sumo_spawn_pattern = re.compile(
        rf"Successfully spawned CARLA actor for SUMO vehicle\s+'{re.escape(vehicle_name)}'"
    )
    external_vehicle_pattern = re.compile(
        rf"EXTERNAL VEHICLE ADDED:\s+Actor ID={re.escape(vehicle_name)}(?:,|\b)"
    )

    for line in carla_log_lines:
        if (
            structured_spawn_pattern.search(line)
            or sumo_spawn_pattern.search(line)
            or external_vehicle_pattern.search(line)
        ):
            write_analysis(f"vehicle {vehicle_name} spawn successful")
            return True

    write_analysis(f"vehicle {vehicle_name} spawn unsuccessful")
    return False


def CheckXmlRpcServer(carla_log_lines) -> bool:
    target_phrase = "Multi-XML-RPC manager actor connection status: true"

    for line in carla_log_lines:
        if (
            target_phrase in line
            or (
                is_cdas_event(line, "carla_actor_connection_status")
                and cdas_event_field(line, "connected") == "true"
            )
        ):
            write_analysis("xml rpc server connect successful")
            return True

    write_analysis("xml rpc server connect unsuccessful")
    return False


def Count_Added_Vehicles_In_Carla_Log(carla_log_lines) -> int:
    legacy_pattern = re.compile(
        r"Received VehicleUpdates interaction.*added=(\d+), updated=(\d+), removed=(\d+)"
    )
    total_added = 0

    for line in carla_log_lines:
        match = legacy_pattern.search(line)
        if match:
            total_added += int(match.group(1))
        elif is_cdas_event(line, "carla_vehicle_updates_received"):
            added = cdas_event_field(line, "added")
            if added is not None:
                total_added += int(added)

    write_analysis(f"total added vehicles reported in carla log: {total_added}")
    return total_added


# =========================
# CDAS PERFORMANCE METRIC CHECKS
# =========================

def parse_float_list(value: str):
    if not value.strip():
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def Parse_Carla_Spawn_Actor_Calls(carla_log_lines):
    """
    Parse CARLA XML-RPC spawn_actor calls and pair each call with the next
    XML-RPC spawn_actor result line. This provides CDAS-2 evidence for the
    vehicle properties CARLA was asked to use at actor creation time.
    """
    structured_call_pattern = re.compile(
        r"CDAS_EVENT\s+event=carla_spawn_request\s+"
        r"actor_type=(?P<type>\S+)\s+"
        r"actor_id=(?P<id>\S+)\s+"
        r"location=\[(?P<location>[^\]]*)\]\s+"
        r"rotation=\[(?P<rotation>[^\]]*)\]\s+"
        r"attributes=(?P<attributes>.*)$"
    )
    legacy_call_pattern = re.compile(
        r"XML-RPC spawn_actor call:\s+"
        r"type=(?P<type>[^,]+),\s+"
        r"id=(?P<id>[^,]+),\s+"
        r"location=\[(?P<location>[^\]]*)\],\s+"
        r"rotation=\[(?P<rotation>[^\]]*)\],\s+"
        r"attributes=(?P<attributes>.*)$"
    )
    structured_result_pattern = re.compile(
        r"CDAS_EVENT\s+event=carla_spawn_result\s+"
        r"actor_id=(?P<actor_id>\S+)\s+"
        r"carla_id=(?P<carla_id>\S+).*?\baccepted=true(?:\s|$)"
    )
    legacy_result_pattern = re.compile(
        r"XML-RPC spawn_actor result:\s+CARLA ID=(?P<carla_id>[^\s]+)"
    )

    spawn_calls = []
    pending_spawn = None

    for line in carla_log_lines:
        call_match = structured_call_pattern.search(line) or legacy_call_pattern.search(line)
        if call_match:
            pending_spawn = {
                "type": call_match.group("type").strip(),
                "id": call_match.group("id").strip(),
                "location": parse_float_list(call_match.group("location")),
                "rotation": parse_float_list(call_match.group("rotation")),
                "attributes": call_match.group("attributes").strip(),
                "carla_id": None,
                "call_line": line.strip(),
                "result_line": None,
            }
            spawn_calls.append(pending_spawn)
            continue

        structured_result_match = structured_result_pattern.search(line)
        if structured_result_match:
            actor_id = structured_result_match.group("actor_id").strip()
            matching_spawn = next(
                (
                    item for item in reversed(spawn_calls)
                    if item["id"] == actor_id and item["carla_id"] is None
                ),
                None,
            )
            if matching_spawn is not None:
                matching_spawn["carla_id"] = structured_result_match.group("carla_id").strip()
                matching_spawn["result_line"] = line.strip()
                if matching_spawn is pending_spawn:
                    pending_spawn = None
            continue

        legacy_result_match = legacy_result_pattern.search(line)
        if legacy_result_match and pending_spawn is not None and pending_spawn["carla_id"] is None:
            pending_spawn["carla_id"] = legacy_result_match.group("carla_id").strip()
            pending_spawn["result_line"] = line.strip()
            pending_spawn = None

    return spawn_calls


def Evaluate_CDAS_2_Carla_Vehicle_Configuration(carla_log_lines, vehicle_name: str):
    """
    CDAS-2: CARLA vehicles are configured with correct properties prior to
    simulation scenario start.

    This log-only implementation supports both vehicle creation paths:
      * SUMO-managed vehicles created through an XML-RPC spawn_actor call.
      * External CARMA vehicles created by carma-carla-integration and then
        discovered by the CARLA ambassador.

    Note: full correctness requires comparing these parsed values with the
    expected scenario configuration. This function gives log-evidence pass/fail.
    """
    spawn_calls = Parse_Carla_Spawn_Actor_Calls(carla_log_lines)
    vehicle_spawn_calls = [item for item in spawn_calls if item["id"] == vehicle_name]
    accepted_spawn_calls = [item for item in vehicle_spawn_calls if item["carla_id"]]
    detected_pattern = re.compile(
        rf"(?:New actor detected:\s+{re.escape(vehicle_name)}(?:\s|$)|"
        rf"CDAS_EVENT\s+event=carla_actor_detected\s+actor_id={re.escape(vehicle_name)}(?:\s|$))"
    )
    added_pattern = re.compile(
        rf"(?:EXTERNAL VEHICLE ADDED:\s+Actor ID={re.escape(vehicle_name)}(?:,|\b)|"
        rf"CDAS_EVENT\s+event=carla_external_vehicle_added\s+actor_id={re.escape(vehicle_name)}(?:\s|$))"
    )
    detected_external_count = sum(
        1 for line in carla_log_lines if detected_pattern.search(line)
    )
    added_external_count = sum(
        1 for line in carla_log_lines if added_pattern.search(line)
    )

    write_analysis("----- CDAS-2 CARLA vehicle configuration evidence -----")
    write_analysis(f"cdas-2 spawn_actor call count: {len(spawn_calls)}")
    write_analysis(f"cdas-2 spawn_actor call count for {vehicle_name}: {len(vehicle_spawn_calls)}")
    write_analysis(f"cdas-2 accepted spawn_actor result count for {vehicle_name}: {len(accepted_spawn_calls)}")
    write_analysis(f"cdas-2 external actor detection count for {vehicle_name}: {detected_external_count}")
    write_analysis(f"cdas-2 external vehicle added count for {vehicle_name}: {added_external_count}")

    for index, item in enumerate(vehicle_spawn_calls, start=1):
        write_analysis(
            f"cdas-2 {vehicle_name} spawn config #{index}: "
            f"type={item['type']}, id={item['id']}, "
            f"location={item['location']}, rotation={item['rotation']}, "
            f"attributes={item['attributes']}, carla_id={item['carla_id']}"
        )

    if accepted_spawn_calls:
        write_analysis(
            "cdas-2 result: pass - spawn_actor configuration found and CARLA returned an actor id "
            "for the checked vehicle"
        )
        return "pass"

    if detected_external_count > 0 and added_external_count > 0:
        write_analysis(
            "cdas-2 result: pass - external CARMA vehicle was detected and added by the CARLA ambassador"
        )
        return "pass"

    if vehicle_spawn_calls:
        write_analysis(
            "cdas-2 result: fail - spawn_actor configuration found for the checked vehicle, "
            "but no CARLA actor id result was found"
        )
        return "fail"

    write_analysis(
        "cdas-2 result: fail - no XML-RPC spawn_actor result or external CARMA vehicle addition "
        "was found for the checked vehicle"
    )
    return "fail"


def Parse_Carla_To_Sumo_VehicleUpdates_By_Timestep(carla_log_lines):
    """
    Group CARLA->SUMO VehicleUpdates publications by the following
    `carla_next_timestep` boundary.

    Preferred event sequence:
      CDAS_EVENT event=carla_vehicle_updates_published added=1 updated=0 removed=0
      CDAS_EVENT event=carla_next_timestep time_ns=100000000

    The VehicleUpdates line does not have to be immediately before the boundary;
    all publications since the previous boundary are assigned to this timestep.
    """
    structured_vehicleupdates_pattern = re.compile(
        r"CDAS_EVENT\s+event=carla_vehicle_updates_published\s+"
        r"added=(?P<added>\d+)\s+updated=(?P<updated>\d+)\s+removed=(?P<removed>\d+)"
    )
    legacy_vehicleupdates_pattern = re.compile(
        r"CARLA->SUMO SYNC:\s+Published VehicleUpdates to SUMO\s+-\s+"
        r"added=(?P<added>\d+),\s+updated=(?P<updated>\d+),\s+removed=(?P<removed>\d+)"
    )
    structured_next_step_pattern = re.compile(
        r"CDAS_EVENT\s+event=carla_next_timestep\s+time_ns=(?P<time>\d+)"
    )
    legacy_next_step_pattern = re.compile(r"Next time step:\s+(?P<time>\d+)")

    timestep_blocks = []
    pending_vehicleupdates = []

    for line in carla_log_lines:
        vu_match = (
            structured_vehicleupdates_pattern.search(line)
            or legacy_vehicleupdates_pattern.search(line)
        )
        if vu_match:
            pending_vehicleupdates.append({
                "added": int(vu_match.group("added")),
                "updated": int(vu_match.group("updated")),
                "removed": int(vu_match.group("removed")),
                "line": line.strip(),
            })
            continue

        step_match = structured_next_step_pattern.search(line) or legacy_next_step_pattern.search(line)
        if step_match:
            timestep_blocks.append({
                "time": int(step_match.group("time")),
                "vehicleupdates": pending_vehicleupdates,
            })
            pending_vehicleupdates = []

    return timestep_blocks


def Evaluate_CDAS_3_Carla_Managed_VehicleUpdates(carla_log_lines):
    """
    CDAS-3: For each timestep, CARLA generates one VehicleUpdate interaction
    per vehicle managed by CARLA.

    This implementation performs a count-level lifecycle check using
    CARLA->SUMO VehicleUpdates counts:
      - added=N starts tracking N CARLA-managed vehicles
      - updated=N must cover active vehicles on later timesteps
      - removed=N removes vehicles from the active tracked count

    Because the Carla.log line contains counts but not vehicle IDs, this proves
    count-level behavior, not identity-level per-vehicle behavior.
    """
    timestep_blocks = Parse_Carla_To_Sumo_VehicleUpdates_By_Timestep(carla_log_lines)

    write_analysis("----- CDAS-3 CARLA-managed VehicleUpdates lifecycle check -----")
    write_analysis(f"cdas-3 timestep block count from Carla.log: {len(timestep_blocks)}")

    if not timestep_blocks:
        write_analysis("cdas-3 result: fail - no Next time step blocks found in Carla.log")
        return "fail"

    active_vehicle_count = 0
    ever_tracked_vehicle = False
    checked_active_blocks = 0
    missing_active_blocks = []
    insufficient_update_blocks = []
    negative_active_blocks = []
    total_added = 0
    total_updated = 0
    total_removed = 0

    for block in timestep_blocks:
        block_time = block["time"]
        vehicleupdates = block["vehicleupdates"]
        added = sum(item["added"] for item in vehicleupdates)
        updated = sum(item["updated"] for item in vehicleupdates)
        removed = sum(item["removed"] for item in vehicleupdates)

        total_added += added
        total_updated += updated
        total_removed += removed

        previous_active_count = active_vehicle_count

        if previous_active_count > 0:
            checked_active_blocks += 1

            if not vehicleupdates:
                missing_active_blocks.append((block_time, previous_active_count))
            elif updated + removed < previous_active_count:
                insufficient_update_blocks.append(
                    (block_time, previous_active_count, added, updated, removed)
                )

        if added > 0:
            ever_tracked_vehicle = True

        active_vehicle_count = previous_active_count + added - removed
        if active_vehicle_count < 0:
            negative_active_blocks.append((block_time, previous_active_count, added, updated, removed))
            active_vehicle_count = 0

    write_analysis(f"cdas-3 total CARLA->SUMO added count: {total_added}")
    write_analysis(f"cdas-3 total CARLA->SUMO updated count: {total_updated}")
    write_analysis(f"cdas-3 total CARLA->SUMO removed count: {total_removed}")
    write_analysis(f"cdas-3 active vehicle timestep blocks checked: {checked_active_blocks}")
    write_analysis(f"cdas-3 missing VehicleUpdates blocks while active count: {len(missing_active_blocks)}")
    write_analysis(f"cdas-3 insufficient update/remove blocks count: {len(insufficient_update_blocks)}")
    write_analysis(f"cdas-3 negative active-count correction count: {len(negative_active_blocks)}")

    for block_time, expected_active in missing_active_blocks[:20]:
        write_analysis(
            f"cdas-3 missing VehicleUpdates while active: time={block_time}, "
            f"expected_active={expected_active}"
        )

    for block_time, expected_active, added, updated, removed in insufficient_update_blocks[:20]:
        write_analysis(
            f"cdas-3 insufficient update/remove: time={block_time}, "
            f"expected_active={expected_active}, added={added}, updated={updated}, removed={removed}"
        )

    if not ever_tracked_vehicle:
        write_analysis(
            "cdas-3 result: fail - no CARLA-managed vehicle add event was found in CARLA->SUMO VehicleUpdates"
        )
        return "fail"

    if missing_active_blocks or insufficient_update_blocks or negative_active_blocks:
        write_analysis(
            "cdas-3 result: fail - one or more active CARLA-managed timestep blocks "
            "did not have enough update/remove evidence"
        )
        return "fail"

    write_analysis(
        "cdas-3 result: pass - count-level lifecycle check passed for CARLA-managed VehicleUpdates"
    )
    return "pass"



def Parse_Carla_Next_Time_Steps(carla_log_lines):
    """
    Parse CARLA timestep progression logs.

    Preferred event:
      CDAS_EVENT event=carla_next_timestep time_ns=100000000

    These logs are used as CDAS-5 evidence that CARLA has completed the
    current simulation step and is proceeding only after timestep advancement.
    """
    structured_pattern = re.compile(
        r"CDAS_EVENT\s+event=carla_next_timestep\s+time_ns=(?P<time>\d+)"
    )
    legacy_pattern = re.compile(r"Next time step:\s+(?P<time>\d+)")
    steps = []

    for line in carla_log_lines:
        match = structured_pattern.search(line) or legacy_pattern.search(line)
        if match:
            steps.append({
                "time": int(match.group("time")),
                "line": line.strip(),
            })

    return steps


def Evaluate_CDAS_5_Carla_Timestep_Progression(carla_log_lines):
    """
    CDAS-5: For each timestep, after CARLA has generated and processed the
    required updates, CARLA waits for the next TimestepAdvanceGrant before
    proceeding.

    The available CARLA log evidence is the repeated `carla_next_timestep` event.
    This function checks that those timestep values exist and strictly increase.
    A constant interval is treated as a stronger PASS; increasing but non-constant
    intervals are reported as PARTIAL because progression exists but the step size
    is not uniform.
    """
    steps = Parse_Carla_Next_Time_Steps(carla_log_lines)

    write_analysis("----- CDAS-5 CARLA timestep progression check -----")
    write_analysis(f"cdas-5 next timestep log count: {len(steps)}")

    if not steps:
        write_analysis("cdas-5 result: fail - no Next time step logs found")
        return "fail"

    times = [item["time"] for item in steps]
    deltas = []
    non_increasing = []

    for previous_time, current_time in zip(times, times[1:]):
        delta = current_time - previous_time
        deltas.append(delta)

        if delta <= 0:
            non_increasing.append((previous_time, current_time, delta))

    write_analysis(f"cdas-5 first next timestep ns: {times[0]}")
    write_analysis(f"cdas-5 last next timestep ns: {times[-1]}")
    write_analysis(f"cdas-5 interval count: {len(deltas)}")

    if deltas:
        unique_deltas = sorted(set(deltas))
        write_analysis(f"cdas-5 min interval ns: {min(deltas)}")
        write_analysis(f"cdas-5 max interval ns: {max(deltas)}")
        write_analysis(f"cdas-5 unique interval count: {len(unique_deltas)}")
        write_analysis(f"cdas-5 first 10 unique intervals ns: {unique_deltas[:10]}")

    write_analysis(f"cdas-5 non-increasing timestep count: {len(non_increasing)}")

    for previous_time, current_time, delta in non_increasing[:20]:
        write_analysis(
            f"cdas-5 non-increasing timestep: previous={previous_time}, "
            f"current={current_time}, delta={delta}"
        )

    if non_increasing:
        write_analysis("cdas-5 result: fail - Next time step values are not strictly increasing")
        return "fail"

    if len(times) == 1:
        write_analysis(
            "cdas-5 result: partial - only one Next time step value found, so progression cannot be strongly verified"
        )
        return "partial"

    if deltas and len(set(deltas)) > 1:
        write_analysis(
            "cdas-5 result: partial - Next time step values increase, but timestep interval is not constant"
        )
        return "partial"

    write_analysis("cdas-5 result: pass - Next time step values are strictly increasing")
    return "pass"


def Parse_Carla_Received_VehicleUpdates(carla_log_lines):
    """
    Parse CARLA-side VehicleUpdates interactions received from CDASim.

    Preferred event:
      CDAS_EVENT event=carla_vehicle_updates_received time_ns=6700000000 added=0 updated=2 removed=0
    """
    structured_pattern = re.compile(
        r"CDAS_EVENT\s+event=carla_vehicle_updates_received\s+"
        r"time_ns=(?P<time>\d+)\s+added=(?P<added>\d+)\s+"
        r"updated=(?P<updated>\d+)\s+removed=(?P<removed>\d+)"
    )
    legacy_pattern = re.compile(
        r"Received VehicleUpdates interaction at time\s+(?P<time>\d+):\s+"
        r"added=(?P<added>\d+),\s+updated=(?P<updated>\d+),\s+removed=(?P<removed>\d+)"
    )

    updates = []

    for line in carla_log_lines:
        match = structured_pattern.search(line) or legacy_pattern.search(line)
        if match:
            updates.append({
                "time": int(match.group("time")),
                "added": int(match.group("added")),
                "updated": int(match.group("updated")),
                "removed": int(match.group("removed")),
                "line": line.strip(),
            })

    return updates


def Find_Carla_XmlRpc_Success_Line(carla_log_lines):
    """
    Find the first log line where the CARLA XML-RPC client reports a successful
    connection to the CARLA XML-RPC server.

    Preferred event:
      CDAS_EVENT event=carla_xmlrpc_connected connected=true
    """
    success_patterns = [
        r"CDAS_EVENT\s+event=carla_xmlrpc_connected\s+connected=true",
        r"Successfully connected to CARLA XML-RPC server",
        r"CDAS_EVENT\s+event=carla_actor_connection_status\b.*\bconnected=true",
        r"Multi-XML-RPC manager actor connection status:\s*true",
    ]

    for index, line in enumerate(carla_log_lines):
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in success_patterns):
            return index

    return None


def Count_Actor_Not_Connected_Before_After_XmlRpc_Success(carla_log_lines):
    """
    Count actor-server-not-connected warnings before and after CARLA XML-RPC
    connection success.

    Warnings before XML-RPC success are expected during initialization and should
    not downgrade CDAS-6. Warnings after XML-RPC success suggest CARLA may still
    be unable to apply some VehicleUpdates.
    """
    warning_pattern = re.compile(
        r"CDAS_EVENT\s+event=carla_vehicle_sync_skipped\b.*\breason=actor_server_disconnected|"
        r"actor.*not connected|not connected.*actor",
        re.IGNORECASE,
    )
    success_index = Find_Carla_XmlRpc_Success_Line(carla_log_lines)

    before_count = 0
    after_count = 0
    total_count = 0

    for index, line in enumerate(carla_log_lines):
        if not warning_pattern.search(line):
            continue

        total_count += 1

        if success_index is None:
            # Without a success marker, keep all warnings in the "after/unknown" bucket.
            after_count += 1
        elif index < success_index:
            before_count += 1
        else:
            after_count += 1

    return {
        "success_index": success_index,
        "total_count": total_count,
        "before_success_count": before_count,
        "after_success_count": after_count,
    }


def Evaluate_CDAS_6_Carla_Consumes_VehicleUpdates(carla_log_lines):
    """
    CDAS-6: For each timestep, CARLA consumes VehicleUpdates from CDASim and
    updates appropriate vehicles in the simulation.

    This log-only check verifies that CARLA received VehicleUpdates interaction
    messages and that those interactions contain non-zero added/updated/removed
    vehicle record counts.

    Important startup handling:
      actor-server-not-connected warnings before
      "CDAS_EVENT event=carla_xmlrpc_connected connected=true"
      are treated as initialization noise and do not downgrade the result.

      actor-server-not-connected warnings after that successful connection line
      downgrade the result to PARTIAL.
    """
    updates = Parse_Carla_Received_VehicleUpdates(carla_log_lines)
    actor_connection_counts = Count_Actor_Not_Connected_Before_After_XmlRpc_Success(carla_log_lines)

    write_analysis("----- CDAS-6 CARLA consumes VehicleUpdates check -----")
    write_analysis(f"cdas-6 received VehicleUpdates interaction count: {len(updates)}")
    write_analysis(
        f"cdas-6 CARLA XML-RPC success line index: "
        f"{actor_connection_counts['success_index'] if actor_connection_counts['success_index'] is not None else 'not found'}"
    )
    write_analysis(f"cdas-6 actor server not connected total warning count: {actor_connection_counts['total_count']}")
    write_analysis(f"cdas-6 actor server not connected before XML-RPC success count: {actor_connection_counts['before_success_count']}")
    write_analysis(f"cdas-6 actor server not connected after XML-RPC success count: {actor_connection_counts['after_success_count']}")

    if not updates:
        write_analysis("cdas-6 result: fail - no Received VehicleUpdates interaction logs found")
        return "fail"

    unique_times = sorted({item["time"] for item in updates})
    total_added = sum(item["added"] for item in updates)
    total_updated = sum(item["updated"] for item in updates)
    total_removed = sum(item["removed"] for item in updates)
    total_vehicle_records = total_added + total_updated + total_removed

    write_analysis(f"cdas-6 unique interaction time count: {len(unique_times)}")
    write_analysis(f"cdas-6 first interaction time ns: {unique_times[0]}")
    write_analysis(f"cdas-6 last interaction time ns: {unique_times[-1]}")
    write_analysis(f"cdas-6 total added received: {total_added}")
    write_analysis(f"cdas-6 total updated received: {total_updated}")
    write_analysis(f"cdas-6 total removed received: {total_removed}")
    write_analysis(f"cdas-6 total vehicle records received: {total_vehicle_records}")

    if total_vehicle_records == 0:
        write_analysis(
            "cdas-6 result: fail - VehicleUpdates interactions were found, but all added/updated/removed counts were zero"
        )
        return "fail"

    if actor_connection_counts["success_index"] is None:
        write_analysis(
            "cdas-6 result: partial - VehicleUpdates were received, but CARLA XML-RPC success line was not found"
        )
        return "partial"

    if actor_connection_counts["after_success_count"] > 0:
        write_analysis(
            "cdas-6 result: partial - VehicleUpdates were received, but actor-server-not-connected warnings continued after XML-RPC success"
        )
        return "partial"

    write_analysis(
        "cdas-6 result: pass - CARLA received VehicleUpdates with vehicle records, and actor-server-not-connected warnings only occurred before XML-RPC success"
    )
    return "pass"

def Count_Carla_TrafficLightUpdates_Processed(carla_log_lines):
    """
    Count CARLA-side TrafficLightUpdates processing logs.

    Expected evidence:
      Processing TrafficLightUpdates interaction - this should forward traffic light commands to CARLA
    """
    patterns = [
        "CDAS_EVENT event=carla_traffic_light_updates_received",
        "Processing TrafficLightUpdates interaction",
        "TrafficLightUpdates interaction",
    ]

    count = sum(
        1 for line in carla_log_lines
        if any(pattern in line for pattern in patterns)
    )

    write_analysis(f"cdas-7 TrafficLightUpdates processing count: {count}")
    return count


def Evaluate_CDAS_7_Carla_TrafficLightUpdates(carla_log_lines):
    """
    CDAS-7: For each timestep, for each virtual signal controller simulated in
    CDASim, CARLA receives the updated signal status and updates the state of
    the simulated traffic light.

    This log-only check verifies that CARLA processed TrafficLightUpdates
    interactions. If the count is lower than the Next time step count, it returns
    PARTIAL because traffic lights were processed but not clearly for every
    parsed timestep.
    """
    steps = Parse_Carla_Next_Time_Steps(carla_log_lines)
    traffic_light_count = Count_Carla_TrafficLightUpdates_Processed(carla_log_lines)

    write_analysis("----- CDAS-7 CARLA TrafficLightUpdates check -----")
    write_analysis(f"cdas-7 next timestep log count: {len(steps)}")
    write_analysis(f"cdas-7 TrafficLightUpdates processing count: {traffic_light_count}")

    if traffic_light_count == 0:
        write_analysis("cdas-7 result: fail - no TrafficLightUpdates processing logs found")
        return "fail"

    if steps and traffic_light_count < len(steps):
        ratio = traffic_light_count / len(steps)
        write_analysis(f"cdas-7 TrafficLightUpdates-to-timestep ratio: {ratio:.6f}")
        write_analysis(
            "cdas-7 result: partial - TrafficLightUpdates were processed, but fewer logs than timestep records were found"
        )
        return "partial"

    write_analysis("cdas-7 result: pass - TrafficLightUpdates processing logs found")
    return "pass"


# =========================
# CDAS-15 / 16 / 17 / 18  -  V2X node-to-node delivery + latency
# =========================
#
# All four are driven by CommunicationDetails.log:
#     insertV2XMessage: id=<id> from node ID[int=<sender> , ext=N] ... time=<ns>   -> SENT
#     Receive V2XMessage : Id(<id>) on Node <receiver> at Time=<ns>                -> RECEIVED
#
# The V2X message id is globally unique and stable across the NS-3 hop, so it is
# the correct key to track a message from sender to receiver (the interaction
# 'id=' in RuntimeEvents.csv is NOT stable and must not be used here).

def Parse_Comm_Sent_By_Node(comm_log_lines, sender_node):
    """
    Return {msg_id(int): {"send_time_ns": int, "wall": datetime|None}} for every
    V2X message inserted (sent) by sender_node. First occurrence of an id wins.
    """
    structured_pattern = re.compile(
        rf"CDAS_EVENT\s+event=v2x_message_inserted\s+message_id=(?P<id>\d+)\s+"
        rf"sender={re.escape(sender_node)}(?:\s|$).*?\btime_ns=(?P<time>\d+)"
    )
    legacy_pattern = re.compile(
        rf"insertV2XMessage:\s+id=(?P<id>\d+)\s+from node ID\[int={re.escape(sender_node)}\s*,[^\]]*\].*?time=(?P<time>\d+)"
    )

    sent = {}
    for line in comm_log_lines:
        match = structured_pattern.search(line) or legacy_pattern.search(line)
        if match:
            msg_id = int(match.group("id"))
            if msg_id not in sent:
                sent[msg_id] = {
                    "send_time_ns": int(match.group("time")),
                    "wall": parse_log_timestamp(line),
                }
    return sent


def Parse_Comm_Received_By_Node(comm_log_lines, receiver_node):
    """
    Return {msg_id(int): {"recv_time_ns": int, "wall": datetime|None}} for every
    V2X message received by receiver_node. First reception of an id wins.
    """
    structured_pattern = re.compile(
        rf"CDAS_EVENT\s+event=v2x_message_received\s+message_id=(?P<id>\d+)\s+"
        rf"receiver={re.escape(receiver_node)}(?:\s|$).*?\btime_ns=(?P<time>\d+)"
    )
    legacy_pattern = re.compile(
        rf"Receive V2XMessage : Id\((?P<id>\d+)\) on Node {re.escape(receiver_node)}\s+at Time=(?P<time>\d+)"
    )

    received = {}
    for line in comm_log_lines:
        match = structured_pattern.search(line) or legacy_pattern.search(line)
        if match:
            msg_id = int(match.group("id"))
            if msg_id not in received:
                received[msg_id] = {
                    "recv_time_ns": int(match.group("time")),
                    "wall": parse_log_timestamp(line),
                }
    return received


def Parse_Comm_Node_Names(comm_log_lines):
    """Return all V2X node names observed as senders, receivers, or added nodes."""
    patterns = [
        re.compile(r"CDAS_EVENT\s+event=v2x_message_inserted\s+.*?\bsender=(?P<node>\S+)"),
        re.compile(r"CDAS_EVENT\s+event=v2x_message_received\s+.*?\breceiver=(?P<node>\S+)"),
        re.compile(r"insertV2XMessage:.*?from node ID\[int=(?P<node>[^,\s]+)\s*,"),
        re.compile(r"Receive V2XMessage : Id\(\d+\) on Node (?P<node>\S+)\s+at Time="),
        re.compile(r"Adding (?:vehicle|infrastructure) ID\[int=(?P<node>[^,\s]+)\s*,"),
    ]
    nodes = set()
    for line in comm_log_lines:
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                nodes.add(match.group("node"))
    return sorted(nodes)


def _nodes_with_prefix(nodes, prefix):
    return [node for node in nodes if node.startswith(prefix)]


def _combine_delivery_results(results):
    if not results:
        return "not_applicable"
    if "fail" in results:
        return "fail"
    if "partial" in results:
        return "partial"
    if all(result == "not_applicable" for result in results):
        return "not_applicable"
    return "pass"


def Evaluate_Node_To_Node_Delivery(comm_log_lines, cdas_label, sender_node, receiver_nodes):
    """
    Count how many V2X messages sent by sender_node were received by each node in
    receiver_nodes, by matching the V2X message id in CommunicationDetails.log.

    Result:
      not_applicable - no matching target nodes are present in this scenario
      fail    - sender sent nothing, or a target node received none of them
      partial - some messages were not received (beyond the sim-end tail)
      pass    - every target received all of sender's messages (sim-end tail excepted)

    Sim-end tail: messages inserted on the final simulation timestep may not be
    received before the run terminates; those misses are reported separately and
    do not, by themselves, downgrade the result.
    """
    if not receiver_nodes:
        write_analysis(
            f"{cdas_label} result: not applicable - no matching target nodes were present in CommunicationDetails.log"
        )
        return "not_applicable"

    sent = Parse_Comm_Sent_By_Node(comm_log_lines, sender_node)
    sent_ids = set(sent)

    write_analysis(
        f"----- {cdas_label.upper()} {sender_node} -> {', '.join(receiver_nodes)} V2X delivery check -----"
    )
    write_analysis(f"{cdas_label} messages sent by {sender_node}: {len(sent_ids)}")

    if not sent_ids:
        write_analysis(
            f"{cdas_label} result: fail - no V2X messages sent by {sender_node} in CommunicationDetails.log"
        )
        return "fail"

    max_send_time = max(item["send_time_ns"] for item in sent.values())

    any_target_received_none = False
    any_target_below_threshold = False

    for receiver in receiver_nodes:
        received_ids = set(Parse_Comm_Received_By_Node(comm_log_lines, receiver))
        delivered = sent_ids & received_ids
        missing = sent_ids - received_ids

        # messages inserted on the last sim step may not be received before sim end
        tail_missing = {mid for mid in missing if sent[mid]["send_time_ns"] == max_send_time}
        real_missing = missing - tail_missing

        rate = len(delivered) / len(sent_ids)
        write_analysis(
            f"{cdas_label} {sender_node} -> {receiver}: "
            f"delivered={len(delivered)}/{len(sent_ids)} ({rate * 100:.2f}%), "
            f"missing={len(missing)}, sim-end-tail missing={len(tail_missing)}, "
            f"real missing={len(real_missing)}"
        )

        if real_missing:
            sample = sorted(real_missing)[:20]
            write_analysis(f"{cdas_label} {sender_node} -> {receiver} real-missing sample ids: {sample}")

        if len(delivered) == 0:
            any_target_received_none = True
        if rate <= CDAS_PASS_RATE:
            any_target_below_threshold = True

    if any_target_received_none:
        write_analysis(
            f"{cdas_label} result: fail - at least one target node received none of {sender_node}'s messages"
        )
        return "fail"

    if any_target_below_threshold:
        write_analysis(
            f"{cdas_label} result: partial - a target node's delivery rate was at or below "
            f"{CDAS_PASS_RATE * 100:.0f}%"
        )
        return "partial"

    write_analysis(
        f"{cdas_label} result: pass - every target node's delivery rate was above {CDAS_PASS_RATE * 100:.0f}%"
    )
    return "pass"


def Evaluate_CDAS_15_Msger_To_Carma(comm_log_lines):
    """CDAS-15"""
    nodes = Parse_Comm_Node_Names(comm_log_lines)
    messenger_nodes = _nodes_with_prefix(nodes, "msger_")
    carma_nodes = _nodes_with_prefix(nodes, "carma_")

    if not messenger_nodes:
        write_analysis(
            "cdas-15 result: not applicable - no CARMA Messenger node was present in CommunicationDetails.log"
        )
        return "not_applicable"

    results = [
        Evaluate_Node_To_Node_Delivery(
            comm_log_lines,
            f"cdas-15 ({messenger_node})",
            messenger_node,
            carma_nodes,
        )
        for messenger_node in messenger_nodes
    ]
    return _combine_delivery_results(results)


def Evaluate_CDAS_16_Carma_To_Msger_And_Rsu(comm_log_lines, vehicle_name):
    """CDAS-16"""
    nodes = Parse_Comm_Node_Names(comm_log_lines)
    target_nodes = _nodes_with_prefix(nodes, "msger_") + _nodes_with_prefix(nodes, "rsu_")
    return Evaluate_Node_To_Node_Delivery(
        comm_log_lines,
        "cdas-16",
        vehicle_name,
        target_nodes,
    )


def Evaluate_CDAS_17_Rsu_To_Carma(comm_log_lines, vehicle_name):
    """CDAS-17"""
    nodes = Parse_Comm_Node_Names(comm_log_lines)
    rsu_nodes = _nodes_with_prefix(nodes, "rsu_")

    if not rsu_nodes:
        write_analysis(
            "cdas-17 result: not applicable - no RSU node was present in CommunicationDetails.log"
        )
        return "not_applicable"

    results = [
        Evaluate_Node_To_Node_Delivery(
            comm_log_lines,
            f"cdas-17 ({rsu_node})",
            rsu_node,
            [vehicle_name],
        )
        for rsu_node in rsu_nodes
    ]
    return _combine_delivery_results(results)


def Parse_Comm_All_Sent(comm_log_lines):
    """Return {msg_id(int): {"sender": str, "send_time_ns": int}} for all senders."""
    structured_pattern = re.compile(
        r"CDAS_EVENT\s+event=v2x_message_inserted\s+message_id=(?P<id>\d+)\s+"
        r"sender=(?P<sender>\S+).*?\btime_ns=(?P<time>\d+)"
    )
    legacy_pattern = re.compile(
        r"insertV2XMessage:\s+id=(?P<id>\d+)\s+from node ID\[int=(?P<sender>[^,\s]+)\s*,[^\]]*\].*?time=(?P<time>\d+)"
    )

    sent = {}
    for line in comm_log_lines:
        match = structured_pattern.search(line) or legacy_pattern.search(line)
        if match:
            msg_id = int(match.group("id"))
            if msg_id not in sent:
                sent[msg_id] = {
                    "sender": match.group("sender"),
                    "send_time_ns": int(match.group("time")),
                }
    return sent


def Parse_Comm_All_Received(comm_log_lines):
    """Return a list of {"id": int, "receiver": str, "recv_time_ns": int} reception events."""
    structured_pattern = re.compile(
        r"CDAS_EVENT\s+event=v2x_message_received\s+message_id=(?P<id>\d+)\s+"
        r"receiver=(?P<receiver>\S+)\s+time_ns=(?P<time>\d+)"
    )
    legacy_pattern = re.compile(
        r"Receive V2XMessage : Id\((?P<id>\d+)\) on Node (?P<receiver>\S+)\s+at Time=(?P<time>\d+)"
    )

    received = []
    for line in comm_log_lines:
        match = structured_pattern.search(line) or legacy_pattern.search(line)
        if match:
            received.append({
                "id": int(match.group("id")),
                "receiver": match.group("receiver"),
                "recv_time_ns": int(match.group("time")),
            })
    return received


def Percentile(sorted_values, fraction):
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_values:
        return 0
    index = int(round(fraction * (len(sorted_values) - 1)))
    index = max(0, min(len(sorted_values) - 1, index))
    return sorted_values[index]


def Evaluate_CDAS_18_V2X_Latency(comm_log_lines):
    """
    CDAS-18: measure latency between a V2X message being sent (insertV2XMessage)
    and received (Receive V2XMessage).

    Latency is computed in SIMULATION time using the ns values carried in each
    line:  latency_ns = recv_time_ns - send_time_ns,  per (message id, receiver).
    This is the modeled network latency (propagation + queuing), not wall-clock
    execution time. Reported in nanoseconds and milliseconds (1 ms = 1e6 ns).

    Result:
      fail    - no sent or received events, or no matchable pairs
      partial - latency measured, but some receives were unmatched or negative
      pass    - latency measured cleanly for every received message
    """
    sent = Parse_Comm_All_Sent(comm_log_lines)
    received = Parse_Comm_All_Received(comm_log_lines)

    write_analysis("----- CDAS-18 V2X send-to-receive latency check -----")
    write_analysis(f"cdas-18 total insertV2XMessage (sent) events: {len(sent)}")
    write_analysis(f"cdas-18 total Receive V2XMessage events: {len(received)}")

    if not sent or not received:
        write_analysis("cdas-18 result: fail - missing sent or received V2X events in CommunicationDetails.log")
        return "fail"

    latencies_ns = []
    per_receiver_ns = {}
    unmatched_receives = 0
    negative_latencies = 0

    for event in received:
        send = sent.get(event["id"])
        if send is None:
            unmatched_receives += 1
            continue

        latency = event["recv_time_ns"] - send["send_time_ns"]
        if latency < 0:
            negative_latencies += 1
            continue

        latencies_ns.append(latency)
        per_receiver_ns.setdefault(event["receiver"], []).append(latency)

    write_analysis(f"cdas-18 matched send/receive pairs: {len(latencies_ns)}")
    write_analysis(f"cdas-18 unmatched receive events (no matching sent id): {unmatched_receives}")
    write_analysis(f"cdas-18 negative-latency pairs (skipped): {negative_latencies}")

    if not latencies_ns:
        write_analysis("cdas-18 result: fail - no matching send/receive pairs to measure latency")
        return "fail"

    latencies_ns.sort()
    ns_to_ms = 1_000_000.0
    count = len(latencies_ns)
    avg_ns = sum(latencies_ns) / count

    write_analysis(f"cdas-18 min latency: {latencies_ns[0]} ns ({latencies_ns[0] / ns_to_ms:.3f} ms)")
    write_analysis(f"cdas-18 avg latency: {avg_ns:.1f} ns ({avg_ns / ns_to_ms:.3f} ms)")
    write_analysis(f"cdas-18 p50 latency: {Percentile(latencies_ns, 0.50)} ns ({Percentile(latencies_ns, 0.50) / ns_to_ms:.3f} ms)")
    write_analysis(f"cdas-18 p95 latency: {Percentile(latencies_ns, 0.95)} ns ({Percentile(latencies_ns, 0.95) / ns_to_ms:.3f} ms)")
    write_analysis(f"cdas-18 p99 latency: {Percentile(latencies_ns, 0.99)} ns ({Percentile(latencies_ns, 0.99) / ns_to_ms:.3f} ms)")
    write_analysis(f"cdas-18 max latency: {latencies_ns[-1]} ns ({latencies_ns[-1] / ns_to_ms:.3f} ms)")

    for receiver in sorted(per_receiver_ns):
        values = per_receiver_ns[receiver]
        receiver_avg = sum(values) / len(values)
        write_analysis(
            f"cdas-18 {receiver}: pairs={len(values)}, "
            f"avg={receiver_avg / ns_to_ms:.3f} ms, "
            f"min={min(values) / ns_to_ms:.3f} ms, max={max(values) / ns_to_ms:.3f} ms"
        )

    total_receives = len(received)
    match_rate = (total_receives - unmatched_receives) / total_receives
    write_analysis(f"cdas-18 send/receive match rate: {match_rate * 100:.2f}%")

    if match_rate > CDAS_PASS_RATE and negative_latencies == 0:
        write_analysis(
            f"cdas-18 result: pass - latency measured and match rate above {CDAS_PASS_RATE * 100:.0f}%"
        )
        return "pass"

    write_analysis(
        f"cdas-18 result: partial - match rate at or below {CDAS_PASS_RATE * 100:.0f}% "
        f"or negative-latency pairs present"
    )
    return "partial"


def Check_TimeSync_Sent(carma_log_lines, vehicle_name: str) -> bool:
    pattern = re.compile(
        rf"(?:CDAS_EVENT\s+event=common_time_sync_sent\s+instance_id={re.escape(vehicle_name)}(?:\s|$)|"
        rf"Sending Common instance\s+{re.escape(vehicle_name)}\b.*time sync message for time)"
    )

    for line in carma_log_lines:
        if pattern.search(line):
            write_analysis(f"vehicle {vehicle_name} time sync message sent")
            return True

    write_analysis(f"vehicle {vehicle_name} time sync message not sent")
    return False


def Check_V2X_Message_Sent(carma_log_lines, vehicle_name: str) -> bool:
    pattern = re.compile(
        rf"(?:CDAS_EVENT\s+event=v2x_reception_forwarded\s+receiver={re.escape(vehicle_name)}(?:\s|$)|"
        rf"Sending V2X message reception event for\s+{re.escape(vehicle_name)}\b)"
    )

    for line in carma_log_lines:
        if pattern.search(line):
            write_analysis(f"vehicle {vehicle_name} v2x message sent")
            return True

    write_analysis(f"vehicle {vehicle_name} v2x message not sent")
    return False


def Check_Common_Instance_Registration(mosaic_log_lines, vehicle_name: str) -> bool:
    pattern = re.compile(
        rf"(?:CDAS_EVENT\s+event=common_instance_received\s+instance_id={re.escape(vehicle_name)}(?:\s|$)|"
        rf"New Common instance '{re.escape(vehicle_name)}' received)"
    )

    for line in mosaic_log_lines:
        if pattern.search(line):
            write_analysis(f"common instance registration found for {vehicle_name}")
            return True

    write_analysis(f"common instance registration not found for {vehicle_name}")
    return False


def Check_Carma_Instance_Registered(carma_log_lines, vehicle_name: str) -> bool:
    pattern = re.compile(
        rf"(?:CDAS_EVENT\s+event=carma_instance_registered\s+instance_id={re.escape(vehicle_name)}(?:\s|$)|"
        rf"New CARMA instance\s+'{re.escape(vehicle_name)}'\s+registered with CARMA Instance Manager)"
    )

    for line in carma_log_lines:
        if pattern.search(line):
            write_analysis(f"vehicle {vehicle_name} carma instance registered")
            return True

    write_analysis(f"vehicle {vehicle_name} carma instance not registered")
    return False


def Check_V2X_Message_Id_Paired(carma_log_lines, vehicle_name: str) -> int:
    processing_ids, sending_ids = get_v2x_message_ids(carma_log_lines, vehicle_name)
    paired_ids = processing_ids & sending_ids
    processing_only = processing_ids - sending_ids
    sending_only = sending_ids - processing_ids

    write_analysis(f"vehicle {vehicle_name} v2x paired msg id count: {len(paired_ids)}")
    write_analysis(f"vehicle {vehicle_name} v2x processing-only msg id count: {len(processing_only)}")
    write_analysis(f"vehicle {vehicle_name} v2x sending-only msg id count: {len(sending_only)}")

    return len(paired_ids)


def Write_V2X_Message_Id_Details(carma_log_lines, vehicle_name: str) -> None:
    processing_ids, sending_ids = get_v2x_message_ids(carma_log_lines, vehicle_name)

    paired_ids = sorted(processing_ids & sending_ids, key=int)
    processing_only = sorted(processing_ids - sending_ids, key=int)
    sending_only = sorted(sending_ids - processing_ids, key=int)

    write_analysis(f"vehicle {vehicle_name} paired v2x msg ids: {paired_ids}")
    write_analysis(f"vehicle {vehicle_name} processing-only v2x msg ids: {processing_only}")
    write_analysis(f"vehicle {vehicle_name} sending-only v2x msg ids: {sending_only}")


def get_v2x_message_ids(carma_log_lines, vehicle_name: str):
    processing_ids = set()
    sending_ids = set()

    processing_pattern = re.compile(
        rf"(?:CDAS_EVENT\s+event=v2x_reception_processing\s+receiver={re.escape(vehicle_name)}\s+message_id=|"
        rf"Processing V2X message reception event for\s+{re.escape(vehicle_name)}\s+of msg id\s+)(\d+)"
    )
    sending_pattern = re.compile(
        rf"(?:CDAS_EVENT\s+event=v2x_reception_forwarded\s+receiver={re.escape(vehicle_name)}\s+message_id=|"
        rf"Sending V2X message reception event for\s+{re.escape(vehicle_name)}\s+of msg id\s+)(\d+)"
    )

    for line in carma_log_lines:
        match = processing_pattern.search(line)
        if match:
            processing_ids.add(match.group(1))

        match = sending_pattern.search(line)
        if match:
            sending_ids.add(match.group(1))

    return processing_ids, sending_ids


def Compare_Carma_And_Comm_Message_IDs(carma_log_lines, comm_log_lines, vehicle_name: str):
    carma_ids = set()
    comm_ids = set()

    carma_pattern = re.compile(
        rf"(?:CDAS_EVENT\s+event=v2x_reception_forwarded\s+receiver={re.escape(vehicle_name)}\s+message_id=|"
        rf"Sending V2X message reception event for\s+{re.escape(vehicle_name)}\s+of msg id\s+)(\d+)"
    )
    comm_pattern = re.compile(
        r"(?:CDAS_EVENT\s+event=v2x_message_inserted\s+message_id=|insertV2XMessage:\s+id=)(\d+)"
    )

    for line in carma_log_lines:
        match = carma_pattern.search(line)
        if match:
            carma_ids.add(match.group(1))

    for line in comm_log_lines:
        match = comm_pattern.search(line)
        if match:
            comm_ids.add(match.group(1))

    matched_ids = carma_ids & comm_ids
    missing_in_comm = carma_ids - comm_ids

    write_analysis(f"matched carma->comm v2x msg ids count: {len(matched_ids)}")
    write_analysis(f"missing carma v2x ids in communicationdetails count: {len(missing_in_comm)}")

    if missing_in_comm:
        write_analysis(f"missing ids in communicationdetails: {sorted(missing_in_comm, key=int)}")

    return matched_ids


def Compare_Carma_And_Comm_Message_ID_Delays(
    carma_log_lines,
    comm_log_lines,
    vehicle_name: str,
    delay_threshold_sec: float = 0.1
):
    carma_id_to_time = {}
    comm_id_to_time = {}

    carma_pattern = re.compile(
        rf"(?:CDAS_EVENT\s+event=v2x_reception_forwarded\s+receiver={re.escape(vehicle_name)}\s+message_id=|"
        rf"Sending V2X message reception event for\s+{re.escape(vehicle_name)}\s+of msg id\s+)(\d+)"
    )
    comm_pattern = re.compile(
        r"(?:CDAS_EVENT\s+event=v2x_message_inserted\s+message_id=|insertV2XMessage:\s+id=)(\d+)"
    )

    for line in carma_log_lines:
        match = carma_pattern.search(line)
        if match:
            msg_id = match.group(1)
            if msg_id not in carma_id_to_time:
                ts = parse_log_timestamp(line)
                if ts is not None:
                    carma_id_to_time[msg_id] = ts

    for line in comm_log_lines:
        match = comm_pattern.search(line)
        if match:
            msg_id = match.group(1)
            if msg_id not in comm_id_to_time:
                ts = parse_log_timestamp(line)
                if ts is not None:
                    comm_id_to_time[msg_id] = ts

    shared_ids = sorted(set(carma_id_to_time.keys()) & set(comm_id_to_time.keys()), key=int)

    delayed_count = 0

    for msg_id in shared_ids:
        delay_sec = (comm_id_to_time[msg_id] - carma_id_to_time[msg_id]).total_seconds()

        if delay_sec > delay_threshold_sec:
            delayed_count += 1
            write_analysis(
                f"v2x msg id {msg_id} delay too high: {delay_sec:.3f} sec "
                f"(carma -> communicationdetails)"
            )

    write_analysis(
        f"v2x msg ids over {delay_threshold_sec:.3f} sec delay count: {delayed_count}"
    )

    return delayed_count


def Count_Sumo_VehicleUpdates_Interactions(traffic_log_lines) -> int:
    count = sum(
        1 for line in traffic_log_lines
        if (
            "Got new interaction VehicleUpdates with time" in line
            or (
                is_cdas_event(line, "sumo_interaction_received")
                and cdas_event_field(line, "interaction_type") == "VehicleUpdates"
            )
        )
    )
    write_analysis(f"sumo vehicleupdates interaction count: {count}")
    return count


def Count_Sumo_VehicleFederateAssignment_Interactions(traffic_log_lines) -> int:
    count = sum(
        1 for line in traffic_log_lines
        if (
            "Got new interaction VehicleFederateAssignment with time" in line
            or (
                is_cdas_event(line, "sumo_interaction_received")
                and cdas_event_field(line, "interaction_type") == "VehicleFederateAssignment"
            )
        )
    )
    write_analysis(f"sumo vehiclefederateassignment interaction count: {count}")
    return count


def Check_Sumo_Simulation_Time_Started(traffic_log_lines) -> bool:
    found = any(
        "Simulation Time:" in line or is_cdas_event(line, "sumo_simulation_time")
        for line in traffic_log_lines
    )

    if found:
        write_analysis("sumo simulation time started")
    else:
        write_analysis("sumo simulation time not started")

    return found


def Check_Federation_Started(mosaic_log_lines) -> bool:
    found = any(
        "Start federation with id" in line or is_cdas_event(line, "federation_started")
        for line in mosaic_log_lines
    )

    write_analysis(f"federation started: {'yes' if found else 'no'}")
    return found


def Count_Initialized_Federates(mosaic_log_lines) -> int:
    count = sum(
        1 for line in mosaic_log_lines
        if (
            ("is initializing" in line and "Federate " in line)
            or is_cdas_event(line, "federate_initializing")
        )
    )

    write_analysis(f"initialized federate count: {count}")
    return count


def Count_Added_Federates(mosaic_log_lines) -> int:
    count = sum(
        1 for line in mosaic_log_lines
        if "Add ambassador/federate with id" in line or is_cdas_event(line, "federate_added")
    )

    write_analysis(f"added federate count: {count}")
    return count


def Count_Common_Instance_Registrations(mosaic_log_lines, vehicle_name: str) -> int:
    pattern = re.compile(
        rf"(?:CDAS_EVENT\s+event=common_instance_received\s+instance_id={re.escape(vehicle_name)}(?:\s|$)|"
        rf"New Common instance '{re.escape(vehicle_name)}' received)"
    )
    count = sum(1 for line in mosaic_log_lines if pattern.search(line))

    write_analysis(f"common registration count for {vehicle_name}: {count}")
    return count


def Count_Carma_VehicleUpdates_Interactions(carla_log_lines) -> int:
    count = sum(
        1 for line in carla_log_lines
        if "Received VehicleUpdates interaction" in line
        or is_cdas_event(line, "carla_vehicle_updates_received")
    )
    write_analysis(f"carla vehicleupdates interaction count: {count}")
    return count


def Count_XmlRpc_Status(carla_log_lines, status: bool) -> int:
    target_phrase = (
        f"Multi-XML-RPC manager actor connection status: {str(status).lower()}"
    )
    status_value = str(status).lower()
    count = sum(
        1 for line in carla_log_lines
        if target_phrase in line
        or (
            is_cdas_event(line, "carla_actor_connection_status")
            and cdas_event_field(line, "connected") == status_value
        )
    )
    write_analysis(f"xml rpc {str(status).lower()} count: {count}")
    return count


def Check_XmlRpc_Recovered(carla_log_lines) -> bool:
    saw_false = False
    for line in carla_log_lines:
        is_false = (
            "Multi-XML-RPC manager actor connection status: false" in line
            or (
                is_cdas_event(line, "carla_actor_connection_status")
                and cdas_event_field(line, "connected") == "false"
            )
        )
        is_true = (
            "Multi-XML-RPC manager actor connection status: true" in line
            or (
                is_cdas_event(line, "carla_actor_connection_status")
                and cdas_event_field(line, "connected") == "true"
            )
        )
        if is_false:
            saw_false = True
        elif saw_false and is_true:
            write_analysis("xml rpc recovered after false status")
            return True

    write_analysis("xml rpc did not recover after false status")
    return False


def Count_TimeSync_Messages(carma_log_lines, vehicle_name: str) -> int:
    pattern = (
        rf"(?:CDAS_EVENT\s+event=common_time_sync_sent\s+instance_id={re.escape(vehicle_name)}(?:\s|$)|"
        rf"Sending Common instance\s+{re.escape(vehicle_name)}\b.*time sync message for time)"
    )
    count = count_matching_lines(carma_log_lines, pattern)
    write_analysis(f"vehicle {vehicle_name} time sync message count: {count}")
    return count


def Count_V2X_Messages(carma_log_lines, vehicle_name: str) -> int:
    pattern = (
        rf"(?:CDAS_EVENT\s+event=v2x_reception_forwarded\s+receiver={re.escape(vehicle_name)}(?:\s|$)|"
        rf"Sending V2X message reception event for\s+{re.escape(vehicle_name)}\b)"
    )
    count = count_matching_lines(carma_log_lines, pattern)
    write_analysis(f"vehicle {vehicle_name} v2x message count: {count}")
    return count


def Count_Duplicate_Registrations(carma_log_lines, vehicle_name: str) -> int:
    pattern = (
        rf"CDAS_EVENT\s+event=common_registration_duplicate\s+"
        rf"instance_id={re.escape(vehicle_name)}(?:\s|$)|"
        rf"(duplicate|already registered).*{re.escape(vehicle_name)}"
    )
    count = count_matching_lines(carma_log_lines, pattern)
    write_analysis(f"duplicate registration count for {vehicle_name}: {count}")
    return count


def Count_Comm_InsertV2X(comm_log_lines) -> int:
    count = count_matching_lines(
        comm_log_lines,
        r"CDAS_EVENT\s+event=v2x_message_inserted\s+message_id=\d+|insertV2XMessage:\s+id=\d+",
    )
    write_analysis(f"communicationdetails inserted v2x count: {count}")
    return count


def Count_Comm_SendV2X(comm_log_lines) -> int:
    count = count_matching_lines(comm_log_lines, r"(sendV2X|ns-?3.*send|send.*ns-?3)")
    write_analysis(f"communicationdetails ns3 send v2x count: {count}")
    return count


def Count_Actor_Not_Connected(carla_log_lines) -> int:
    count = count_matching_lines(
        carla_log_lines,
        r"CDAS_EVENT\s+event=carla_vehicle_sync_skipped\b.*reason=actor_server_disconnected|"
        r"actor.*not connected|not connected.*actor",
    )
    write_analysis(f"actor server not connected warning count: {count}")
    return count


def Count_Sumo_To_Carla_Sync_Starts(carla_log_lines) -> int:
    count = count_matching_lines(
        carla_log_lines,
        r"CDAS_EVENT\s+event=carla_vehicle_sync_started|sync.*sumo.*carla|sumo.*carla.*sync",
    )
    write_analysis(f"sumo->carla sync start count: {count}")
    return count


def Count_Successful_Actor_Updates(carla_log_lines) -> int:
    count = count_matching_lines(
        carla_log_lines,
        r"CDAS_EVENT\s+event=carla_actor_updated|success.*actor.*update|updated.*actor",
    )
    write_analysis(f"successful actor update count: {count}")
    return count


def Count_Existing_Actor_Mapping_Skips(carla_log_lines) -> int:
    count = count_matching_lines(
        carla_log_lines,
        r"CDAS_EVENT\s+event=carla_spawn_skipped\b.*reason=existing_mapping|"
        r"existing.*actor.*mapping|actor.*mapping.*exist",
    )
    write_analysis(f"existing actor mapping skip count: {count}")
    return count


def Count_Simulation_Unit_Warnings(application_log_lines, action: str) -> int:
    count = count_matching_lines(
        application_log_lines,
        rf"CDAS_EVENT\s+event=application_vehicle_ignored\s+action={re.escape(action)}\s+"
        rf"reason=missing_simulation_unit|{action}.*without.*simulation unit|"
        rf"without.*simulation unit.*{action}",
    )
    write_analysis(f"{action} vehicle without simulation unit count: {count}")
    return count


def Check_Sumo_Connected(traffic_log_lines) -> bool:
    found = any(
        is_cdas_event(line, "sumo_connection_established")
        or any(
            phrase in line
            for phrase in ("TraCI connection established", "Connected to SUMO", "connection to SUMO established")
        )
        for line in traffic_log_lines
    )
    write_analysis(f"sumo connection result: {'pass' if found else 'fail'}")
    return found


def Count_Sumo_Retries(traffic_log_lines) -> int:
    count = count_matching_lines(
        traffic_log_lines,
        r"CDAS_EVENT\s+event=sumo_connection_retry|retry|trying again",
    )
    write_analysis(f"sumo connection retry warning count: {count}")
    return count


def Check_Sumo_Api_Logged(traffic_log_lines) -> bool:
    found = any(
        is_cdas_event(line, "sumo_api_version")
        or any(phrase in line for phrase in ("TraCI API version", "SUMO API version"))
        for line in traffic_log_lines
    )
    write_analysis(f"sumo api version logged: {'yes' if found else 'no'}")
    return found


def Count_Sumo_Ignored_External(traffic_log_lines, pattern: str | None = None) -> int:
    ignored_lines = [
        line for line in traffic_log_lines
        if (
            is_cdas_event(line, "sumo_external_vehicle_ignored")
            or re.search(r"ignor.*external vehicle", line, re.IGNORECASE)
        )
    ]
    if pattern:
        regex = re.compile(pattern, re.IGNORECASE)
        ignored_lines = [line for line in ignored_lines if regex.search(line)]
    return len(ignored_lines)


def Check_No_Mapping_Spawners(mosaic_log_lines) -> bool:
    found = count_matching_lines(
        mosaic_log_lines,
        r"CDAS_EVENT\s+event=mapping_no_spawners|mapping.*no spawner|"
        r"no spawner|spawners?\s*[:=]\s*\[\s*\]",
    ) > 0
    write_analysis(f"mapping config has no spawners: {'yes' if found else 'no'}")
    return found


def Count_V2X_Receiver_Starts(mosaic_log_lines) -> int:
    count = sum(
        1 for line in mosaic_log_lines
        if "CarmaV2xMessageReceiver started listening on UDP port" in line
        or is_cdas_event(line, "v2x_receiver_started")
    )

    write_analysis(f"v2x receiver start count: {count}")
    return count


# =========================
# SUMMARY
# =========================

def Write_Summary(
    vehicle_name: str,
    cdas2_result: str,
    cdas3_result: str,
    cdas5_result: str,
    cdas6_result: str,
    cdas7_result: str,
    cdas15_result: str,
    cdas16_result: str,
    cdas17_result: str,
    cdas18_result: str,
    spawn_ok: bool,
    xmlrpc_ok: bool,
    xmlrpc_false_count: int,
    xmlrpc_true_count: int,
    xmlrpc_recovered: bool,
    actor_not_connected_count: int,
    vehicleupdates_count: int,
    total_added_vehicles: int,
    sync_start_count: int,
    successful_actor_updates: int,
    existing_mapping_skips: int,
    add_without_sim_count: int,
    update_without_sim_count: int,
    sim_unit_problem: bool,
    carma_registered: bool,
    time_sync_ok: bool,
    time_sync_count: int,
    v2x_ok: bool,
    v2x_count: int,
    duplicate_registration_count: int,
    v2x_paired_count: int,
    comm_insert_count: int,
    comm_send_count: int,
    comm_v2x_healthy: bool,
    matched_comm_ids_count: int,
    delayed_v2x_count: int,
    comm_non_simulated_count: int,
    comm_duplicate_vehicle_count: int,
    sumo_connected: bool,
    sumo_retry_count: int,
    sumo_api_logged: bool,
    sumo_sim_started: bool,
    sumo_vehicleupdates_count: int,
    sumo_assignment_count: int,
    sumo_ignored_external_count: int,
    sumo_ignored_vehicle_count: int,
    sumo_ignored_carma_source_count: int,
    sumo_ignored_msger_source_count: int,
    sumo_missing_assignment_issue: bool,
    federation_started: bool,
    initialized_federates: int,
    added_federates: int,
    no_mapping_spawners: bool,
    common_registration_count: int,
    common_registration_spam: bool,
    v2x_receiver_count: int,
) -> None:
    write_analysis("----- summary -----")
    write_analysis(f"vehicle checked: {vehicle_name}")
    write_analysis("----- CDAS performance metric summary -----")
    write_analysis(f"CDAS-2 result: {cdas2_result}")
    write_analysis(f"CDAS-3 result: {cdas3_result}")
    write_analysis(f"CDAS-5 result: {cdas5_result}")
    write_analysis(f"CDAS-6 result: {cdas6_result}")
    write_analysis(f"CDAS-7 result: {cdas7_result}")
    write_analysis(f"CDAS-15 result: {cdas15_result}")
    write_analysis(f"CDAS-16 result: {cdas16_result}")
    write_analysis(f"CDAS-17 result: {cdas17_result}")
    write_analysis(f"CDAS-18 result: {cdas18_result}")
    write_analysis(f"spawn result: {'pass' if spawn_ok else 'fail'}")
    write_analysis(f"xml rpc result: {'pass' if xmlrpc_ok else 'fail'}")
    write_analysis(f"xml rpc false count: {xmlrpc_false_count}")
    write_analysis(f"xml rpc true count: {xmlrpc_true_count}")
    write_analysis(f"xml rpc recovered after failures: {'yes' if xmlrpc_recovered else 'no'}")
    write_analysis(f"actor server not connected warning count: {actor_not_connected_count}")
    write_analysis(f"vehicleupdates interaction count: {vehicleupdates_count}")
    write_analysis(f"total added vehicles reported in carla log: {total_added_vehicles}")
    write_analysis(f"sumo->carla sync start count: {sync_start_count}")
    write_analysis(f"successful actor update count: {successful_actor_updates}")
    write_analysis(f"existing actor mapping skip count: {existing_mapping_skips}")
    write_analysis(f"add vehicle without simulation unit count: {add_without_sim_count}")
    write_analysis(f"update vehicle without simulation unit count: {update_without_sim_count}")
    write_analysis(f"simulation unit issue: {'yes' if sim_unit_problem else 'no'}")
    write_analysis(f"carma instance registered: {'yes' if carma_registered else 'no'}")
    write_analysis(f"time sync sent: {'yes' if time_sync_ok else 'no'}")
    write_analysis(f"time sync sent count: {time_sync_count}")
    write_analysis(f"v2x message sent: {'yes' if v2x_ok else 'no'}")
    write_analysis(f"v2x message sent count: {v2x_count}")
    write_analysis(f"duplicate registration count: {duplicate_registration_count}")
    write_analysis(f"v2x paired msg id count: {v2x_paired_count}")

    write_analysis(f"communicationdetails inserted v2x count: {comm_insert_count}")
    write_analysis(f"communicationdetails ns3 send v2x count: {comm_send_count}")
    write_analysis(f"communicationdetails v2x healthy: {'yes' if comm_v2x_healthy else 'no'}")
    write_analysis(f"matched carma->comm v2x ids count: {matched_comm_ids_count}")
    write_analysis(f"v2x msg ids over 0.100 sec delay count: {delayed_v2x_count}")
    write_analysis(f"communicationdetails non-simulated node warnings: {comm_non_simulated_count}")
    write_analysis(f"communicationdetails duplicate vehicle warnings: {comm_duplicate_vehicle_count}")

    write_analysis(f"sumo connection result: {'pass' if sumo_connected else 'fail'}")
    write_analysis(f"sumo connection retry warning count: {sumo_retry_count}")
    write_analysis(f"sumo api version logged: {'yes' if sumo_api_logged else 'no'}")
    write_analysis(f"sumo simulation time started: {'yes' if sumo_sim_started else 'no'}")
    write_analysis(f"sumo vehicleupdates interaction count: {sumo_vehicleupdates_count}")
    write_analysis(f"sumo vehiclefederateassignment interaction count: {sumo_assignment_count}")
    write_analysis(f"sumo ignored external vehicle count: {sumo_ignored_external_count}")
    write_analysis(f"sumo ignored external vehicle count for {vehicle_name}: {sumo_ignored_vehicle_count}")
    write_analysis(f"sumo ignored external vehicle count from carma: {sumo_ignored_carma_source_count}")
    write_analysis(f"sumo ignored external vehicle count from carma-messenger: {sumo_ignored_msger_source_count}")
    write_analysis(f"sumo missing VehicleFederateAssignment issue: {'yes' if sumo_missing_assignment_issue else 'no'}")

    write_analysis(f"federation started: {'yes' if federation_started else 'no'}")
    write_analysis(f"initialized federate count: {initialized_federates}")
    write_analysis(f"added federate count: {added_federates}")
    write_analysis(f"mapping config has no spawners: {'yes' if no_mapping_spawners else 'no'}")
    write_analysis(f"common registration count for {vehicle_name}: {common_registration_count}")
    write_analysis(f"common registration spam detected: {'yes' if common_registration_spam else 'no'}")
    write_analysis(f"v2x receiver start count: {v2x_receiver_count}")


def Write_Root_Cause_Hints(
    vehicle_name: str,
    spawn_ok: bool,
    xmlrpc_ok: bool,
    xmlrpc_recovered: bool,
    actor_not_connected_count: int,
    sim_unit_problem: bool,
    add_without_sim_count: int,
    update_without_sim_count: int,
    sync_start_count: int,
    successful_actor_updates: int,
    time_sync_ok: bool,
    v2x_ok: bool,
    duplicate_registration_count: int,
    comm_v2x_healthy: bool,
    sumo_retry_count: int,
    sumo_connected: bool,
    sumo_missing_assignment_issue: bool,
    sumo_ignored_vehicle_count: int,
    sumo_ignored_msger_source_count: int,
    no_mapping_spawners: bool,
    common_registration_spam: bool,
    initialized_federates: int,
    added_federates: int,
) -> None:
    write_analysis("----- root cause hints -----")

    if not xmlrpc_ok:
        write_analysis("possible issue: actor xml rpc server did not become healthy")

    if xmlrpc_recovered:
        write_analysis("possible issue: actor xml rpc server had delayed startup / temporary readiness issue")

    if actor_not_connected_count > 0:
        write_analysis("possible issue: SUMO->CARLA sync was skipped during part of startup because actor server was not connected")

    if sim_unit_problem and (add_without_sim_count > 0 or update_without_sim_count > 0):
        write_analysis("possible issue: application side is missing or not attaching a simulation unit")

    if sync_start_count > 0 and successful_actor_updates > 0:
        write_analysis("actor synchronization eventually became healthy after initialization")

    if time_sync_ok:
        write_analysis("carma time sync messaging is active")

    if v2x_ok:
        write_analysis("carma v2x message flow is active")

    if duplicate_registration_count > 0:
        write_analysis("possible issue: carma vehicle registration is being repeated many times")

    if comm_v2x_healthy:
        write_analysis("communicationdetails indicates the network-layer v2x pipeline is active")

    if sumo_retry_count > 0 and sumo_connected:
        write_analysis("sumo needed at least one retry before TraCI connection succeeded")

    if sumo_missing_assignment_issue:
        write_analysis("possible issue: external vehicles are reaching SUMO before VehicleFederateAssignment is established")

    if sumo_ignored_vehicle_count > 0:
        write_analysis(f"possible issue: vehicle {vehicle_name} was ignored by SUMO because no prior VehicleFederateAssignment was seen")

    if sumo_ignored_msger_source_count > 0:
        write_analysis("possible issue: carma-messenger vehicles were ignored by SUMO because no prior VehicleFederateAssignment was seen")

    if no_mapping_spawners:
        write_analysis("note: mapping config contains no spawners; only externally injected vehicles will appear")

    if common_registration_spam:
        write_analysis(f"possible issue: repeated Common registration for {vehicle_name} may indicate registration/assignment handshake loop")

    if initialized_federates != added_federates:
        write_analysis("possible issue: some initialized federates may not have been fully added to federation")

    if not spawn_ok and (not xmlrpc_ok or sim_unit_problem or sumo_missing_assignment_issue):
        write_analysis("possible issue: vehicle spawn failed because prerequisites were not healthy early in startup")


def get_latest_log_dir(base_path="/opt/carma-simulation/logs") -> Path:
    base = Path(base_path)

    if not base.exists():
        raise FileNotFoundError(f"Base log path not found: {base_path}")
    dirs = [d for d in base.iterdir() if d.is_dir()]

    if not dirs:
        raise RuntimeError("No log directories found")

    latest_dir = max(dirs, key=lambda d: d.stat().st_mtime)

    return latest_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze CDASim logs from a chosen log directory."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        type=Path,
        help=(
            "Directory containing Carla.log, Application.log, Carma.log, "
            "CommunicationDetails.log, Traffic.log, and MOSAIC.log. "
            "If omitted, the newest directory under --log-base is used."
        ),
    )
    parser.add_argument(
        "--log-base",
        default=DEFAULT_LOG_BASE,
        help="Base directory to search for the newest log directory when log_dir is omitted.",
    )
    parser.add_argument(
        "--vehicle-name",
        required=True,
        help="Vehicle name to check in the analysis (for example, carma_1).",
    )
    return parser.parse_args()


def resolve_log_dir(args) -> Path:
    if args.log_dir:
        log_dir = args.log_dir.expanduser()
        if not log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {log_dir}")
        if not log_dir.is_dir():
            raise NotADirectoryError(f"Log path is not a directory: {log_dir}")
        return log_dir

    return get_latest_log_dir(args.log_base)


def main() -> None:
    global ANALYSIS_LOG

    args = parse_args()
    vehicle_name = args.vehicle_name

    log_dir = resolve_log_dir(args)
    ANALYSIS_LOG = build_analysis_log_path(log_dir)

    carla_log_path = log_dir / "Carla.log"
    application_log_path = log_dir / "Application.log"
    carma_log_path = log_dir / "Carma.log"
    comm_log_path = log_dir / "CommunicationDetails.log"
    traffic_log_path = log_dir / "Traffic.log"
    mosaic_log_path = log_dir / "MOSAIC.log"

    with open(ANALYSIS_LOG, "w", encoding="utf-8") as out:
        out.write("CDASim CDAS performance metric analysis\n")

    write_analysis(f"using log directory: {log_dir}")

    carla_log_lines = read_log_lines(carla_log_path)
    application_log_lines = read_log_lines(application_log_path)
    carma_log_lines = read_log_lines(carma_log_path)
    comm_log_lines = read_log_lines(comm_log_path)
    traffic_log_lines = read_log_lines(traffic_log_path)
    mosaic_log_lines = read_log_lines(mosaic_log_path)

    spawn_ok = Check_Vehicle_Spawn(carla_log_lines, vehicle_name)
    xmlrpc_ok = CheckXmlRpcServer(carla_log_lines)
    xmlrpc_false_count = Count_XmlRpc_Status(carla_log_lines, False)
    xmlrpc_true_count = Count_XmlRpc_Status(carla_log_lines, True)
    xmlrpc_recovered = Check_XmlRpc_Recovered(carla_log_lines)
    actor_not_connected_count = Count_Actor_Not_Connected(carla_log_lines)
    vehicleupdates_count = Count_Carma_VehicleUpdates_Interactions(carla_log_lines)
    total_added_vehicles = Count_Added_Vehicles_In_Carla_Log(carla_log_lines)
    cdas2_result = Evaluate_CDAS_2_Carla_Vehicle_Configuration(carla_log_lines, vehicle_name)
    cdas3_result = Evaluate_CDAS_3_Carla_Managed_VehicleUpdates(carla_log_lines)
    cdas5_result = Evaluate_CDAS_5_Carla_Timestep_Progression(carla_log_lines)
    cdas6_result = Evaluate_CDAS_6_Carla_Consumes_VehicleUpdates(carla_log_lines)
    cdas7_result = Evaluate_CDAS_7_Carla_TrafficLightUpdates(carla_log_lines)
    cdas15_result = Evaluate_CDAS_15_Msger_To_Carma(comm_log_lines)
    cdas16_result = Evaluate_CDAS_16_Carma_To_Msger_And_Rsu(comm_log_lines, vehicle_name)
    cdas17_result = Evaluate_CDAS_17_Rsu_To_Carma(comm_log_lines, vehicle_name)
    cdas18_result = Evaluate_CDAS_18_V2X_Latency(comm_log_lines)
    sync_start_count = Count_Sumo_To_Carla_Sync_Starts(carla_log_lines)
    successful_actor_updates = Count_Successful_Actor_Updates(carla_log_lines)
    existing_mapping_skips = Count_Existing_Actor_Mapping_Skips(carla_log_lines)
    add_without_sim_count = Count_Simulation_Unit_Warnings(application_log_lines, "add")
    update_without_sim_count = Count_Simulation_Unit_Warnings(application_log_lines, "update")
    sim_unit_problem = add_without_sim_count > 0 or update_without_sim_count > 0
    carma_registered = Check_Carma_Instance_Registered(carma_log_lines, vehicle_name)
    time_sync_ok = Check_TimeSync_Sent(carma_log_lines, vehicle_name)
    time_sync_count = Count_TimeSync_Messages(carma_log_lines, vehicle_name)
    v2x_ok = Check_V2X_Message_Sent(carma_log_lines, vehicle_name)
    v2x_count = Count_V2X_Messages(carma_log_lines, vehicle_name)
    duplicate_registration_count = Count_Duplicate_Registrations(carma_log_lines, vehicle_name)
    Check_Common_Instance_Registration(mosaic_log_lines, vehicle_name)
    v2x_paired_count = Check_V2X_Message_Id_Paired(carma_log_lines, vehicle_name)
    Write_V2X_Message_Id_Details(carma_log_lines, vehicle_name)
    comm_insert_count = Count_Comm_InsertV2X(comm_log_lines)
    comm_send_count = Count_Comm_SendV2X(comm_log_lines)
    comm_v2x_healthy = comm_insert_count > 0 and comm_send_count > 0
    matched_comm_ids = Compare_Carma_And_Comm_Message_IDs(
        carma_log_lines,
        comm_log_lines,
        vehicle_name
    )
    delayed_v2x_count = Compare_Carma_And_Comm_Message_ID_Delays(
        carma_log_lines,
        comm_log_lines,
        vehicle_name,
        delay_threshold_sec=0.1
    )
    comm_non_simulated_count = count_matching_lines(comm_log_lines, r"non-?simulated")
    write_analysis(f"communicationdetails non-simulated node warnings: {comm_non_simulated_count}")
    comm_duplicate_vehicle_count = count_matching_lines(comm_log_lines, r"duplicate.*vehicle|vehicle.*duplicate")
    write_analysis(f"communicationdetails duplicate vehicle warnings: {comm_duplicate_vehicle_count}")
    sumo_connected = Check_Sumo_Connected(traffic_log_lines)
    sumo_retry_count = Count_Sumo_Retries(traffic_log_lines)
    sumo_api_logged = Check_Sumo_Api_Logged(traffic_log_lines)
    sumo_sim_started = Check_Sumo_Simulation_Time_Started(traffic_log_lines)
    sumo_vehicleupdates_count = Count_Sumo_VehicleUpdates_Interactions(traffic_log_lines)
    sumo_assignment_count = Count_Sumo_VehicleFederateAssignment_Interactions(traffic_log_lines)
    sumo_ignored_external_count = Count_Sumo_Ignored_External(traffic_log_lines)
    sumo_ignored_vehicle_count = Count_Sumo_Ignored_External(traffic_log_lines, vehicle_name)
    sumo_ignored_carma_source_count = Count_Sumo_Ignored_External(traffic_log_lines, "carma")
    sumo_ignored_msger_source_count = Count_Sumo_Ignored_External(
        traffic_log_lines,
        "carma-messenger|msger",
    )
    sumo_missing_assignment_issue = (
        sumo_ignored_external_count > 0 and sumo_assignment_count == 0
    )
    federation_started = Check_Federation_Started(mosaic_log_lines)
    initialized_federates = Count_Initialized_Federates(mosaic_log_lines)
    added_federates = Count_Added_Federates(mosaic_log_lines)
    no_mapping_spawners = Check_No_Mapping_Spawners(mosaic_log_lines)
    common_registration_count = Count_Common_Instance_Registrations(mosaic_log_lines, vehicle_name)
    common_registration_spam = common_registration_count > 1
    v2x_receiver_count = Count_V2X_Receiver_Starts(mosaic_log_lines)

    Write_Summary(
        vehicle_name,
        cdas2_result,
        cdas3_result,
        cdas5_result,
        cdas6_result,
        cdas7_result,
        cdas15_result,
        cdas16_result,
        cdas17_result,
        cdas18_result,
        spawn_ok,
        xmlrpc_ok,
        xmlrpc_false_count,
        xmlrpc_true_count,
        xmlrpc_recovered,
        actor_not_connected_count,
        vehicleupdates_count,
        total_added_vehicles,
        sync_start_count,
        successful_actor_updates,
        existing_mapping_skips,
        add_without_sim_count,
        update_without_sim_count,
        sim_unit_problem,
        carma_registered,
        time_sync_ok,
        time_sync_count,
        v2x_ok,
        v2x_count,
        duplicate_registration_count,
        v2x_paired_count,
        comm_insert_count,
        comm_send_count,
        comm_v2x_healthy,
        len(matched_comm_ids),
        delayed_v2x_count,
        comm_non_simulated_count,
        comm_duplicate_vehicle_count,
        sumo_connected,
        sumo_retry_count,
        sumo_api_logged,
        sumo_sim_started,
        sumo_vehicleupdates_count,
        sumo_assignment_count,
        sumo_ignored_external_count,
        sumo_ignored_vehicle_count,
        sumo_ignored_carma_source_count,
        sumo_ignored_msger_source_count,
        sumo_missing_assignment_issue,
        federation_started,
        initialized_federates,
        added_federates,
        no_mapping_spawners,
        common_registration_count,
        common_registration_spam,
        v2x_receiver_count,
    )

    Write_Root_Cause_Hints(
        vehicle_name,
        spawn_ok,
        xmlrpc_ok,
        xmlrpc_recovered,
        actor_not_connected_count,
        sim_unit_problem,
        add_without_sim_count,
        update_without_sim_count,
        sync_start_count,
        successful_actor_updates,
        time_sync_ok,
        v2x_ok,
        duplicate_registration_count,
        comm_v2x_healthy,
        sumo_retry_count,
        sumo_connected,
        sumo_missing_assignment_issue,
        sumo_ignored_vehicle_count,
        sumo_ignored_msger_source_count,
        no_mapping_spawners,
        common_registration_spam,
        initialized_federates,
        added_federates,
    )

    print(f"Analysis written to: {ANALYSIS_LOG}")


if __name__ == "__main__":
    main()
