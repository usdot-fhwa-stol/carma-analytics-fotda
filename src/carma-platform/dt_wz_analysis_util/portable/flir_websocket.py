"""Read the FLIR camera's websocket stream out of the pc2 V2XHub log.

This is the closest measurement point to the camera itself. Every message the
camera sends is logged here verbatim, before the plugin parses it, queues it or
produces it to Kafka, so a count taken here is a property of the camera and its
link and of nothing downstream.

Each message carries:

``dataNumber``
    The camera's own frame counter. It advances on **every** camera frame, but
    the camera only transmits a message when it has at least one track. So a
    counter gap means the camera produced frames in which it detected nobody,
    and during a pedestrian's dwell that is exactly a missed detection. Between
    runs the same gap means only that nobody was there, so a gap is meaningful
    only inside a known dwell window.

``time``
    The camera's own capture time, as an ISO timestamp with an offset. This is
    the same clock the Kafka detection records report, so the two sources can be
    compared frame for frame.

``track``
    The detected objects in that frame. One message can carry several, so a
    message is one camera frame, not one detection.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List

# The plugin logs each received websocket message at this source line. An anchor
# rather than a text match: the surrounding wording has changed between builds
# while the line has stayed put.
RECEIVE_ANCHOR = "WebsockAsyncClnSession.cpp (139)"

_TRAILING_JSON = re.compile(r"\{.*\}\s*$")


def parse_camera_frames(log_path) -> List[Dict]:
    """Every ``Data`` message the camera sent, in camera-time order.

    Returns ``{camera_time_ms, data_number, track_count}`` dicts. Messages that
    are not ``Data`` (subscription acknowledgements and the like) are skipped, as
    are any whose JSON will not decode.
    """
    frames: List[Dict] = []
    with open(log_path, encoding="utf8", errors="ignore") as handle:
        for line in handle:
            if RECEIVE_ANCHOR not in line:
                continue
            match = _TRAILING_JSON.search(line)
            if match is None:
                continue
            try:
                message = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if message.get("messageType") != "Data":
                continue
            try:
                camera_time = datetime.fromisoformat(message["time"]).timestamp() * 1000.0
                data_number = int(message["dataNumber"])
            except (KeyError, ValueError, TypeError):
                continue
            frames.append({
                "camera_time_ms": camera_time,
                "data_number": data_number,
                "track_count": len(message.get("track", [])),
            })
    frames.sort(key=lambda frame: frame["camera_time_ms"])
    return frames


def frames_in_window(frames: List[Dict], start_ms: float, end_ms: float) -> List[Dict]:
    """Frames whose camera time falls in ``[start_ms, end_ms)``."""
    return [frame for frame in frames if start_ms <= frame["camera_time_ms"] < end_ms]


def counter_gaps(frames: List[Dict]) -> int:
    """Camera frames the counter accounts for but that carried no message.

    Inside a dwell this is the number of frames in which the camera saw nobody,
    which is the camera's own missed detections. It is independent of the frame
    rate, because it compares the counter against itself rather than against an
    assumed 10 Hz.
    """
    if len(frames) < 2:
        return 0
    numbers = [frame["data_number"] for frame in frames]
    return int(numbers[-1] - numbers[0] + 1 - len(numbers))
