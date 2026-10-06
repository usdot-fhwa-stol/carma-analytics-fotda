"""Timestamp normalisation. Everything in this project is epoch milliseconds UTC."""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta, timezone

# "[2026-09-11 16:48:36.635] ..."
_BRACKET_TS = re.compile(r"^\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})\]")


def parse_bracket_ts(line: str, utc_offset_h: int = 0) -> float | None:
    """Epoch ms for a leading ``[YYYY-MM-DD HH:MM:SS.mmm]`` stamp.

    ``utc_offset_h`` is the offset of the *log's* clock from UTC, so EDT logs
    pass ``-4``. Returns None when the line does not start with a stamp.
    """
    m = _BRACKET_TS.match(line)
    if m is None:
        return None
    y, mo, d, h, mi, s, ms = (int(g) for g in m.groups())
    dt = datetime(y, mo, d, h, mi, s, tzinfo=timezone(timedelta(hours=utc_offset_h)))
    return dt.timestamp() * 1000.0 + ms


def sdsm_timestamp_to_epoch_ms(ts: dict) -> float:
    """Epoch ms for a J3224 ``sdsm_time_stamp`` object.

    The DSecond field (``second``) is *milliseconds within the minute*, not
    seconds; ``offset`` is minutes from UTC.
    """
    ms_in_minute = int(ts["second"])
    base = calendar.timegm(
        (int(ts["year"]), int(ts["month"]), int(ts["day"]), int(ts["hour"]), int(ts["minute"]), 0, 0, 0, 0)
    )
    epoch_ms = base * 1000.0 + ms_in_minute
    return epoch_ms - int(ts.get("offset", 0)) * 60_000.0


def ros_time_to_epoch_ms(sec: int, nanosec: int) -> float:
    return sec * 1000.0 + nanosec / 1e6


def fmt_ms(epoch_ms: float | None) -> str:
    """Human-readable UTC rendering, for QA output."""
    if epoch_ms is None:
        return "-"
    dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(epoch_ms % 1000):03d}"
