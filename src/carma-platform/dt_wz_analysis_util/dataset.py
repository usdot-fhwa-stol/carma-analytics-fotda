"""Dataset description for a DT-WZ verification session, driven by ``runs.csv``.

A session directory looks like this::

    20260914_verification_test/
        runs.csv
        rsu_pcap/      capture_<date>_<cond>_run<N>[_take2].pcap   binary pcap
        obu/  <cond>-run<N>.pcap                      tcpdump TEXT
        rosbags/       [recovered_]rosbag2_<cond>_run<N>.mcap
        pc1/           v2xhub_pc1_<date>.log
        pc2/           v2xhub_pc2_<date>.log, sdss_<date>.log,
                       kafka_topics_<date>/*.log

``runs.csv`` is the authority on which files constitute a run, and globbing is
not an acceptable substitute. The 2026-09-14 session holds 20 MCAPs and 19 RSU
pcaps for 15 real runs: the 5-second condition was run twice and only the
``_take2`` captures and ``run7..run11`` bags are valid, and each bag exists in
both a truncated ``rosbag2_*`` and a readable ``recovered_rosbag2_*`` copy. A
glob picks up the abandoned first attempt and the unreadable bags alongside the
good ones, with nothing in the data to mark which is which.

Times in ``runs.csv`` are America/New_York wall clock. The OBU captures are UTC
and carry no date, so the run's date is taken from here (see
``readers.tcpdump_text``).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from . import config as cfg

# runs.csv wall-clock times. Defined once in ``config``; re-exported here
# because this module is where the manifest is parsed.
SESSION_TZ = cfg.SESSION_TZ

# Seconds the pedestrian dwells in the detection zone, read out of the
# run_condition name ("20sec" -> 20). Parsed rather than tabulated so a session
# introducing a new dwell -- 2026-09-15 added 20sec -- needs no code change, and
# so a condition can never silently fall back to a dwell of zero and sort first.
_CONDITION_SEC = re.compile(r"(\d+)\s*s", re.IGNORECASE)


def condition_dwell_sec(condition: str) -> int:
    """Dwell seconds for a run_condition name, or 0 if it carries no number."""
    match = _CONDITION_SEC.search(condition or "")
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class RunSpec:
    """One demo run and the four files that record it."""

    condition: str                    # "5sec", "20sec", "pl-01", ...
    run_id: int                       # position within the condition
    obu_capture: Path                 # OBU radio traffic, binary pcap or tcpdump text
    mcap: Path                        # CARMA Platform rosbag
    # Optional, because not every session records them. A session dedicated to
    # message rates has no RSU capture and needs no wall-clock start: its runs
    # are found from the recording itself.
    start_time: Optional[datetime] = None
    rsu_pcap: Optional[Path] = None
    # The next row's start time. runs.csv lists runs in chronological order, so
    # a trial can extend no further than the moment the next one began. None
    # for the last row, and for sessions that record no start times.
    end_time: Optional[datetime] = None

    @property
    def name(self) -> str:
        """Stable identifier used for this run's output directory."""
        return f"{self.condition}_run{self.run_id}"

    @property
    def dwell_sec(self) -> int:
        return condition_dwell_sec(self.condition)

    def missing(self) -> List[str]:
        """Files named in runs.csv that are not on disk.

        A column the manifest omits is not missing -- it was never claimed. Only
        files the manifest actually names are checked.
        """
        return [
            f"{field}={path}"
            for field, path in (
                ("rsu_pcap", self.rsu_pcap),
                ("obu_capture", self.obu_capture),
                ("mcap", self.mcap),
            )
            if path is not None and not path.is_file()
        ]


@dataclass(frozen=True)
class SessionPaths:
    """The continuous, session-wide logs that span every run."""

    root: Path
    pc2_v2xhub: Optional[Path]
    pc1_v2xhub: Optional[Path]
    sdss: Optional[Path]
    kafka_detected_object: Optional[Path]
    kafka_sdsm: Optional[Path]

    def describe(self) -> str:
        lines = [f"session root: {self.root}"]
        for field in ("pc2_v2xhub", "pc1_v2xhub", "sdss", "kafka_detected_object", "kafka_sdsm"):
            path = getattr(self, field)
            lines.append(f"  {field:<22} {path if path else '-- not found --'}")
        return "\n".join(lines)


def _find(root: Path, *patterns: str) -> Optional[Path]:
    """First file matching any pattern, searched recursively, in sorted order."""
    for pattern in patterns:
        matches = sorted(path for path in root.rglob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def discover_session(data_root, layout: cfg.SessionLayout = cfg.LAYOUT) -> SessionPaths:
    """Locate the session-wide logs under ``data_root``.

    These are found by pattern because their names carry a session date that
    varies; the per-run files are not, because ``runs.csv`` names them exactly.
    The patterns are a property of the deployment, so they come from
    ``config.SessionLayout`` rather than from this function.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Data root does not exist: {root}")
    found = {field: _find(root, pattern) for pattern, field in layout.session_logs}
    return SessionPaths(root=root, **found)


def _resolve(root: Path, subdir: str, filename: str) -> Path:
    """Locate one run file, preferring ``subdir`` but searching the tree as a fallback."""
    direct = root / subdir / filename
    if direct.is_file():
        return direct
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    return matches[0] if matches else direct


def session_manifest(data_root, layout: cfg.SessionLayout = cfg.LAYOUT) -> Path:
    """The run manifest inside a session directory.

    Every entry point needs this path, so it is built here rather than by each
    script joining the filename itself.
    """
    return Path(data_root) / layout.manifest


def load_session(data_root, layout: cfg.SessionLayout = cfg.LAYOUT):
    """``(runs, session)`` for one session directory: the usual first step."""
    root = Path(data_root)
    return (load_runs_csv(session_manifest(root, layout), root, layout),
            discover_session(root, layout))


def load_runs_csv(runs_csv, data_root=None,
                  layout: cfg.SessionLayout = cfg.LAYOUT) -> List[RunSpec]:
    """Parse ``runs.csv`` into RunSpecs, resolving each named file to a real path.

    The header and every field are whitespace-stripped: the file is written with
    ``", "`` separators, so unstripped keys arrive as ``" run_id"`` and values
    carry a leading space.

    Raises FileNotFoundError listing every missing file at once, rather than
    failing on the first, so a broken session is diagnosed in one pass.
    """
    runs_csv = Path(runs_csv)
    if not runs_csv.is_file():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv}")
    root = Path(data_root) if data_root is not None else runs_csv.parent

    runs: List[RunSpec] = []
    with open(runs_csv, newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            clean = {
                (key or "").strip(): (value or "").strip()
                for key, value in row.items()
            }
            if not clean.get("run_condition"):
                continue
            start = None
            if clean.get("start_time_etc"):
                start = datetime.strptime(
                    clean["start_time_etc"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=cfg.SESSION_TZ)
            runs.append(
                RunSpec(
                    condition=clean["run_condition"],
                    run_id=int(clean["run_id"]),
                    start_time=start,
                    rsu_pcap=(
                        _resolve(root, layout.rsu_pcap_dir, clean["rsu_pcap_fn"])
                        if clean.get("rsu_pcap_fn") else None
                    ),
                    obu_capture=_resolve(root, layout.obu_capture_dir, clean["obu_pcap_fn"]),
                    mcap=_resolve(root, layout.rosbag_dir, clean["rosbag_fn"]),
                )
            )

    # Rows are in chronological order, which bounds each trial by the next
    # row's start. Checked rather than assumed: a mistyped date -- 2026-09-23
    # runs once entered as 2026-09-14 -- would otherwise shift every window.
    timed = [run for run in runs if run.start_time is not None]
    for earlier, later in zip(timed, timed[1:]):
        if later.start_time <= earlier.start_time:
            raise ValueError(
                f"{runs_csv}: rows must be in chronological order, but "
                f"{later.name} ({later.start_time:%Y-%m-%d %H:%M:%S}) does not start "
                f"after {earlier.name} ({earlier.start_time:%Y-%m-%d %H:%M:%S})")
    for index, run in enumerate(runs[:-1]):
        if run.start_time is not None and runs[index + 1].start_time is not None:
            runs[index] = replace(run, end_time=runs[index + 1].start_time)

    if not runs:
        raise ValueError(f"No runs parsed from {runs_csv}")

    missing = [f"{run.name}: {item}" for run in runs for item in run.missing()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} file(s) named in {runs_csv} are missing:\n  "
            + "\n  ".join(missing)
        )
    return runs


def group_by_condition(runs: List[RunSpec]) -> Dict[str, List[RunSpec]]:
    """Runs bucketed by dwell condition, each bucket ordered by run_id."""
    grouped: Dict[str, List[RunSpec]] = {}
    for run in runs:
        grouped.setdefault(run.condition, []).append(run)
    for bucket in grouped.values():
        bucket.sort(key=lambda run: run.run_id)
    return dict(sorted(grouped.items(), key=lambda item: condition_dwell_sec(item[0])))


def condition_order(runs: List[RunSpec]) -> List[str]:
    """Condition names in ascending dwell order, for stable plot and table ordering."""
    return list(group_by_condition(runs).keys())
