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
``portable.tcpdump_text``).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# runs.csv wall-clock times. Fixed offset rather than a tz database lookup so the
# module stays dependency-free; sessions run inside a single afternoon, well away
# from a DST transition.
SESSION_TZ = timezone(timedelta(hours=-4), name="America/New_York (EDT)")

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

    condition: str          # "5sec" | "10sec" | "15sec"
    run_id: int             # 1..5 within the condition
    start_time: datetime    # runs.csv start_time_etc, tz-aware
    rsu_pcap: Path          # SDSM broadcasts, binary pcap
    obu_capture: Path       # OBU radio traffic, tcpdump text
    mcap: Path              # CARMA Platform rosbag

    @property
    def name(self) -> str:
        """Stable identifier used for this run's output directory."""
        return f"{self.condition}_run{self.run_id}"

    @property
    def dwell_sec(self) -> int:
        return condition_dwell_sec(self.condition)

    def missing(self) -> List[str]:
        return [
            f"{field}={path}"
            for field, path in (
                ("rsu_pcap", self.rsu_pcap),
                ("obu_capture", self.obu_capture),
                ("mcap", self.mcap),
            )
            if not path.is_file()
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


def discover_session(data_root) -> SessionPaths:
    """Locate the session-wide logs under ``data_root``.

    These are found by pattern because their names carry a session date that
    varies; the per-run files are not, because ``runs.csv`` names them exactly.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Data root does not exist: {root}")
    return SessionPaths(
        root=root,
        pc2_v2xhub=_find(root, "v2xhub_pc2_*.log"),
        pc1_v2xhub=_find(root, "v2xhub_pc1_*.log"),
        sdss=_find(root, "sdss_*.log"),
        kafka_detected_object=_find(root, "v2xhub_sim_sensor_detected_object*.log"),
        kafka_sdsm=_find(root, "v2xhub_sdsm_sub*.log"),
    )


def _resolve(root: Path, subdir: str, filename: str) -> Path:
    """Locate one run file, preferring ``subdir`` but searching the tree as a fallback."""
    direct = root / subdir / filename
    if direct.is_file():
        return direct
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    return matches[0] if matches else direct


def load_runs_csv(runs_csv, data_root=None) -> List[RunSpec]:
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
            start = datetime.strptime(clean["start_time_etc"], "%Y-%m-%d %H:%M:%S")
            runs.append(
                RunSpec(
                    condition=clean["run_condition"],
                    run_id=int(clean["run_id"]),
                    start_time=start.replace(tzinfo=SESSION_TZ),
                    rsu_pcap=_resolve(root, "rsu_pcap", clean["rsu_pcap_fn"]),
                    obu_capture=_resolve(root, "obu", clean["obu_pcap_fn"]),
                    mcap=_resolve(root, "rosbags", clean["rosbag_fn"]),
                )
            )

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
