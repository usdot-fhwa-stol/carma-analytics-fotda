"""The test catalogue: what is measured, against what, and where it is written.

The modules beside this one are the measurement API. They must work for any
session, any site and any acceptance limit, so they hold no numbers and no test
identifiers of their own: a function that needs a threshold, a file name or a
plot title takes it as an argument. This file holds all of it, and
``run_dt_wz_analysis.py`` passes it in.

The split is deliberate. Before it, the 2% drop limit appeared in six places,
the session timezone in four, and the camera's 10 Hz in five -- including one
copy that made ``--rate-hz`` only half work. Re-targeting the suite at a new
site or a new acceptance limit meant editing measurement code, and a limit
printed in a report could differ from the limit actually applied.

**Three layers.**

``SiteGeometry``, ``VehicleTopics``, ``SessionLayout``, ``PlotStyle``
    What this deployment looks like. Shared by every test.

``*Config``
    One frozen dataclass per measurement, holding its limits and tunables. The
    class is named for what it measures, never for the test code that happens
    to ask for it, because the same measurement can serve more than one test.

``TESTS``
    The catalogue. One ``TestCase`` per entry in the test plan, naming the
    measurement it runs, the settings it runs with, its acceptance criterion in
    words, and the files it writes. This is the only place a test code such as
    ``CP-01`` appears.

**The command line is generated from all this.** Every scalar field of a
settings class with a ``cli_help`` entry becomes an option, so adding a tunable
here adds the flag that overrides it. The value the measurement uses and the
value the report states are then the same object.

The dataclasses are frozen, so a new configuration is made with
``dataclasses.replace`` rather than by assignment. That keeps a value that was
used for a result from changing after the fact.

**What does not live here.** Anything fixed by a standard or by a message
definition is a property of the protocol, not of the test. J2735 wire units,
the ROS topic names of standard messages and the cascade's stage table stay
with the code that decodes them. The rule is: if a different site or a revised
test plan could change it, it belongs here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import timedelta, timezone
from typing import ClassVar, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# The session: clocks and file layout
# ---------------------------------------------------------------------------

# Wall-clock offset of the test site. ``runs.csv`` times, the burst times the
# camera report prints for a human to check, and the SDSS log are all on this
# clock, while V2XHub logs UTC. A fixed offset rather than a tz-database lookup
# keeps the package dependency-free, and a session runs inside one afternoon,
# well away from a DST transition.
SESSION_UTC_OFFSET_H = -4
SESSION_TZ = timezone(timedelta(hours=SESSION_UTC_OFFSET_H),
                      name="America/New_York (EDT)")

# V2XHub writes its bracketed wall clock in UTC; the SDSS writes local time.
V2XHUB_UTC_OFFSET_H = 0
SDSS_UTC_OFFSET_H = SESSION_UTC_OFFSET_H


@dataclass(frozen=True)
class SessionLayout:
    """Where a session directory keeps each kind of file.

    ``runs.csv`` names the per-run files, so only their parent directories are
    configured. The session-wide logs carry a date in their names, so they are
    found by glob instead.
    """

    manifest: str = "runs.csv"
    rsu_pcap_dir: str = "rsu_pcap"
    obu_capture_dir: str = "obu"
    rosbag_dir: str = "rosbags"
    # glob -> the SessionPaths field it fills.
    session_logs: Tuple[Tuple[str, str], ...] = (
        ("v2xhub_pc2_*.log", "pc2_v2xhub"),
        ("v2xhub_pc1_*.log", "pc1_v2xhub"),
        ("sdss_*.log", "sdss"),
        ("v2xhub_sim_sensor_detected_object*.log", "kafka_detected_object"),
        ("v2xhub_sdsm_sub*.log", "kafka_sdsm"),
    )


LAYOUT = SessionLayout()


# ---------------------------------------------------------------------------
# The site: fixed ground truth for the 2026-09 work-zone demo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteGeometry:
    """The trial route, in WGS84. Survey values, not measurements."""

    # Where the vehicle begins each run, and the origin of every local ENU
    # coordinate the analysis reports.
    start_latlon: Tuple[float, float] = (38.95503211512239, -77.14757752506146)
    # Where the run ends, past the pedestrian crossing.
    end_latlon: Tuple[float, float] = (38.95497471661251, -77.1491050181905)


SITE = SiteGeometry()


# ---------------------------------------------------------------------------
# The vehicle: which topics carry what
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleTopics:
    """Recorded topics the analysis reads, named per deployment.

    The standard message topics (``/message/incoming_sdsm`` and the rest) are
    fixed by V2XHub and stay with the code. These two are hardware-interface
    topics whose names depend on which sensors the vehicle is fitted with.
    """

    gps_fix: str = "/hardware_interface/novatel/oem7/fix"
    twist: str = "/hardware_interface/vehicle/twist"


TOPICS = VehicleTopics()


# ---------------------------------------------------------------------------
# Measurement settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraDetectionConfig:
    """Camera frame accounting over each pedestrian dwell."""

    #: One line per tunable, for the generated command line. The reasoning
    #: stays in the comments below, which the ``--help`` text points at.
    cli_help: ClassVar[Dict[str, str]] = {
        "detection_rate_hz": "nominal camera frame rate",
        "burst_gap_ms": "gap that separates one detection burst from the next",
        "gap_report_factor": "frame periods a gap must exceed to be reported",
        "search_before_ms": "how far before the nominal start to look for frames",
        "search_after_ms": "how far after the nominal start to look for frames",
        "stall_lag_ms": "capture-to-arrival lag above which a frame counts as stalled",
    }

    # Nominal FLIR frame rate. 100 ms cadence, confirmed against the websocket
    # logs: the median inter-arrival is 100.0 ms and about 90% of intervals fall
    # between 90 and 110 ms.
    detection_rate_hz: float = 10.0

    # Detections further apart than this start a new burst. Chosen from the
    # data: the largest gap observed *inside* a dwell is 1.5 s, and the smallest
    # pause separating a false start from the real dwell is 2.9 s. 2 s sits
    # between them, so a dwell containing dropped frames stays one burst while a
    # false start stays separate. Below about 1.5 s a dwell would split at its
    # own drops, which is the very thing being measured.
    burst_gap_ms: float = 2000.0

    # A gap wider than this many frame periods is reported as a stall span.
    gap_report_factor: float = 1.5

    # How far either side of the nominal start time to look for a run's
    # detections. The search also stops at the next row's start in runs.csv
    # (RunSpec.end_time), because runs can be closer together than this: 2-4
    # minutes on 2026-09-23.
    search_before_ms: float = 60_000.0
    search_after_ms: float = 240_000.0

    # A frame whose arrival at V2XHub trails its capture by more than this is
    # held back by a link stall. Normal lag is a few milliseconds (4 ms median,
    # 11 ms p95), and the stalls found so far hold frames back for over 3 s, so
    # the threshold sits far from both.
    stall_lag_ms: float = 500.0

    @property
    def frame_interval_ms(self) -> float:
        """One frame period. Derived, so changing the rate really changes it."""
        return 1000.0 / self.detection_rate_hz


CAMERA_DETECTIONS = CameraDetectionConfig()


@dataclass(frozen=True)
class DetectionDeliveryConfig:
    """Detections that never reached the vehicle inside an SDSM."""

    cli_help: ClassVar[Dict[str, str]] = {
        "max_drop_rate_pct": "per-run acceptance limit on the drop rate",
        "match_tolerance_sec": "how close two detections must be to be the same one",
    }

    # Acceptance limit, per run. Carried over from
    # carma_cooperative_perception_scripts so results stay comparable with
    # earlier sessions.
    max_drop_rate_pct: float = 2.0

    # How close a reported detection's reconstructed observation time must be to
    # a raw one before the two are the same detection.
    match_tolerance_sec: float = 0.05


DETECTION_DELIVERY = DetectionDeliveryConfig()


@dataclass(frozen=True)
class BroadcastDeliveryConfig:
    """Broadcasts the vehicle never received, matched byte for byte."""

    cli_help: ClassVar[Dict[str, str]] = {
        "max_drop_rate_pct": "per-run acceptance limit on the drop rate",
        "late_threshold_ms": "above this a matched pair counts as late",
    }

    max_drop_rate_pct: float = 2.0

    # Above this |rx - tx| a matched pair is counted as late rather than on time.
    late_threshold_ms: float = 200.0


BROADCAST_DELIVERY = BroadcastDeliveryConfig()


@dataclass(frozen=True)
class DeliveryLatencyConfig:
    """End-to-end latency the vehicle actually experiences."""

    cli_help: ClassVar[Dict[str, str]] = {
        "max_median_latency_sec": "per-run acceptance limit on the median latency",
    }

    # Per-run acceptance limit on the median.
    max_median_latency_sec: float = 0.3
    unit: str = "s"


DELIVERY_LATENCY = DeliveryLatencyConfig()


# How a topic's rate is averaged.
#
#   None           continuous; average over the whole engaged window.
#   "detections"   flows only while an object is detected, so the periods come
#                  from the raw detection log. SDSM is in this position:
#                  averaged over a 32 s window containing 6 s of pedestrian it
#                  reports about 1.8 Hz for a stream running at its nominal
#                  10 Hz whenever it runs at all.
#   "self"         stops for an expected reason, so the periods come from its
#                  own arrivals. MOM stops when the vehicle drives out of range
#                  of the source, which is behaviour, not a rate fault.
GATING_CONTINUOUS = None
GATING_DETECTIONS = "detections"
GATING_SELF = "self"


@dataclass(frozen=True)
class TopicRate:
    """One topic's expected rate and how to average it."""

    label: str
    expected_rate_hz: float
    gating: Optional[str] = GATING_CONTINUOUS


@dataclass(frozen=True)
class MessageRateConfig:
    """Per-topic rates, split by which session carries which message.

    No single session carries every message type: MAP and SPAT were not
    broadcast during the verification runs, and the session recorded to exercise
    them carries no SDSM. So the topics are in two groups and each group is
    measured on a session that actually contains it.
    """

    cli_help: ClassVar[Dict[str, str]] = {
        "rate_tolerance_pct": "fraction of the expected rate a topic may deviate by",
        "out_of_range_gap_sec": "MOM gap that means the vehicle left the source's range",
        "detection_gap_sec": "gap that separates one detection period from the next",
    }

    # Measured on the primary sessions.
    primary_topics: Dict[str, TopicRate] = field(default_factory=lambda: {
        "/message/incoming_map": TopicRate("MAP", 1.0),
        "/message/incoming_spat": TopicRate("SPAT", 10.0),
        "/message/incoming_mobility_operation": TopicRate("MOM", 1.0, GATING_SELF),
        "/message/bsm_outbound": TopicRate("BSM", 10.0),
    })

    # Measured on the secondary sessions, when a second group is given.
    secondary_topics: Dict[str, TopicRate] = field(default_factory=lambda: {
        "/message/incoming_sdsm": TopicRate("SDSM", 10.0, GATING_DETECTIONS),
    })

    # A measured rate passes inside this fraction of its expected rate.
    rate_tolerance_pct: float = 0.2

    # A MOM gap longer than this means the vehicle left the source's range
    # rather than a message being missed. The nominal interval is 1 s.
    out_of_range_gap_sec: float = 2.0

    # Camera detections further apart than this start a new detection period,
    # which is what a detection-gated rate is averaged over.
    detection_gap_sec: float = 0.5

    @property
    def all_topics(self) -> Dict[str, TopicRate]:
        return {**self.primary_topics, **self.secondary_topics}


MESSAGE_RATES = MessageRateConfig()


@dataclass(frozen=True)
class VehicleYieldConfig:
    """Whether the vehicle stopped short of the pedestrian it was warned about."""

    cli_help: ClassVar[Dict[str, str]] = {
        "target_success_pct": "share of valid runs that must yield",
        "start_tolerance_m": "how near the start point the vehicle must pass",
        "end_tolerance_m": "how near the end point the vehicle must pass",
        "stop_speed_mps": "speed below which the vehicle counts as halted",
        "min_stop_sec": "how long a halt must last to count",
        "moved_from_start_m": "distance travelled before a halt is not the parked start",
        "heading_lookback": "samples used to estimate the heading at a halt",
        "yield_tolerance_m": "how far past the pedestrian a halt may still yield",
        "yield_max_distance_m": "beyond this the halt was for something else",
        "detection_gap_limit_sec": "camera gap that invalidates a run",
        "plugin_loss_limit_pct": "plugin-side frame loss that invalidates a run",
        "sdsm_drop_limit_pct": "SDSM drop rate that invalidates a run",
        "profile_step_m": "resolution of the distance grid the runs are averaged on",
    }

    # The share of *valid* runs that must yield.
    target_success_pct: float = 90.0

    # -- route fences ------------------------------------------------------
    # Checked inside the engaged window, and looser than the nominal 5 m because
    # the vehicle can creep forward before guidance engages: the measured start
    # gap reaches 8.8 m.
    start_tolerance_m: float = 10.0

    # Checked over the **whole recording**, not the engaged window. CARMA
    # routinely disengages once it is past the pedestrian, leaving up to 37 m of
    # the route undriven under guidance. Gating on the engaged window would fail
    # 19 of 30 runs for behaving normally.
    #
    # Relaxed from 5 m: the fence exists to confirm the vehicle drove the
    # intended route, not to measure where it came to rest. 38 of 41 runs stop
    # within 0.7 m of the end point, but where guidance hands back early the
    # recording can end before the vehicle rolls the last few metres -- the
    # worst two reach 8.2 m and 8.4 m, which on a 132 m route is still plainly
    # the right route. 10 m clears those and matches the start fence, which was
    # relaxed for the same reason.
    end_tolerance_m: float = 10.0

    # -- finding a halt ----------------------------------------------------
    stop_speed_mps: float = 0.2
    min_stop_sec: float = 0.5

    # How far the vehicle must have travelled before a low-speed period counts
    # as a stop rather than the stationary start it always begins from. Distinct
    # from the two fences above and from the yield tolerance below, although all
    # three currently read 5 m.
    moved_from_start_m: float = 5.0

    # Samples used to estimate the vehicle's heading at a stop. At about 50 Hz
    # this looks back roughly 0.4 s, long enough to outrun GPS jitter and short
    # enough to reflect the direction the vehicle was actually travelling.
    heading_lookback: int = 20

    # -- scoring a halt as a yield ----------------------------------------
    # How far past the pedestrian's crossing point a stop may sit and still
    # count as a yield. The crossing point is a median of noisy SDSM reports and
    # GPS has its own error, so a metre or two past it is measurement noise, not
    # an overshoot.
    yield_tolerance_m: float = 5.0

    # Beyond this the vehicle stopped for something other than the pedestrian.
    yield_max_distance_m: float = 30.0

    # -- validity ----------------------------------------------------------
    # Criterion [1]: a detection burst containing a gap this long means the
    # camera stalled and the run cannot test vehicle behaviour.
    detection_gap_limit_sec: float = 1.0

    # Criterion [1] continued: frames lost between the camera and Kafka.
    # Proportional rather than any-loss. A single frame missing from a
    # ~200-frame run is not "inconsistent detection"; the plugin faults worth
    # excluding lost 5-6% of the run.
    plugin_loss_limit_pct: float = 2.0

    # Criterion [3]: SDSMs must have been consistently received.
    sdsm_drop_limit_pct: float = 2.0

    # -- outputs -----------------------------------------------------------
    # Resolution of the common distance grid the speed profiles are averaged on.
    profile_step_m: float = 0.5

    # Cached trajectory geometry, so the figures can be redrawn without
    # re-reading the recordings.
    cache_name: str = "pl03_tracks.npz"
    trajectory_plot: str = "pl03_trajectories.png"
    speed_profile_plot: str = "pl03_speed_profile.png"


VEHICLE_YIELD = VehicleYieldConfig()


@dataclass(frozen=True)
class LocationSpoofingConfig:
    """That each SDSM places the object at the configured remote reference.

    The acceptance limits themselves live in
    ``src/carma-streets/sdsm_location_spoofing_verification.py`` and are read
    from there, so the two tools cannot disagree about what passes. Only the
    windowing is a property of these sessions.
    """

    cli_help: ClassVar[Dict[str, str]] = {
        "window_margin_ms": "margin added around the session's runs before windowing",
    }

    # Margin added around the session's runs before windowing the Kafka dumps.
    # A dump holds the broker's whole retention; unwindowed, the check would
    # verify every detection back to 2026-09-09 and report a figure for several
    # days of testing rather than for the session asked about.
    window_margin_ms: float = 120_000.0


LOCATION_SPOOFING = LocationSpoofingConfig()


@dataclass(frozen=True)
class LatencyCascadeConfig:
    """Tuning for the per-detection stage table.

    The stage list itself is in ``cascade/cascade_config.py``: it describes the
    deployed pipeline rather than this test, and changing it changes what the
    parsers look for.
    """

    cli_help: ClassVar[Dict[str, str]] = {
        "fused_match_window_ms": "how late a fused track may be and still match",
        "fused_match_radius_m": "how far a fused track may sit and still match",
        "max_cascade_rows": "rows past which the cascade figure is thinned",
    }

    # Tolerance when matching a fused track back to the detection that produced
    # it. The radius is applied only where the fused track carries a position.
    fused_match_window_ms: float = 600.0
    fused_match_radius_m: float = 15.0

    # Rows are thinned past this count to keep the cascade figure a readable
    # shape.
    max_cascade_rows: int = 420

    # Drop-rate columns of ``detections.csv`` worth plotting together, as
    # (column, label). Named here because they are this suite's column names,
    # not a property of the cascade.
    drop_rate_columns: Tuple[Tuple[str, str], ...] = (
        ("cp02_drop_rate_pct", "Detection → SDSM"),
        ("cp03_drop_rate_pct", "RSU → vehicle"),
    )


LATENCY_CASCADE = LatencyCascadeConfig()


# ---------------------------------------------------------------------------
# The test catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestCase:
    """One entry in the test plan.

    The only place a test code appears. A measurement module is named for what
    it measures and knows nothing about which test asked for it, so the code,
    the human title and the output file names all arrive from here.
    """

    code: str          # "cp01": the CLI word and the output sub-directory
    metric: str        # "CP-01", as it appears in the report
    analysis: str      # which measurement to run; the runner maps it to a handler
    summary: str       # one line: what this test measures
    criterion: str     # the acceptance rule, in words
    settings: object   # the frozen settings the command line is generated from
    prefix: Optional[str] = None   # output file stem, or None if it writes no table

    @property
    def title(self) -> str:
        """The heading a figure or a console block carries."""
        return f"{self.metric}: {self.summary}"

    @property
    def result_file(self) -> Optional[str]:
        """The JSON the wrapper reads back for its combined summary."""
        return f"{self.prefix}.json" if self.prefix else None


TESTS: Tuple[TestCase, ...] = (
    TestCase(
        code="cp01", metric="CP-01", analysis="camera_detections",
        summary="camera detection drops, counted at the websocket",
        criterion="reported, no threshold",
        settings=CAMERA_DETECTIONS, prefix="cp01_detection_drops",
    ),
    TestCase(
        code="cp02", metric="CP-02", analysis="detection_delivery",
        summary="detection to SDSM received at the vehicle",
        criterion="drop rate at or below the limit, per run",
        settings=DETECTION_DELIVERY, prefix="cp02_detection_to_sdsm",
    ),
    TestCase(
        code="cp03", metric="CP-03", analysis="broadcast_delivery",
        summary="RSU broadcast to CARMA Platform receipt",
        criterion="drop rate at or below the limit, per run",
        settings=BROADCAST_DELIVERY, prefix="cp03_rsu_to_vehicle",
    ),
    TestCase(
        code="dt05", metric="DT-05", analysis="delivery_latency",
        summary="detection to SDSM receipt latency",
        criterion="median latency under the limit, per run",
        settings=DELIVERY_LATENCY, prefix="dt05_detection_to_sdsm_receipt",
    ),
    TestCase(
        code="pl01", metric="PL-01", analysis="message_rates",
        summary="per-topic message rates and OBU radio activity",
        criterion="each rate inside its tolerance",
        settings=MESSAGE_RATES, prefix="pl01_message_communication",
    ),
    TestCase(
        code="pl03", metric="PL-03", analysis="vehicle_yield",
        summary="CARMA Platform vehicle yield",
        criterion="the target share of valid runs must yield",
        settings=VEHICLE_YIELD, prefix="pl03_vehicle_yield",
    ),
    TestCase(
        code="cs01", metric="CS-01", analysis="location_spoofing",
        summary="SDSM location spoofing verification",
        criterion="mean position and heading error under their limits",
        settings=LOCATION_SPOOFING, prefix="cs01_location_spoofing",
    ),
    TestCase(
        code="cascade", metric="Cascade", analysis="latency_cascade",
        summary="end-to-end per-detection latency breakdown",
        criterion="reported, no threshold",
        settings=LATENCY_CASCADE,
    ),
)

BY_CODE: Dict[str, TestCase] = {case.code: case for case in TESTS}


# ---------------------------------------------------------------------------
# Turning a config into a command line
# ---------------------------------------------------------------------------

# Field types the generated command line can carry. A dict of topic rates or a
# tuple of coordinates cannot be expressed as one flag, so it is configured
# here and not exposed; only scalars become options.
_CLI_TYPES = (float, int, str, bool)


def cli_fields(instance):
    """``(flag, name, type, default, help)`` for each scalar field.

    The entry point builds its argument parser from this, so a tunable added to
    a settings dataclass is overridable from the command line without touching
    the parser. A field with no ``cli_help`` entry is deliberately not exposed:
    it is configuration, not an experiment knob.
    """
    help_map = getattr(type(instance), "cli_help", {})
    exposed = []
    for spec in dataclasses.fields(instance):
        if spec.name not in help_map:
            continue
        value = getattr(instance, spec.name)
        if not isinstance(value, _CLI_TYPES) or isinstance(value, bool):
            continue
        flag = "--" + spec.name.replace("_", "-")
        exposed.append((flag, spec.name, type(value), value, help_map[spec.name]))
    return exposed


def apply_overrides(instance, values: Dict):
    """A copy of ``instance`` with any non-``None`` override in ``values`` applied.

    ``values`` is the parsed namespace as a dict, so unrelated arguments are
    ignored and an option the user did not give leaves the configured value
    alone.
    """
    changes = {
        name: values[name]
        for _flag, name, _type, _default, _help in cli_fields(instance)
        if values.get(name) is not None
    }
    return dataclasses.replace(instance, **changes) if changes else instance


def describe(instance) -> Dict:
    """The settings as plain data, to record beside the result they produced."""
    return {
        spec.name: getattr(instance, spec.name)
        for spec in dataclasses.fields(instance)
        if isinstance(getattr(instance, spec.name), _CLI_TYPES)
    }


# ---------------------------------------------------------------------------
# Plot style, shared by every figure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotStyle:
    """One look for the whole suite, so the figures read as one report."""

    # Sequential ramp for the cascade stages. Position in the pipeline is an
    # order, so it gets an ordered ramp; a rainbow would invent banding that is
    # not in the data.
    stage_colormap: str = "viridis"
    stage_cmap_range: Tuple[float, float] = (0.0, 1.0)

    # Cool at rest through hot at speed, for the trajectory plot. Sequential
    # rather than a literal blue-to-red: plasma's lightness rises monotonically,
    # so a mid-speed segment cannot read as a fast one.
    speed_colormap: str = "plasma"

    # Unit the speed figures are drawn in. Display only: the recorded speeds,
    # the halt rule, the tables and the cache all stay in m/s, so changing this
    # cannot change a result.
    speed_unit: str = "mph"
    speed_unit_per_mps: float = 2.2369362920544   # 1 m/s = 2.237 mph

    # Acceptance lines and the runs that cross them.
    threshold_color: str = "#b00020"
    annotation_color: str = "#333333"

    # Pedestrian reports, drawn as a density rather than as marks.
    pedestrian_color: str = "#6b6b6b"
    pedestrian_alpha: float = 0.02

    # Mean and spread on the speed profile.
    mean_color: str = "#2a78d6"
    individual_color: str = "#999999"

    # Line width at the slowest and fastest speed on the trajectory plot. Width
    # runs *inverse* to speed, so a halted vehicle draws a thick stroke and a
    # fast one a thin thread, encoding speed a second time alongside the colour.
    # Set both equal to turn the width encoding off and let colour carry it
    # alone.
    linewidth_slow: float = 5.0
    linewidth_fast: float = 5.0
    trajectory_alpha: float = 0.2

    figure_dpi: int = 200
    grid_alpha: float = 0.25
    grid_linewidth: float = 0.6


STYLE = PlotStyle()


def despine(axis) -> None:
    """Drop the top and right spines. Every figure in the suite does this."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def grid(axis, which: str = "y") -> None:
    """The suite's recessive grid, behind the data."""
    axis.grid(axis=which, alpha=STYLE.grid_alpha, linewidth=STYLE.grid_linewidth)
    axis.set_axisbelow(True)


def threshold_line(axis, value: float, label: str, right_at: float,
                   va: str = "bottom") -> None:
    """Draw an acceptance limit and name it, the same way on every figure."""
    axis.axhline(value, color=STYLE.threshold_color, linestyle="--", linewidth=1.0)
    axis.text(right_at, value, f" {label}", color=STYLE.threshold_color,
              fontsize=8, va=va, ha="right")
