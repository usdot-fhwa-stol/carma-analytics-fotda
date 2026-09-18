#!/usr/bin/env python3
"""One entry point for every DT-WZ analysis::

    python run_dt_wz_analysis.py cp02 \\
        --data-root .../20260914_verification_test \\
        --data-root .../20260915_verification_test \\
        --output-dir out/cp02

There used to be eight near-identical scripts, one per test. They shared their
argument parsing, their session loop and their output layout, and drifted apart
anyway: two of them accepted a ``--max-drop-rate-pct`` that never reached the
measurement, so a report could state one limit while applying another.

So there is one script, and the tests differ from each other in data rather
than in code. Which tests exist, what each measures, its acceptance criterion
and its output names are all in ``config.TESTS``. This module holds only the
glue: a handler per *measurement*, named for what it measures, and a generated
command line.

**The command line comes from the configuration.** Each test names a frozen
settings dataclass, and every scalar field of it with a ``cli_help`` entry
becomes an option. Adding a tunable to ``config.py`` therefore adds the flag
that overrides it, and the value the measurement uses is the value the report
records -- they are the same object.

Run ``--list`` to see the tests, or ``<test> --help`` for one test's options::

    python run_dt_wz_analysis.py --list
    python run_dt_wz_analysis.py pl03 --help

``run_all_dt_wz_analysis.py`` is a thin loop over this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from dt_wz_analysis_util import config as cfg
from dt_wz_analysis_util import (camera_detections, location_spoofing,
                                 message_rates, metrics, report, vehicle_yield)
from dt_wz_analysis_util.cascade import latency_cascade
from dt_wz_analysis_util.readers import kafka_log


def _option(*names, **kwargs):
    """Declare a non-settings option, deferred until the parser exists."""
    return lambda parser: parser.add_argument(*names, **kwargs)


# Options that belong to one analysis and are not settings: an extra input
# group, or a switch that changes what the run does rather than what it
# measures. Keyed by ``TestCase.analysis``.
EXTRA_ARGS: Dict[str, Tuple[Callable, ...]] = {
    "message_rates": (
        _option("--secondary-data-root", "--sdsm-data-root", type=Path,
                action="append", dest="secondary_data_root",
                help="Session(s) carrying the secondary topic group; defaults "
                     "to --data-root. Needed because no single session carries "
                     "every message type."),
    ),
    "vehicle_yield": (
        _option("--plot-only", action="store_true",
                help="Redraw the figures from the cached geometry instead of "
                     "re-reading the recordings. Falls back to a full run if "
                     "no cache is present."),
        _option("--speed-cmap", default=cfg.STYLE.speed_colormap,
                help=f"Colour ramp for speed (default "
                     f"{cfg.STYLE.speed_colormap}; 'coolwarm' for a literal "
                     f"blue-to-red)"),
    ),
    "location_spoofing": (
        _option("--max-position-error-m", type=float,
                default=location_spoofing.DEFAULT_MAX_MEAN_POSITION_ERROR_M,
                help="Acceptance limit on the mean position error "
                     "(default from the carma-streets verifier)"),
        _option("--max-heading-error-deg", type=float,
                default=location_spoofing.DEFAULT_MAX_MEAN_HEADING_ERROR_DEG,
                help="Acceptance limit on the mean heading error "
                     "(default from the carma-streets verifier)"),
    ),
}


# ---------------------------------------------------------------------------
# One handler per measurement
#
# Each is glue: it reads the arguments, calls the measurement API in the
# package and hands the result to that measurement's writer. The measurements
# and the figures live in their own modules, so nothing below decides anything.
# ---------------------------------------------------------------------------

def _run_camera_detections(case, args):
    results = camera_detections.analyse_sessions(args.data_root, args.settings)
    summary = camera_detections.summarise(results, args.settings)
    summary["sessions"] = [str(root) for root in args.data_root]
    camera_detections.write_outputs(results, summary, args.output_dir, case)
    return summary


def _detection_delivery_measure(settings):
    """One run's row. The limit reaches the measurement, not just the report."""

    def measure(run, window, detection_records):
        passed, stats = metrics.detection_to_sdsm_drop_rate(
            run.mcap, detection_records, window,
            max_drop_rate_pct=settings.max_drop_rate_pct,
            match_tolerance_sec=settings.match_tolerance_sec,
        )
        return {
            "checked": stats.get("total_raw_detections"),
            "matched": stats.get("total_matched"),
            "dropped": stats.get("total_dropped"),
            "drop_rate_pct": _round(stats.get("drop_rate_pct"), 3),
            "sdsm_objects_at_vehicle": stats.get("sdsm_object_detections"),
            "passed": passed,
        }

    return measure


def _detection_log(session):
    """Parse the session-wide detection log once, not once per run."""
    if session.kafka_detected_object is None:
        raise FileNotFoundError("the v2xhub_sim_sensor_detected_object log is required")
    print(f"  parsing {session.kafka_detected_object.name} ...", flush=True)
    return kafka_log.parse_kafka_log_records(session.kafka_detected_object)


def _run_detection_delivery(case, args):
    rows = report.analyse_runs(
        args.data_root, _detection_delivery_measure(args.settings), _detection_log)
    summary = report.summarise(rows, case.metric, args.settings.max_drop_rate_pct)
    summary["sessions"] = [str(root) for root in args.data_root]
    report.write_outputs(summary, args.output_dir, case.prefix, case.title)
    return summary


def _broadcast_delivery_measure(settings):
    def measure(run, window, _context):
        passed, stats = metrics.rsu_transmission_drop_rate(
            run.mcap, run.rsu_pcap, window,
            max_drop_rate_pct=settings.max_drop_rate_pct,
            late_threshold_ms=settings.late_threshold_ms,
        )
        latency = stats.get("receive_latency_ms") or {}
        row = {
            "checked": stats.get("total_broadcasts_checked"),
            "received": stats.get("total_received"),
            "received_late": stats.get("total_received_late"),
            "dropped": stats.get("total_dropped"),
            "drop_rate_pct": _round(stats.get("drop_rate_pct"), 3),
            "receive_latency_median_ms": _round(latency.get("median"), 3),
            "passed": passed,
        }
        # The radio's own reception, where the OBU capture carries payloads.
        # Reported alongside because it separates the air link from the
        # vehicle's software receiving the message.
        try:
            obu = metrics.obu_radio_activity(
                run.obu_capture, run.start_time.date(), run.mcap, window,
                rsu_pcap=run.rsu_pcap)
            row["obu_payload_matched"] = obu.get("payload_matched")
            if obu.get("ota_reception_rate_pct") is not None:
                row["ota_received_by_radio"] = obu["ota_received_by_radio"]
                row["ota_missed_by_radio"] = obu["ota_missed_by_radio"]
                row["ota_reception_pct"] = round(obu["ota_reception_rate_pct"], 3)
        except Exception as error:
            row["obu_error"] = str(error)
        return row

    return measure


def _run_broadcast_delivery(case, args):
    rows = report.analyse_runs(
        args.data_root, _broadcast_delivery_measure(args.settings))
    summary = report.summarise(rows, case.metric, args.settings.max_drop_rate_pct)
    summary["sessions"] = [str(root) for root in args.data_root]

    # Pooled air-link reception, over the runs where it could be measured.
    broadcast = sum(int(row.get("ota_received_by_radio") or 0)
                    + int(row.get("ota_missed_by_radio") or 0) for row in rows)
    missed = sum(int(row.get("ota_missed_by_radio") or 0) for row in rows)
    if broadcast:
        summary["ota_broadcasts_checked"] = broadcast
        summary["ota_missed_by_radio"] = missed
        summary["ota_reception_pct"] = round((broadcast - missed) / broadcast * 100.0, 3)

    report.write_outputs(summary, args.output_dir, case.prefix, case.title)
    if broadcast:
        print(f"       air link: {broadcast - missed}/{broadcast} broadcasts reached "
              f"the radio ({summary['ota_reception_pct']:.2f}%)")
    return summary


def _delivery_latency_measure(settings):
    def measure(run, window, _context):
        passed, stats = metrics.detection_to_sdsm_receipt_latency(
            run.mcap, window, threshold_sec=settings.max_median_latency_sec)
        latency = stats.get("latency_s") or {}
        return {
            "samples": stats.get("sample_count"),
            "median_s": _round(stats.get("median_latency_s"), 4),
            "p95_s": _round(latency.get("maximum"), 4),
            "passed": passed,
        }

    return measure


def _run_delivery_latency(case, args):
    rows = report.analyse_runs(args.data_root, _delivery_latency_measure(args.settings))
    summary = report.summarise_latency(
        rows, case.metric, args.settings.max_median_latency_sec, args.settings.unit)
    summary["sessions"] = [str(root) for root in args.data_root]
    report.write_outputs(summary, args.output_dir, case.prefix, case.title)
    return summary


def _run_message_rates(case, args):
    summary = message_rates.analyse(
        args.data_root, args.secondary_data_root, args.settings, case.metric)
    message_rates.write_outputs(summary, args.output_dir, case)
    return summary


def _run_vehicle_yield(case, args):
    settings = args.settings
    cache_path = args.output_dir / settings.cache_name
    summary_path = args.output_dir / f"{case.prefix}.json"

    # --plot-only redraws from the cached geometry. Re-reading 30 recordings
    # costs minutes and yields exactly the same points, so iterating on a figure
    # should not pay for it.
    results, summary, from_cache = None, None, False
    if args.plot_only:
        tracks = vehicle_yield.load_tracks(cache_path)
        if tracks and summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            results = [vehicle_yield.RunYield.from_row(row)
                       for row in summary["runs_detail"]]
            for item in results:
                item.track = tracks.get(item.run)
            from_cache = True
            print(f"Redrawing from {cache_path.name} ({len(tracks)} cached runs); "
                  f"the recordings were not read.")
        else:
            missing = "cache" if not tracks else "summary"
            print(f"No {missing} in {args.output_dir}; running the full analysis "
                  f"once to build it.")

    if results is None:
        results = vehicle_yield.analyse(args.data_root, settings)
        summary = vehicle_yield.summarise(
            results, settings.target_success_pct, settings, case.metric)
        summary["sessions"] = [str(root) for root in args.data_root]
        args.output_dir.mkdir(parents=True, exist_ok=True)

    vehicle_yield.write_outputs(results, summary, args.output_dir, case, settings,
                                speed_cmap=args.speed_cmap,
                                write_tables=not from_cache)
    if not from_cache and vehicle_yield.save_tracks(results, cache_path):
        print(f"Cached trajectories -> {cache_path}")
    return summary


def _run_location_spoofing(case, args):
    result = location_spoofing.analyse_data_roots(
        args.data_root, output_dir=args.output_dir, config=args.settings,
        metric_label=case.title,
        max_mean_position_error_m=args.max_position_error_m,
        max_mean_heading_error_deg=args.max_heading_error_deg,
    )
    location_spoofing.write_outputs(result, args.output_dir, case)
    return result


def _run_latency_cascade(case, args):
    tables, rows = latency_cascade.collect(args.data_root)
    latency_cascade.write_outputs(tables, rows, args.output_dir)
    return None


# ``TestCase.analysis`` -> the handler that runs it.
HANDLERS: Dict[str, Callable] = {
    "camera_detections": _run_camera_detections,
    "detection_delivery": _run_detection_delivery,
    "broadcast_delivery": _run_broadcast_delivery,
    "delivery_latency": _run_delivery_latency,
    "message_rates": _run_message_rates,
    "vehicle_yield": _run_vehicle_yield,
    "location_spoofing": _run_location_spoofing,
    "latency_cascade": _run_latency_cascade,
}


def _round(value, places):
    return None if value is None else round(float(value), places)


# ---------------------------------------------------------------------------
# The generated command line
# ---------------------------------------------------------------------------

def _add_common(parser) -> None:
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session directory with runs.csv; repeat to pool sessions")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where this test writes its table, summary and figures")


def _add_settings_options(parser, case: cfg.TestCase) -> None:
    """One option per exposed settings field, so the config *is* the interface."""
    if case.settings is None:
        return
    group = parser.add_argument_group(
        f"{case.metric} settings",
        f"Override a value from config.{type(case.settings).__name__}. Each "
        f"default below is the configured value; the reasoning for it is in "
        f"the comment above that field in config.py.",
    )
    for flag, name, kind, default, help_text in cfg.cli_fields(case.settings):
        shown = f"{default:g}" if isinstance(default, (int, float)) else default
        group.add_argument(flag, dest=name, type=kind, default=None,
                           help=f"{help_text} (default {shown})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_dt_wz_analysis.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="List the tests and what each one measures, then exit")
    subparsers = parser.add_subparsers(dest="test", metavar="TEST")
    for case in cfg.TESTS:
        child = subparsers.add_parser(
            case.code, help=case.summary,
            description=f"{case.title}.\n\nCriterion: {case.criterion}.",
            formatter_class=argparse.RawDescriptionHelpFormatter)
        _add_common(child)
        for declare in EXTRA_ARGS.get(case.analysis, ()):
            declare(child)
        _add_settings_options(child, case)
    return parser


def _print_listing() -> None:
    print(f"{'test':<9}{'metric':<9}measures / criterion")
    for case in cfg.TESTS:
        print(f"{case.code:<9}{case.metric:<9}{case.summary}")
        print(f"{'':<18}criterion: {case.criterion}")


def run_test(code: str, argv) -> Tuple[int, Optional[Dict]]:
    """Run one test with its own argument list. Returns ``(exit_code, summary)``.

    Used by ``run_all_dt_wz_analysis.py``, which needs the summary as data
    rather than as a printed report.
    """
    args = build_parser().parse_args([code] + list(argv))
    return _dispatch(cfg.BY_CODE[code], args)


def _dispatch(case: cfg.TestCase, args) -> Tuple[int, Optional[Dict]]:
    # Fold any overridden option back into a copy of the configured values, so
    # one object carries what the measurement uses and what the report states.
    args.settings = (cfg.apply_overrides(case.settings, vars(args))
                     if case.settings else None)
    print(f"\n{'=' * 70}\n{case.title}\n{'=' * 70}")
    summary = HANDLERS[case.analysis](case, args)
    if summary is None:
        return 0, None
    # A test "fails" when its own criterion was not met. That is a result, not
    # an error, so it is reported through the exit code and the caller decides.
    passed = summary.get("is_passed", summary.get("pass"))
    return (1 if passed is False else 0), summary


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list or not args.test:
        _print_listing()
        return 0 if args.list else 2
    code, _summary = _dispatch(cfg.BY_CODE[args.test], args)
    return code


if __name__ == "__main__":
    sys.exit(main())
