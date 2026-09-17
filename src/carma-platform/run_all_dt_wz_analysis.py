#!/usr/bin/env python3
"""Run every DT-WZ analysis over one or more verification sessions.

A wrapper. It runs each metric's own script in turn and collects their results
into one summary::

    python run_all_dt_wz_analysis.py \\
        --data-root .../20260914_verification_test \\
        --data-root .../20260915_verification_test \\
        --output-dir out

Each analysis writes into its own sub-directory, exactly as it does when run on
its own:

=========  ==========================================================  ==========
folder     measures                                                    criterion
=========  ==========================================================  ==========
cp01/      camera detection drops, counted at the websocket            (reported)
cp02/      detection -> SDSM received at the vehicle                   <= 2%
cp03/      RSU broadcast -> CARMA Platform receipt                     <= 2%
dt05/      detection -> SDSM receipt at the vehicle                    median < 0.3 s
pl01/      per-topic message rates, OBU radio activity                 +/-20%
cs01/      SDSM location spoofing verification                         0.2 m, 1 deg
cascade/   end-to-end per-detection latency breakdown                  (reported)
=========  ==========================================================  ==========

The individual scripts remain the real entry points. Use one of them directly to
re-run a single metric without repeating the rest, which matters because the
cascade and CP-02 each parse logs of over a million lines.

Pass ``--only`` to select a subset, for example ``--only cp02 cp03``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import run_cascade_analysis
import run_cp01_analysis
import run_cp02_analysis
import run_cp03_analysis
import run_cs01_analysis
import run_dt05_analysis
import run_pl01_analysis

# folder -> (module, the JSON it writes)
ANALYSES = {
    "cp01": (run_cp01_analysis, "cp01_detection_drops.json"),
    "cp02": (run_cp02_analysis, "cp02_detection_to_sdsm.json"),
    "cp03": (run_cp03_analysis, "cp03_rsu_to_vehicle.json"),
    "dt05": (run_dt05_analysis, "dt05_detection_to_sdsm_receipt.json"),
    "pl01": (run_pl01_analysis, "pl01_message_communication.json"),
    "cs01": (run_cs01_analysis, "cs01_location_spoofing.json"),
    "cascade": (run_cascade_analysis, None),
}


def _headline(name: str, result_path: Path):
    """The one-line result each analysis wrote, for the combined summary."""
    if result_path is None or not result_path.is_file():
        return None
    summary = json.loads(result_path.read_text())
    headline = {"output": str(result_path.parent)}
    for key in ("metric", "headline", "pass_rate", "runs_passed", "runs_evaluated",
                "pooled_drop_rate_pct", "pooled_median", "unit", "total_dropped",
                "total_checked", "pass", "mean_position_error_m", "mean_heading_error_deg",
                "verified_objects", "failed_runs"):
        if key in summary:
            headline[key] = summary[key]
    return headline


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session directory with runs.csv; repeat to pool sessions")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Parent directory; each analysis writes into its own sub-folder")
    parser.add_argument("--only", nargs="+", choices=sorted(ANALYSES),
                        help="Run only these analyses (default: all)")
    args = parser.parse_args(argv)

    selected = args.only or list(ANALYSES)
    roots = []
    for root in args.data_root:
        roots += ["--data-root", str(root)]

    results, failures = {}, []
    for name in selected:
        module, result_file = ANALYSES[name]
        output_dir = args.output_dir / name
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        try:
            module.main(roots + ["--output-dir", str(output_dir)])
            results[name] = _headline(name, output_dir / result_file if result_file else None)
        except SystemExit as exit_code:
            # A metric script exits non-zero when its own criterion failed; that
            # is a result, not an error, so the remaining analyses still run.
            if exit_code.code:
                results[name] = _headline(name, output_dir / result_file if result_file else None)
            else:
                results[name] = None
        except Exception as error:
            print(f"ERROR in {name}: {error}")
            traceback.print_exc(limit=3)
            failures.append(name)
            results[name] = {"error": str(error)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "analysis_time": datetime.now().isoformat(),
        "sessions": [str(root) for root in args.data_root],
        "analyses": results,
        "failed_to_run": failures,
    }
    summary_path = args.output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for name, headline in results.items():
        if not headline:
            print(f"  {name:<9} (no machine-readable summary; see {args.output_dir / name})")
        elif "error" in headline:
            print(f"  {name:<9} ERROR: {headline['error']}")
        else:
            parts = []
            if headline.get("headline"):
                parts.append(headline["headline"])
            if headline.get("pass_rate"):
                parts.append(f"{headline['runs_passed']}/{headline['runs_evaluated']} runs passed")
            if "pass" in headline:
                parts.append("PASS" if headline["pass"] else "FAIL")
            print(f"  {name:<9} {' | '.join(parts) if parts else 'done'}")
    print(f"\nCombined summary -> {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
