#!/usr/bin/env python3
"""Run every DT-WZ analysis over one or more verification sessions.

A loop over ``run_dt_wz_analysis.py``. It runs each test in turn and collects
their results into one summary::

    python run_all_dt_wz_analysis.py \\
        --data-root .../20260914_verification_test \\
        --data-root .../20260915_verification_test \\
        --output-dir out

Each test writes into its own sub-directory, exactly as it does when run on its
own. The table of tests, what each measures and its acceptance criterion is the
``TESTS`` catalogue in ``dt_wz_analysis_util.config``; this script reads it
rather than repeating it, so the two cannot disagree.

Use ``run_dt_wz_analysis.py <test>`` directly to re-run a single test without
repeating the rest, which matters because the cascade and CP-02 each parse logs
of over a million lines.

Pass ``--only`` to select a subset, for example ``--only cp02 cp03``. Options
that belong to one test -- ``--plot-only``, ``--max-drop-rate-pct`` and the rest
-- are not forwarded from here: run that test on its own to change them.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dt_wz_analysis_util import config as cfg  # noqa: E402
from dt_wz_analysis_util.run_dt_wz_analysis import run_test  # noqa: E402

# Keys worth lifting out of a test's own summary into the combined one.
_HEADLINE_KEYS = (
    "metric", "headline", "pass_rate", "runs_passed", "runs_evaluated",
    "pooled_drop_rate_pct", "pooled_median", "unit", "total_dropped",
    "total_checked", "pass", "mean_position_error_m", "mean_heading_error_deg",
    "success_rate_pct", "valid_runs", "invalid_runs", "is_passed",
    "verified_objects", "failed_runs",
)


def _headline(summary, output_dir: Path):
    """The one-line result of one test, for the combined summary."""
    if not summary:
        return None
    headline = {"output": str(output_dir)}
    headline.update({key: summary[key] for key in _HEADLINE_KEYS if key in summary})
    return headline


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session directory with runs.csv; repeat to pool sessions")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Parent directory; each test writes into its own sub-folder")
    parser.add_argument("--pl01-data-root", type=Path, action="append",
                        help="Session(s) carrying MAP/SPAT/MOM/BSM for PL-01. When "
                             "given, --data-root is used for PL-01's SDSM rate "
                             "instead. Needed because no single session carries "
                             "every message.")
    parser.add_argument("--only", nargs="+", choices=sorted(cfg.BY_CODE),
                        help="Run only these tests (default: all)")
    args = parser.parse_args(argv)

    selected = args.only or list(cfg.BY_CODE)
    roots = []
    for root in args.data_root:
        roots += ["--data-root", str(root)]

    results, failures = {}, []
    for code in selected:
        output_dir = args.output_dir / code
        arguments = list(roots)
        if code == "pl01" and args.pl01_data_root:
            # PL-01 measures MAP/SPAT/MOM/BSM on the dedicated session and SDSM
            # on the verification sessions, so its two groups swap roles here.
            arguments = []
            for root in args.pl01_data_root:
                arguments += ["--data-root", str(root)]
            for root in args.data_root:
                arguments += ["--secondary-data-root", str(root)]
        try:
            # A non-zero code means the test's own criterion failed. That is a
            # result, not an error, so the remaining tests still run.
            _code, summary = run_test(code, arguments + ["--output-dir", str(output_dir)])
            results[code] = _headline(summary, output_dir)
        except Exception as error:
            print(f"ERROR in {code}: {error}")
            traceback.print_exc(limit=3)
            failures.append(code)
            results[code] = {"error": str(error)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps({
        "analysis_time": datetime.now().isoformat(),
        "sessions": [str(root) for root in args.data_root],
        "analyses": results,
        "failed_to_run": failures,
    }, indent=2, default=str))

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for code, headline in results.items():
        if not headline:
            print(f"  {code:<9} (no machine-readable summary; "
                  f"see {args.output_dir / code})")
        elif "error" in headline:
            print(f"  {code:<9} ERROR: {headline['error']}")
        else:
            parts = []
            if headline.get("headline"):
                parts.append(headline["headline"])
            if headline.get("pass_rate"):
                parts.append(f"{headline['runs_passed']}/{headline['runs_evaluated']} "
                             f"runs passed")
            verdict = headline.get("is_passed", headline.get("pass"))
            if verdict is not None:
                parts.append("PASS" if verdict else "FAIL")
            print(f"  {code:<9} {' | '.join(parts) if parts else 'done'}")
    print(f"\nCombined summary -> {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
