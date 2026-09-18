#!/usr/bin/env python3
"""CP-03: RSU broadcast to CARMA Platform receipt, pooled across sessions.

Counts SDSMs the RSU put on the air that the vehicle never received::

    python run_cp03_analysis.py \\
        --data-root .../20260914_verification_test \\
        --data-root .../20260915_verification_test \\
        --output-dir out/cp03

Matching is byte-for-byte on the J2735 payload, so a broadcast and its receipt
are the same message by identity rather than by time proximity. Only broadcasts
inside the engaged window are scored, since the vehicle was not listening
outside it.

Where the OBU capture is a binary pcap it also carries payloads, and the radio's
own reception rate is reported alongside -- that measures the air link on its
own, separately from the vehicle's software receiving the message.

Writes ``cp03_rsu_to_vehicle.json``, a per-run CSV and a per-run plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dt_wz_analysis_util import report
from dt_wz_analysis_util import metrics

METRIC = "CP-03"
PREFIX = "cp03_rsu_to_vehicle"
TITLE = "CP-03: RSU broadcast to CARMA Platform receipt"


def measure(run, window, _context):
    passed, stats = metrics.rsu_transmission_drop_rate(run.mcap, run.rsu_pcap, window)
    latency = stats.get("receive_latency_ms") or {}
    row = {
        "checked": stats.get("total_broadcasts_checked"),
        "received": stats.get("total_received"),
        "received_late": stats.get("total_received_late"),
        "dropped": stats.get("total_dropped"),
        "drop_rate_pct": (
            round(stats["drop_rate_pct"], 3) if stats.get("drop_rate_pct") is not None else None
        ),
        "receive_latency_median_ms": (
            round(latency["median"], 3) if latency.get("median") is not None else None
        ),
        "passed": passed,
    }

    # The radio's own reception, where the OBU capture carries payloads.
    try:
        obu = metrics.obu_radio_activity(
            run.obu_capture, run.start_time.date(), run.mcap, window, rsu_pcap=run.rsu_pcap
        )
        row["obu_payload_matched"] = obu.get("payload_matched")
        if obu.get("ota_reception_rate_pct") is not None:
            row["ota_received_by_radio"] = obu["ota_received_by_radio"]
            row["ota_missed_by_radio"] = obu["ota_missed_by_radio"]
            row["ota_reception_pct"] = round(obu["ota_reception_rate_pct"], 3)
    except Exception as error:
        row["obu_error"] = str(error)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Session directory with runs.csv; repeat to pool sessions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-drop-rate-pct", type=float,
                        default=metrics.RSU_TRANSMISSION_DROP_RATE_THRESHOLD_PCT)
    args = parser.parse_args(argv)

    rows = report.analyse_runs(args.data_root, measure)
    summary = report.summarise(rows, METRIC, args.max_drop_rate_pct)
    summary["sessions"] = [str(root) for root in args.data_root]

    # Pooled air-link reception, where it could be measured at all.
    broadcast = sum(int(row.get("ota_received_by_radio") or 0)
                    + int(row.get("ota_missed_by_radio") or 0) for row in rows)
    missed = sum(int(row.get("ota_missed_by_radio") or 0) for row in rows)
    if broadcast:
        summary["ota_broadcasts_checked"] = broadcast
        summary["ota_missed_by_radio"] = missed
        summary["ota_reception_pct"] = round((broadcast - missed) / broadcast * 100.0, 3)

    report.write_outputs(summary, args.output_dir, PREFIX, TITLE)
    if broadcast:
        print(f"       air link: {broadcast - missed}/{broadcast} broadcasts reached the "
              f"radio ({summary['ota_reception_pct']:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
