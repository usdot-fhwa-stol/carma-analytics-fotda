#!/usr/bin/env python3
"""
Pools per-run pcap<->mcap boundary-crossing latency (see
correlate_pcap_mcap.py) across every run for one or more OBU vendors, so the
combined mean/median/p95/p99/max are computed over the actual pooled sample
set rather than being approximated from each run's own summary stats.

Discovers runs automatically within each vendor directory by pairing
`run-N-eth0-*.pcap` with whichever `run-N*.mcap` exists alongside it (mcap
naming isn't fully consistent across recordings - e.g. one run's mcap has an
extra descriptive suffix, another vendor's run-4 mcap is missing its usual
"-<vendor>" suffix - so pairing is done by run number, not exact filename).

Also separates message types whose payload is effectively static (e.g. a
fixed-geometry MAP, or a fixed test-string custom-PSID request) from types
with per-instance-unique content (BSM/SPAT/SDSM): for static-payload types,
correlate_pcap_mcap.py's payload-based matching can't prove which specific
wire packet corresponds to which specific event, so a consistent added
latency for those types should be read as a possible pipeline/session
warm-up effect worth checking (see README), not taken at face value the same
way a BSM/SPAT/SDSM latency number can be.

Usage:
    python3 aggregate_boundary_latency.py \\
        --vendor Ettifos=/workspaces/carma_ws/data/ettifos \\
        --vendor Commsignia=/workspaces/carma_ws/data/commsignia-v2
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correlate_j2735_latency as base  # noqa: E402
from correlate_pcap_mcap import correlate_across_boundary, extract_mcap_binary_messages  # noqa: E402
from correlation_plots import find_duplicate_content_types  # noqa: E402


def discover_runs(vendor_dir: Path):
    """Pairs each run-N-eth0-*.pcap with the run-N*.mcap in the same
    directory. Matches by run number with a negative lookahead so run-1
    doesn't also swallow run-10, etc."""
    runs = []
    for eth0_pcap in sorted(vendor_dir.glob("run-*-eth0-*.pcap")):
        m = re.match(r"run-(\d+)-eth0-", eth0_pcap.name)
        if not m:
            continue
        run_num = m.group(1)
        mcap_candidates = [
            p for p in vendor_dir.glob(f"run-{run_num}*.mcap")
            if re.match(rf"run-{run_num}(?!\d)", p.name)
        ]
        if not mcap_candidates:
            print(f"WARNING: no mcap found for {eth0_pcap.name}, skipping", file=sys.stderr)
            continue
        if len(mcap_candidates) > 1:
            print(f"WARNING: multiple mcap candidates for run-{run_num}: "
                  f"{[p.name for p in mcap_candidates]}, using {mcap_candidates[0].name}", file=sys.stderr)
        runs.append((f"run-{run_num}", eth0_pcap, mcap_candidates[0]))
    return runs


def pct(vals_sorted, p):
    n = len(vals_sorted)
    return vals_sorted[int(p * (n - 1))] if n else float("nan")


def stats(vals):
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    return {
        "n": n, "min": vals_sorted[0], "mean": sum(vals_sorted) / n,
        "median": vals_sorted[n // 2], "p95": pct(vals_sorted, 0.95),
        "p99": pct(vals_sorted, 0.99), "max": vals_sorted[-1],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", action="append", required=True, metavar="NAME=DIR",
                     help="Vendor label and its data directory, e.g. Ettifos=/path/to/ettifos. Repeatable.")
    ap.add_argument("--drop-threshold-ms", type=float, default=base.DROP_LATENCY_THRESHOLD_MS)
    args = ap.parse_args()

    vendors = []
    for spec in args.vendor:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--vendor must be NAME=DIR, got: {spec!r}")
        vendors.append((name, Path(path)))

    overall = {}
    per_run_rows = []
    by_type = {}
    dup_flagged = {}

    for obu, vendor_dir in vendors:
        runs = discover_runs(vendor_dir)
        if not runs:
            print(f"WARNING: no runs discovered under {vendor_dir}", file=sys.stderr)
        pooled = {
            "inbound": {"fresh": [], "matched": 0, "stale": 0, "dropped": 0, "out_of_window": 0, "total": 0},
            "outbound": {"fresh": [], "matched": 0, "stale": 0, "dropped": 0, "out_of_window": 0, "total": 0},
        }
        for run_label, eth0_pcap, mcap_path in runs:
            pcap_by_dir, _ = base.extract_messages(str(eth0_pcap))
            mcap_by_dir = extract_mcap_binary_messages(str(mcap_path))

            pcap_incoming_pool = pcap_by_dir["incoming"] + pcap_by_dir["other"]
            in_lat, in_drop, in_stale, in_oow = correlate_across_boundary(
                pcap_incoming_pool, mcap_by_dir["inbound"], args.drop_threshold_ms)
            out_lat, out_drop, out_stale, out_oow = correlate_across_boundary(
                mcap_by_dir["outbound"], pcap_by_dir["outgoing"], args.drop_threshold_ms)

            dup_flagged.setdefault((obu, "inbound"), set()).update(
                find_duplicate_content_types(pcap_incoming_pool))
            dup_flagged.setdefault((obu, "outbound"), set()).update(
                find_duplicate_content_types(mcap_by_dir["outbound"]))

            for direction, (lat, drop, stale, oow, total) in {
                "inbound": (in_lat, in_drop, in_stale, in_oow, len(pcap_incoming_pool)),
                "outbound": (out_lat, out_drop, out_stale, out_oow, len(mcap_by_dir["outbound"])),
            }.items():
                pooled[direction]["fresh"].extend(l[2] for l in lat)
                pooled[direction]["matched"] += len(lat)
                pooled[direction]["stale"] += len(stale)
                pooled[direction]["dropped"] += len(drop)
                pooled[direction]["out_of_window"] += len(oow)
                pooled[direction]["total"] += total
                for l in lat:
                    by_type.setdefault((obu, direction, l[1]), []).append(l[2])

                s = stats([l[2] for l in lat])
                per_run_rows.append({
                    "obu": obu, "run": run_label, "direction": direction,
                    "total": total, "matched": len(lat), "stale": len(stale),
                    "dropped": len(drop), "out_of_window": len(oow), "stats": s,
                })
            print(f"done: {obu} {run_label}", file=sys.stderr)

        overall[obu] = pooled

    print("\n=== Per-run detail (fresh-matched latency, ms; signed rx-tx) ===")
    hdr = (f"{'OBU':<11}{'Run':<7}{'Dir':<9}{'Total':>7}{'Match':>7}{'Match%':>8}{'Stale':>7}{'Drop':>6}"
           f"{'Mean':>8}{'Med':>8}{'p95':>8}{'p99':>8}{'Max':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in per_run_rows:
        s = r["stats"]
        matchpct = 100 * r["matched"] / r["total"] if r["total"] else 0.0
        if s:
            print(f"{r['obu']:<11}{r['run']:<7}{r['direction']:<9}{r['total']:>7}{r['matched']:>7}{matchpct:>7.1f}%"
                  f"{r['stale']:>7}{r['dropped']:>6}{s['mean']:>8.2f}{s['median']:>8.2f}{s['p95']:>8.2f}"
                  f"{s['p99']:>8.2f}{s['max']:>9.2f}")
        else:
            print(f"{r['obu']:<11}{r['run']:<7}{r['direction']:<9}{r['total']:>7}{r['matched']:>7}{matchpct:>7.1f}%"
                  f"{r['stale']:>7}{r['dropped']:>6}{'--':>8}{'--':>8}{'--':>8}{'--':>8}{'--':>9}")

    print("\n=== Pooled summary per OBU (all runs combined, fresh-matched only) ===")
    hdr2 = (f"{'OBU':<11}{'Direction':<10}{'Total':>7}{'Matched':>8}{'Match%':>8}{'Stale':>7}{'Dropped':>8}"
            f"{'OutWin':>7}{'Mean':>8}{'Median':>8}{'p95':>8}{'p99':>8}{'Max':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    for obu, pooled in overall.items():
        for direction in ("inbound", "outbound"):
            d = pooled[direction]
            s = stats(d["fresh"])
            matchpct = 100 * d["matched"] / d["total"] if d["total"] else 0.0
            if s:
                print(f"{obu:<11}{direction:<10}{d['total']:>7}{d['matched']:>8}{matchpct:>7.1f}%{d['stale']:>7}"
                      f"{d['dropped']:>8}{d['out_of_window']:>7}{s['mean']:>8.2f}{s['median']:>8.2f}"
                      f"{s['p95']:>8.2f}{s['p99']:>8.2f}{s['max']:>9.2f}")
            else:
                print(f"{obu:<11}{direction:<10}{d['total']:>7}{d['matched']:>8}{matchpct:>7.1f}%{d['stale']:>7}"
                      f"{d['dropped']:>8}{d['out_of_window']:>7}{'--':>8}{'--':>8}{'--':>8}{'--':>8}{'--':>9}")

    print("\n=== Pooled fresh-matched latency by message type (ms; signed rx-tx) ===")
    hdr3 = f"{'OBU':<11}{'Direction':<10}{'MsgType':<19}{'N':>6}{'Mean':>8}{'Median':>8}{'p95':>8}{'p99':>8}{'Max':>9}  Note"
    print(hdr3)
    print("-" * len(hdr3))
    for (obu, direction, msg_type), vals in sorted(by_type.items()):
        s = stats(vals)
        note = ("duplicate/static payload - matching can't prove causality; a consistent offset "
                 "may be a pipeline/session warm-up rather than per-message latency, see docstring")
        note = note if msg_type in dup_flagged.get((obu, direction), set()) else ""
        print(f"{obu:<11}{direction:<10}{msg_type:<19}{s['n']:>6}{s['mean']:>8.2f}{s['median']:>8.2f}"
              f"{s['p95']:>8.2f}{s['p99']:>8.2f}{s['max']:>9.2f}  {note}")


if __name__ == "__main__":
    main()
