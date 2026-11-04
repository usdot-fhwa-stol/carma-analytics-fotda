#!/usr/bin/env python3
"""
Shared plotting for the two J2735 drop/latency correlators in this directory
(correlate_j2735_latency.py for pcap<->pcap, correlate_pcap_mcap.py for
pcap<->mcap). Both produce the same four buckets per flow - latencies
(matched, fresh), stale (matched, but slower than the drop threshold), drops
(no match found), out_of_window (no match, but the two captures simply
weren't both recording at that instant) - each a list of (timestamp,
msg_type, value) tuples, so one plotting function serves both.

Two-panel figure per flow:
  - top: latency (ms) of fresh matches over time, colored by message type
    (identity - a J2735 message keeps its type's color everywhere).
  - bottom: every tx candidate plotted at its timestamp on a per-type row,
    colored by outcome (matched/stale/dropped/out-of-window) - this is what
    makes drops visible as a shape (a red cluster at a point in time) rather
    than a number in a table.

Color is assigned by the job it's doing, not from one shared cycle: message
type is identity (categorical, fixed hue order), outcome is state (the fixed
status palette - green/amber/red/gray - never reused for series identity).
"""
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical palette, fixed order (never re-sorted per chart - a message
# type keeps its color whether it's the 1st or 5th type present in a given
# run). Values from the design system's validated default palette.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

# Status palette - fixed, reserved for outcome only, never reused for
# message-type identity.
STATUS_COLOR = {
    "matched": "#0ca30c",        # good
    "stale": "#fab219",          # warning
    "dropped": "#d03b3b",        # critical
    "out_of_window": "#898781",  # muted ink (not a real drop)
}
STATUS_ORDER = ["matched", "stale", "dropped", "out_of_window"]
STATUS_Y_OFFSET = {"matched": -0.3, "stale": -0.1, "dropped": 0.1, "out_of_window": 0.3}

CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def windowed_count_report(title, tx_messages, rx_messages):
    """Per-message-type count comparison, robust to the payload-duplication
    problem that makes exact-match latency correlation unreliable for some
    J2735 types: MAP (static intersection geometry) and MobilityOperation
    (in this dataset, a fixed-content periodic broadcast) repeat one
    byte-identical payload hundreds of times in a row, so nearest-timestamp
    matching can't tell individual broadcast instances apart and can
    misattribute a match across an unrelated gap elsewhere in the run
    (confirmed: a real ~32s reception gap in one capture produced a tight
    cluster of ~9000ms "stale" matches for MAP, which was actually just
    every post-gap instance binding to the wrong identical-payload sibling).
    Restricting to each side's own overlapping recording window and just
    comparing per-type counts sidesteps that: it can't mis-time an instance,
    it can only under- or over-count within a span both sides were actually
    recording. BSM/SPAT/SDSM payloads are effectively unique per instance in
    this dataset, so this is a secondary cross-check for those, but it's the
    primary reliable signal for MAP/MobilityOperation-like fixed-content
    types. Equally applicable to pcap<->pcap and pcap<->mcap correlation -
    nothing here is specific to either side's message format."""
    tx_ts = [m["timestamp"] for m in tx_messages if m["timestamp"] is not None]
    rx_ts = [m["timestamp"] for m in rx_messages if m["timestamp"] is not None]
    print(f"\n-- {title}: windowed message-count cross-check --")
    if not tx_ts or not rx_ts:
        print("(nothing to compare - one side is empty)")
        return
    window_lo, window_hi = max(min(tx_ts), min(rx_ts)), min(max(tx_ts), max(rx_ts))
    if window_lo > window_hi:
        print("(no time overlap between the two captures)")
        return

    tx_types = Counter(m["msg_type"] for m in tx_messages if window_lo <= m["timestamp"] <= window_hi)
    rx_types = Counter(m["msg_type"] for m in rx_messages if window_lo <= m["timestamp"] <= window_hi)
    for msg_type in sorted(set(tx_types) | set(rx_types)):
        tx_n, rx_n = tx_types.get(msg_type, 0), rx_types.get(msg_type, 0)
        deficit = tx_n - rx_n
        pct = f"{100*deficit/tx_n:.1f}%" if tx_n else "n/a"
        flag = "" if deficit <= 0 else f"  <-- {pct} apparent shortfall"
        print(f"  {msg_type}: tx={tx_n} rx={rx_n}{flag}")


def find_duplicate_content_types(messages, unique_ratio_threshold=0.5, min_count=5):
    """Returns message types where distinct payloads are a minority of total
    instances - a static MAP or a fixed-content periodic status broadcast,
    as opposed to BSM/SPAT/SDSM where embedded position/time data makes each
    instance's payload effectively unique. Used to caption plots so a
    matching-ambiguity artifact for these types isn't mistaken for measured
    latency or a real drop."""
    by_type = {}
    for m in messages:
        by_type.setdefault(m["msg_type"], []).append(m["payload_hex"])
    flagged = []
    for msg_type, payloads in by_type.items():
        if len(payloads) < min_count:
            continue
        ratio = len(set(payloads)) / len(payloads)
        if ratio < unique_ratio_threshold:
            flagged.append(msg_type)
    return sorted(flagged)


def _style_axes(ax):
    ax.set_facecolor(CHART_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=SECONDARY_INK, labelsize=8)


def plot_flow(output_path, title, latencies, stale, drops, out_of_window, drop_threshold_ms,
              latency_label="Latency (ms)", duplicate_content_types=()):
    """Renders and saves the two-panel figure described in the module
    docstring. Returns True if a figure was written, False if there was
    nothing to plot (all four buckets empty).

    duplicate_content_types: message types (e.g. a static MAP, or a
    fixed-content periodic status broadcast) where the payload repeats
    byte-identically across many instances in this run, so a "stale"/
    "dropped" outcome for that type may just be per-instance matching
    ambiguity rather than real latency or loss - annotated as a caption
    rather than silently shown as if it were as reliable as a unique-content
    type's outcome."""
    all_entries = latencies + stale + drops + out_of_window
    if not all_entries:
        return False

    msg_types_order = sorted({e[1] for e in all_entries})
    type_color = {t: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, t in enumerate(msg_types_order)}
    y_pos = {t: i for i, t in enumerate(msg_types_order)}

    fig, (ax_latency, ax_status) = plt.subplots(
        2, 1, figsize=(11, 3 + 0.9 * len(msg_types_order)), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}, facecolor=CHART_SURFACE,
    )

    # Top panel: fresh-match latency over time, by message type.
    if latencies:
        for t in msg_types_order:
            pts = [(ts, val) for ts, mt, val in latencies if mt == t]
            if not pts:
                continue
            xs = [datetime.fromtimestamp(ts) for ts, _ in pts]
            ys = [val for _, val in pts]
            ax_latency.scatter(xs, ys, s=18, color=type_color[t], label=t, alpha=0.8,
                                edgecolors=CHART_SURFACE, linewidths=0.4)
        ax_latency.legend(loc="upper right", fontsize=8, framealpha=0.9,
                           ncol=min(len(msg_types_order), 4), labelcolor=SECONDARY_INK)
    else:
        ax_latency.text(0.5, 0.5, "No fresh matches", ha="center", va="center",
                         transform=ax_latency.transAxes, color=MUTED_INK)
    ax_latency.set_ylabel(latency_label, color=SECONDARY_INK, fontsize=9)
    ax_latency.set_title(title, fontsize=11, color=PRIMARY_INK, loc="left")
    ax_latency.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
    _style_axes(ax_latency)

    # Bottom panel: every tx candidate, plotted at its own timestamp on its
    # type's row, colored by outcome - drops show up as a visible red cluster
    # rather than only a percentage in text output.
    buckets = {"matched": latencies, "stale": stale, "dropped": drops, "out_of_window": out_of_window}
    for status in STATUS_ORDER:
        entries = buckets[status]
        if not entries:
            continue
        xs = [datetime.fromtimestamp(e[0]) for e in entries]
        ys = [y_pos[e[1]] + STATUS_Y_OFFSET[status] for e in entries]
        label = {
            "matched": "Matched (fresh)",
            "stale": f"Stale (> {drop_threshold_ms:.0f} ms)",
            "dropped": "Dropped",
            "out_of_window": "Outside recording window (not a real drop)",
        }[status]
        ax_status.scatter(xs, ys, s=22, color=STATUS_COLOR[status], label=label, alpha=0.85,
                           edgecolors=CHART_SURFACE, linewidths=0.3)
    ax_status.set_yticks(range(len(msg_types_order)))
    ax_status.set_yticklabels(msg_types_order, color=SECONDARY_INK, fontsize=9)
    ax_status.set_ylim(-0.6, len(msg_types_order) - 0.4)
    ax_status.set_xlabel("Time", color=SECONDARY_INK, fontsize=9)
    ax_status.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2, labelcolor=SECONDARY_INK)
    ax_status.grid(True, axis="x", color=GRIDLINE, linewidth=0.8)
    _style_axes(ax_status)

    fig.autofmt_xdate()
    fig.tight_layout()

    duplicate_content_types = [t for t in duplicate_content_types if t in y_pos]
    if duplicate_content_types:
        fig.subplots_adjust(bottom=0.16)
        fig.text(
            0.01, 0.01,
            f"Note: {', '.join(duplicate_content_types)} repeat byte-identical content across "
            "many instances in this run - stale/dropped outcomes above may reflect per-instance "
            "matching ambiguity rather than real latency or loss; cross-check the windowed count report.",
            fontsize=7, color=MUTED_INK, wrap=True,
        )

    fig.savefig(output_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)
    return True
