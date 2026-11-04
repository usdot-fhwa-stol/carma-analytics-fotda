"""Plots for the DT-WZ cascade, grouped by pedestrian dwell condition.

Three figures, each answering a different question:

* ``plot_latency_by_stage``  -- where does the time go? (boxplot per stage)
* ``plot_cascade``           -- how does one detection's journey look, and how
  much does it vary between detections? (dot cascade, one row per detection)
* ``plot_condition_comparison`` -- does dwell duration change anything?

Stages are an ordered sequence, so they are coloured with a single sequential
ramp (viridis) rather than a categorical palette: the colour then encodes
position in the pipeline, and a reader can see progression without consulting
the legend. Rainbow ramps are avoided -- their non-monotonic lightness invents
banding that is not in the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import config as cfg  # noqa: E402
from . import cascade_config as config  # noqa: E402

COLORMAP = cfg.STYLE.stage_colormap
# Full range: with ~10 stages this widens adjacent-pair OKLab separation from
# ~6.9 to ~7.9 (x100) at no cost. A sequential ramp this finely stepped still
# cannot carry identity by colour alone -- adjacent pairs fall to dE ~2.7 under
# deuteranopia -- so every figure here also encodes the stage by position, with
# axis tick labels on the boxplots and direct labels on the cascade.
CMAP_RANGE = cfg.STYLE.stage_cmap_range

# Cascade rows are thinned past this count to keep the figure a readable shape.
MAX_CASCADE_ROWS = cfg.LATENCY_CASCADE.max_cascade_rows
# Minimum gap between two direct labels, as a fraction of the plotted x-range.
LABEL_MIN_SEPARATION_FRAC = 0.055


# The suite's shared despine, re-exported under the name this module already
# uses so the cascade figures cannot drift from the rest.
_despine = cfg.despine


def stage_colors(stages, colormap=COLORMAP, cmap_range=CMAP_RANGE):
    cmap = matplotlib.colormaps[colormap]
    if len(stages) == 1:
        return [cmap(cmap_range[0])]
    return [
        cmap(cmap_range[0] + (cmap_range[1] - cmap_range[0]) * i / (len(stages) - 1))
        for i in range(len(stages))
    ]


def _latency_columns(table: pd.DataFrame, stages) -> List[str]:
    return [f"lat_{stage[2:]}_ms" for stage in stages if f"lat_{stage[2:]}_ms" in table.columns]


def available_stages(table: pd.DataFrame, stages=None) -> List[str]:
    """Milestone stages that carry at least one measurement in this table."""
    stages = stages or config.MILESTONE_STAGES
    return [
        stage for stage in stages
        if stage != "t_flir_detect"
        and f"lat_{stage[2:]}_ms" in table.columns
        and table[f"lat_{stage[2:]}_ms"].notna().any()
    ]


def plot_latency_by_stage(table: pd.DataFrame, title: str, output_path,
                          stages=None, mode="cumulative"):
    """Boxplot of latency at each stage.

    ``mode="cumulative"`` measures every stage from the camera detection, so the
    boxes march rightwards and the last one is the end-to-end total.
    ``mode="delta"`` shows each hop's own cost instead, which is what to read
    when looking for the expensive hop.
    """
    stages = available_stages(table, stages)
    if not stages:
        return None

    if mode == "delta":
        ordered = [stage for stage in (config.MILESTONE_STAGES) if stage in stages or stage == "t_flir_detect"]
        labels, series = [], []
        for earlier, later in zip(ordered, ordered[1:]):
            values = (table[later] - table[earlier]).dropna() if (
                earlier in table.columns and later in table.columns
            ) else pd.Series(dtype=float)
            if values.empty:
                continue
            labels.append(
                f"{config.STAGE_SHORT_LABEL.get(earlier, earlier)}→"
                f"{config.STAGE_SHORT_LABEL.get(later, later)}"
            )
            series.append(values.to_numpy())
        xlabel = "Latency of this hop (ms)"
    else:
        labels, series = [], []
        for stage in stages:
            values = table[f"lat_{stage[2:]}_ms"].dropna()
            if values.empty:
                continue
            labels.append(config.STAGE_SHORT_LABEL.get(stage, stage))
            series.append(values.to_numpy())
        xlabel = "Latency since camera detection (ms)"

    if not series:
        return None

    figure, axis = plt.subplots(figsize=(11, 0.52 * len(series) + 2.4))
    colors = stage_colors(range(len(series)))
    boxes = axis.boxplot(
        series, orientation="horizontal", patch_artist=True, widths=0.62,
        tick_labels=labels, showfliers=True,
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35,
                    "markerfacecolor": "#555555", "markeredgecolor": "none"},
        medianprops={"color": "white", "linewidth": 1.6},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.2)
    for element in ("whiskers", "caps"):
        for artist in boxes[element]:
            artist.set_color("#666666")

    # Label each median just past its upper whisker, vertically centred on the
    # box. Placing it at the median itself would sit the text on top of the box
    # and the white median line, which is where it becomes unreadable.
    span = max(np.max(values) for values in series) or 1.0
    pad = span * 0.012
    for index, values in enumerate(series, start=1):
        whisker_end = boxes["whiskers"][2 * index - 1].get_xdata()[1]
        axis.text(
            whisker_end + pad, index, f"{np.median(values):.0f} ms",
            fontsize=7.5, color="#333333", ha="left", va="center",
            # The outlier cloud extends past the whisker on several stages, so
            # the label needs a backing to stay legible on top of it.
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )

    # Clip to the whisker extent. A few multi-second outliers -- a camera stall
    # puts detections 3 s late -- otherwise stretch the axis by 20x and flatten
    # every box into an invisible sliver. The count beyond the axis is stated in
    # the subtitle so nothing is hidden, only moved out of the way.
    upper = max(
        boxes["whiskers"][2 * index - 1].get_xdata()[1] for index in range(1, len(series) + 1)
    )
    lower = min(
        boxes["whiskers"][2 * index - 2].get_xdata()[1] for index in range(1, len(series) + 1)
    )
    beyond = sum(int((values > upper).sum() + (values < lower).sum()) for values in series)
    headroom = (upper - lower) * 0.22 or 1.0
    axis.set_xlim(min(lower - headroom * 0.1, 0), upper + headroom)

    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    subtitle = f"n = {len(table)} detections"
    if beyond:
        subtitle += f"   ·   {beyond} points beyond axis, up to {max(v.max() for v in series):,.0f} ms"
    axis.set_title(f"{title}\n{subtitle}", loc="left", fontsize=11)
    cfg.grid(axis, "x")
    _despine(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=cfg.STYLE.figure_dpi)
    plt.close(figure)
    return output_path


def plot_cascade(table: pd.DataFrame, title: str, output_path,
                 stages=None, row_step=1, row_spacing=1.0, marker_size=3.0):
    """One horizontal line of dots per detection, sorted fastest to slowest.

    Sorting by end-to-end latency turns what would be a noisy band into a
    monotone sweep, which makes it obvious *which* hop widens on the slow rows
    rather than merely that some rows are slow.
    """
    stages = available_stages(table, stages)
    columns = _latency_columns(table, stages)
    if not columns:
        return None

    frame = table[columns].dropna(how="all").copy()
    if frame.empty:
        return None
    frame = frame.sort_values(columns[-1], na_position="last").reset_index(drop=True)
    # Keep the figure a readable shape regardless of run length: past ~MAX_ROWS
    # the dots are denser than the output resolution anyway, so thinning them
    # costs no visible detail and avoids a metre-tall image.
    if row_step <= 1 and len(frame) > MAX_CASCADE_ROWS:
        row_step = int(np.ceil(len(frame) / MAX_CASCADE_ROWS))
    if row_step > 1:
        frame = frame.iloc[::row_step].reset_index(drop=True)

    colors = stage_colors(range(len(columns)))
    height = min(11.0, max(3.5, 0.019 * len(frame) * row_spacing + 2.4))
    figure, axis = plt.subplots(figsize=(11, height))
    y = np.arange(len(frame)) * row_spacing

    for column, color, stage in zip(columns, colors, stages):
        axis.scatter(
            frame[column].to_numpy(), y, s=marker_size, color=color,
            label=config.STAGE_SHORT_LABEL.get(stage, stage),
            edgecolors="none", alpha=0.85,
        )

    axis.set_xlabel("Latency since camera detection (ms)")
    axis.set_ylabel(f"Detections, sorted by end-to-end latency (n = {len(frame)})")
    axis.set_title(title, loc="left", fontsize=11, pad=26)
    axis.set_yticks([])
    cfg.grid(axis, "x")
    _despine(axis)
    axis.legend(
        loc="lower right", frameon=False, fontsize=8, markerscale=3,
        ncol=2, handletextpad=0.3,
    )

    # Direct labels along the top, so identity never rests on colour alone --
    # ten steps of a sequential ramp put adjacent pairs well below the contrast
    # needed to tell them apart, especially for a red-green colourblind reader.
    #
    # Only stages far enough apart to place text without overlapping get one.
    # The early hops are a few milliseconds apart and their labels would collide
    # into an unreadable smear; those stay identified by the legend, which is
    # itself ordered along the pipeline.
    medians = [float(frame[column].median()) for column in columns]
    span = max(medians) - min(medians) or 1.0
    ticks, tick_labels, tick_colors = [], [], []
    for median, stage, color in zip(medians, stages, colors):
        if ticks and median - ticks[-1] < span * LABEL_MIN_SEPARATION_FRAC:
            continue
        ticks.append(median)
        tick_labels.append(config.STAGE_SHORT_LABEL.get(stage, stage))
        tick_colors.append(color)

    top = axis.secondary_xaxis("top")
    top.set_xticks(ticks)
    top.set_xticklabels(tick_labels, rotation=30, ha="left", fontsize=7)
    top.tick_params(axis="x", length=2, pad=1)
    top.spines["top"].set_visible(False)
    for label, color in zip(top.get_xticklabels(), tick_colors):
        label.set_color(color)
    figure.tight_layout()
    figure.savefig(output_path, dpi=cfg.STYLE.figure_dpi)
    plt.close(figure)
    return output_path


def plot_condition_comparison(tables: Dict[str, pd.DataFrame], title: str, output_path,
                              stages=None):
    """End-to-end latency distribution per dwell condition, side by side.

    This is the figure that answers the experiment's own question: dwell duration
    is the independent variable, so if it has no effect these three boxes should
    be indistinguishable.
    """
    stages = stages or config.MILESTONE_STAGES
    labels, series = [], []
    for condition, table in tables.items():
        if table.empty:
            continue
        final = [
            f"lat_{stage[2:]}_ms" for stage in reversed(stages)
            if f"lat_{stage[2:]}_ms" in table.columns
            and table[f"lat_{stage[2:]}_ms"].notna().any()
        ]
        if not final:
            continue
        values = table[final[0]].dropna()
        if values.empty:
            continue
        labels.append(f"{condition}\n(n = {len(values)})")
        series.append(values.to_numpy())

    if not series:
        return None

    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    colors = stage_colors(range(len(series)))
    boxes = axis.boxplot(
        series, patch_artist=True, widths=0.5, tick_labels=labels, showfliers=True,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35,
                    "markerfacecolor": "#555555", "markeredgecolor": "none"},
        medianprops={"color": "white", "linewidth": 1.6},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.2)
    for element in ("whiskers", "caps"):
        for artist in boxes[element]:
            artist.set_color("#666666")
    # Label above each upper whisker: inside the box the text lands on the white
    # median line against a dark fill, where it is unreadable.
    for index, values in enumerate(series, start=1):
        whisker_end = boxes["whiskers"][2 * index - 1].get_ydata()[1]
        axis.text(index, whisker_end, f"{np.median(values):.0f} ms median\n",
                  fontsize=8, color="#333333", ha="center", va="bottom")

    axis.set_ylabel("End-to-end latency (ms)")
    axis.set_xlabel("Pedestrian dwell condition")
    axis.set_title(title, loc="left", fontsize=11)
    cfg.grid(axis, "y")
    _despine(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=cfg.STYLE.figure_dpi)
    plt.close(figure)
    return output_path


def plot_drop_rates(summary: pd.DataFrame, title: str, output_path,
                    columns=None,
                    threshold_pct: float = cfg.DETECTION_DELIVERY.max_drop_rate_pct):
    """Per-run drop rates for the two drop metrics, grouped by condition.

    ``columns`` names the drop-rate columns to plot, and ``threshold_pct``
    draws the acceptance line. The two delivery measurements share a limit, so
    one line serves both; pass a different one if they ever diverge.
    """
    columns = columns or cfg.LATENCY_CASCADE.drop_rate_columns
    present = [(column, label) for column, label in columns if column in summary.columns]
    if not present or summary.empty:
        return None

    figure, axis = plt.subplots(figsize=(9.5, 4.4))
    labels = summary["run"].tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(present)
    colors = stage_colors(range(max(len(present), 2)))

    for index, ((column, label), color) in enumerate(zip(present, colors)):
        values = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)
        axis.bar(x + index * width - 0.4 + width / 2, values, width=width * 0.9,
                 label=label, color=color, edgecolor="white", linewidth=0.8)

    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Drop rate (%)")
    axis.set_title(title, loc="left", fontsize=11)
    cfg.threshold_line(axis, threshold_pct, f"{threshold_pct:g}% threshold",
                       right_at=len(labels) - 0.4)
    cfg.grid(axis, "y")
    _despine(axis)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=cfg.STYLE.figure_dpi)
    plt.close(figure)
    return output_path
