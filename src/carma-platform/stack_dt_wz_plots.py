from pathlib import Path
import argparse
import os

import argcomplete
from PIL import Image

# The two per-mcap plots produced by run_all_dt_wz_analysis.py (CP-helper, CP02) that this script
# stacks together so gaps in the SDSM interval plot (CP-helper) can be visually compared against
# dropped detections in the drop-rate plot (CP02) on the same time axis.
CP_HELPER_INTERVAL_PLOT_NAME = "_message_incoming_sdsm_message_intervals.png"
CP02_DROP_RATE_PLOT_NAME = "sdsm_detection_drop_rate_analysis.png"
STACKED_PLOT_NAME = "cp_helper_cp02_stacked.png"


def stack_plots_vertically(top_image_path: Path, bottom_image_path: Path, output_path: Path) -> None:
    """
    Stacks two PNGs vertically into one image, matching widths (scaling the narrower image up)
    so that a shared time axis lines up between the two.

    Args:
        top_image_path: Path to the PNG to place on top (CP-helper SDSM message interval plot)
        bottom_image_path: Path to the PNG to place on the bottom (CP02 drop rate plot)
        output_path: Path to save the combined PNG to
    """
    top_image = Image.open(top_image_path)
    bottom_image = Image.open(bottom_image_path)

    width = max(top_image.width, bottom_image.width)
    if top_image.width != width:
        top_image = top_image.resize((width, round(top_image.height * width / top_image.width)))
    if bottom_image.width != width:
        bottom_image = bottom_image.resize((width, round(bottom_image.height * width / bottom_image.width)))

    stacked_image = Image.new("RGB", (width, top_image.height + bottom_image.height), "white")
    stacked_image.paste(top_image, (0, 0))
    stacked_image.paste(bottom_image, (0, top_image.height))
    stacked_image.save(output_path)


def stack_all_dt_wz_plots(analysis_dir: Path) -> list:
    """
    Finds every directory under analysis_dir containing both CP_HELPER_INTERVAL_PLOT_NAME and
    CP02_DROP_RATE_PLOT_NAME (i.e. each mcap's plots/ directory from run_all_dt_wz_analysis.py)
    and saves a vertically stacked copy alongside them.

    Args:
        analysis_dir: Directory to search (e.g. a dt_wz_analysis_<timestamp> output directory)

    Returns:
        List of paths to the stacked images that were created
    """
    stacked_paths = []
    for root, _, files in os.walk(analysis_dir):
        if CP_HELPER_INTERVAL_PLOT_NAME in files and CP02_DROP_RATE_PLOT_NAME in files:
            root = Path(root)
            output_path = root / STACKED_PLOT_NAME
            stack_plots_vertically(root / CP_HELPER_INTERVAL_PLOT_NAME, root / CP02_DROP_RATE_PLOT_NAME, output_path)
            print(f"Stacked plot saved to: {output_path}")
            stacked_paths.append(output_path)

    if not stacked_paths:
        print(f"No {CP_HELPER_INTERVAL_PLOT_NAME} + {CP02_DROP_RATE_PLOT_NAME} pairs found under {analysis_dir}")

    return stacked_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stack the CP-helper SDSM message interval plot and CP02 drop rate plot for "
        "each analyzed MCAP so drops can be visually checked against interval gaps/latency"
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Directory to search for per-mcap plots (a run_all_dt_wz_analysis.py output directory)",
        required=True,
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    try:
        stack_all_dt_wz_plots(args.analysis_dir)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
