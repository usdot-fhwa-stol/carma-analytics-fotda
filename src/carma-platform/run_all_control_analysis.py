from pathlib import Path
from typing import Dict
from guidance_scripts import run_crosstrack_analysis, run_turn_accuracy_analysis
from run_all_analysis import run_all_analysis
import argparse
import argcomplete


def analyze_mcap_file_for_control_analysis(
    mcap_path: Path, output_dir: Path, data_dir: Path, plots_dir: Path
) -> Dict:
    """Extract single MCAP file and run all control analysis on it"""
    try:
        # 1. Cross_track analysis
        stats, _, _, _ = run_crosstrack_analysis(mcap_path, data_dir, plots_dir)

        # 2. Turn accuracy analysis by spline fitting
        stats, _, _, _ = run_turn_accuracy_analysis(mcap_path, data_dir, plots_dir)

        return stats
    except Exception as e:
        print(f"Error analyzing {mcap_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all control analysis on multiple MCAP files in a given directory"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing MCAP files to analyze",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base directory for saving analysis results (optional)",
        default=None,
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    try:
        run_all_analysis(
            args.input_dir, analyze_mcap_file_for_control_analysis, args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
