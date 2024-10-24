import os
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Tuple, List, Optional, Callable, Any


def find_mcap_files(input_dir: Path) -> List[Path]:
    """Find all MCAP files in the input directory and its subdirectories."""
    mcap_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mcap"):
                mcap_files.append(Path(root) / file)
    return mcap_files


def create_output_structure(base_dir: Path) -> Tuple[Path, Path, Path]:
    """Create output directory structure with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"analysis_{timestamp}"
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"

    # Create directories
    for dir_path in [output_dir, data_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return output_dir, data_dir, plots_dir


def run_all_analysis(
    input_dir: Path,
    analysis_func: Callable[[Path, Path, Path, Path], Dict[str, Any]],
    output_base_dir: Optional[Path] = None,
    analysis_name: str = "analysis",
) -> None:
    """
    Run analysis on all MCAP files in the input directory using a custom analysis function.

    Args:
        input_dir (Path): Directory containing MCAP files to analyze
        analysis_func (Callable): Function that performs analysis on a single MCAP file
                                Should accept (mcap_file, output_dir, data_dir, plots_dir)
        output_base_dir (Optional[Path]): Base directory for saving results
        analysis_name (str): Name of the analysis for directory naming
    """
    # Find all MCAP files
    mcap_files = find_mcap_files(input_dir)
    if not mcap_files:
        raise ValueError(f"No MCAP files found in {input_dir}")

    # Create output directory structure
    output_base_dir = output_base_dir or input_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"{analysis_name}_{timestamp}"
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"

    # Create directories
    for dir_path in [output_dir, data_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCreated output directories:")
    print(f"- Main output dir: {output_dir}")
    print(f"- Data dir: {data_dir}")
    print(f"- Plots dir: {plots_dir}")

    # Analyze each file
    results = {}
    for mcap_file in mcap_files:
        print(f"\nAnalyzing {mcap_file}...")
        try:
            results[str(mcap_file)] = analysis_func(
                mcap_file, output_dir, data_dir, plots_dir
            )
        except Exception as e:
            print(f"Error analyzing {mcap_file}: {e}")
            results[str(mcap_file)] = None

    # Create summary report
    summary = {
        "analysis_time": datetime.now().isoformat(),
        "analysis_type": analysis_name,
        "files_analyzed": len(mcap_files),
        "files_succeeded": sum(1 for r in results.values() if r is not None),
        "output_location": str(output_dir),
        "data_directory": str(data_dir),
        "plots_directory": str(plots_dir),
        "failed_files": [
            str(file) for file, result in results.items() if result is None
        ],
    }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete. Results saved to:")
    print(f"- Data files: {data_dir}")
    print(f"- Plot files: {plots_dir}")
    print(f"- Summary: {summary_path}")

    if summary["failed_files"]:
        print("\nWarning: Some files failed analysis:")
        for failed_file in summary["failed_files"]:
            print(f"- {failed_file}")
