import os
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Tuple, List, Optional, Callable, Any
from collections import defaultdict


def find_mcap_files(input_dir: Path) -> List[Path]:
    """Find all MCAP files in the input directory and its subdirectories."""
    mcap_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mcap"):
                mcap_files.append(Path(root) / file)
    return mcap_files


def create_output_directories(output_dir: Path, file_name: str) -> Tuple[Path, Path, Path, Path]:
    """Create per-file directory structure for analysis results."""
    file_output_dir = output_dir / file_name
    file_stats_dir = file_output_dir / "stats"
    file_data_dir = file_output_dir / "data"
    file_plots_dir = file_output_dir / "plots"

    for dir_path in [file_output_dir, file_stats_dir, file_data_dir, file_plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return file_output_dir, file_stats_dir, file_data_dir, file_plots_dir


def analyze_mcap_file(
    mcap_file: Path,
    analysis_func: Callable[[Path, Path, Path, Path, Path], Dict[str, Optional[bool]]],
    output_dir: Path
) -> Dict[str, Optional[bool]]:
    """Analyze a single MCAP file and return the results."""
    file_name = mcap_file.stem
    file_output_dir, file_stats_dir, file_data_dir, file_plots_dir = create_output_directories(output_dir, file_name)

    try:
        result = analysis_func(mcap_file, file_output_dir, file_stats_dir, file_data_dir, file_plots_dir)
        if not isinstance(result, dict):
            print(f"Warning: Analysis result for {mcap_file} is not a dictionary")
            return None
        return result
    except Exception as e:
        print(f"Error analyzing {mcap_file}: {e}")
        return None


def update_metrics_results(result: Dict[str, Optional[bool]], metrics_results: defaultdict) -> None:
    """Update metrics results based on the analysis result."""
    for metric, passed in result.items():
        if passed is None:
            metrics_results[metric]["errors"] += 1
        elif passed:
            metrics_results[metric]["passed"] += 1
        else:
            metrics_results[metric]["failed"] += 1


def create_summary(mcap_files: List[Path], results: defaultdict, metrics_results: defaultdict, analysis_name: str, output_dir: Path) -> None:
    """Create a summary report of the analysis."""
    summary = {
        "analysis_time": datetime.now().isoformat(),
        "analysis_type": analysis_name,
        "total_files_analyzed": len(mcap_files),
        "metrics_summary": {
            metric: {
                "total_files": len(mcap_files),
                "passed": results["passed"],
                "failed": results["failed"],
                "errors": results["errors"],
                "pass_rate": f"{results['passed']/len(mcap_files):.2%}",
                "error_rate": f"{results['errors']/len(mcap_files):.2%}",
            }
            for metric, results in metrics_results.items()
        },
        "analyzed_files": {
            mcap_file.name: {
                "output_dir": str(output_dir / mcap_file.stem),
                "metrics_results": results[str(mcap_file)],
            }
            for mcap_file in mcap_files
        },
    }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print metrics summary for convenience
    print("\nMetrics Summary:")
    for metric, results in metrics_results.items():
        print(f"\n{metric}:")
        print(f"- Passed: {results['passed']} ({results['passed']/len(mcap_files):.2%})")
        print(f"- Failed: {results['failed']} ({results['failed']/len(mcap_files):.2%})")
        if results["errors"] > 0:
            print(f"- Errors: {results['errors']} ({results['errors']/len(mcap_files):.2%})")

    print(f"\nAnalysis complete. Summary saved to: {summary_path}")


def run_all_analysis(
    input_dir: Path,
    analysis_func: Callable[[Path, Path, Path, Path], Dict[str, Optional[bool]]],
    output_base_dir: Optional[Path] = None,
    analysis_name: str = "analysis",
) -> None:
    """
    Run analysis on all MCAP files in the input directory using a custom analysis function.
    Creates separate directories for each MCAP file.

    Args:
        input_dir (Path): Directory containing MCAP files to analyze
        analysis_func (Callable): Function that performs analysis on a single MCAP file
                                Should accept (mcap_file, output_dir, stats_dir, data_dir, plots_dir)
                                Should return Dict[str, Optional[bool]] where True means pass, None is error
        output_base_dir (Optional[Path]): Base directory for saving results
        analysis_name (str): Name of the analysis for directory naming
    """
    # Find all MCAP files
    mcap_files = find_mcap_files(input_dir)
    if not mcap_files:
        raise ValueError(f"No MCAP files found in {input_dir}")

    # Create main output directory structure
    output_base_dir = output_base_dir or input_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"{analysis_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreated main output directory: {output_dir}")

    # Analyze each file
    results = {}
    metrics_results = defaultdict(lambda: {"passed": 0, "failed": 0, "errors": 0})

    for mcap_file in mcap_files:
        print(f"\nAnalyzing {mcap_file}...")
        result = analyze_mcap_file(mcap_file, analysis_func, output_dir)
        results[str(mcap_file)] = result

        if result is not None:
            update_metrics_results(result, metrics_results)

    # Create summary report
    create_summary(mcap_files, results, metrics_results, analysis_name, output_dir)