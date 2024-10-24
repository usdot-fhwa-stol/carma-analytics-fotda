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


def run_all_analysis(
    input_dir: Path,
    analysis_func: Callable[[Path, Path, Path, Path], Dict[str, Any]],
    output_base_dir: Optional[Path] = None,
    analysis_name: str = "analysis",
) -> None:
    """
    Run analysis on all MCAP files in the input directory using a custom analysis function.
    Creates separate directories for each MCAP file.

    Args:
        input_dir (Path): Directory containing MCAP files to analyze
        analysis_func (Callable): Function that performs analysis on a single MCAP file
                                Should accept (mcap_file, output_dir, data_dir, plots_dir)
                                Should return Dict[str, Optional[Dict[str, Any]]]
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
    for mcap_file in mcap_files:
        print(f"\nAnalyzing {mcap_file}...")

        # Create per-file directory structure
        file_name = mcap_file.stem
        file_output_dir = output_dir / file_name
        file_data_dir = file_output_dir / "data"
        file_plots_dir = file_output_dir / "plots"

        # Create per-file directories
        for dir_path in [file_output_dir, file_data_dir, file_plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Created directories for {file_name}:")
        print(f"- Output dir: {file_output_dir}")
        print(f"- Data dir: {file_data_dir}")
        print(f"- Plots dir: {file_plots_dir}")

        try:
            result = analysis_func(
                mcap_file, file_output_dir, file_data_dir, file_plots_dir
            )
            if not isinstance(result, dict):
                print(f"Warning: Analysis result for {mcap_file} is not a dictionary")
                results[str(mcap_file)] = None
            else:
                results[str(mcap_file)] = result
        except Exception as e:
            print(f"Error analyzing {mcap_file}: {e}")
            results[str(mcap_file)] = None

    # Create summary report
    summary = {
        "analysis_time": datetime.now().isoformat(),
        "analysis_type": analysis_name,
        "files_analyzed": len(mcap_files),
        "analyzed_files": {
            mcap_file.name: {
                "output_dir": str(output_dir / mcap_file.stem),
                "data_dir": str(output_dir / mcap_file.stem / "data"),
                "plots_dir": str(output_dir / mcap_file.stem / "plots"),
                "analysis_results": {} if results[str(mcap_file)] is None else {},
            }
            for mcap_file in mcap_files
        },
    }

    # Process analysis results for summary
    failed_analyses = defaultdict(list)
    successful_analyses = defaultdict(list)

    for mcap_file in mcap_files:
        result = results[str(mcap_file)]
        if result is None:
            summary["analyzed_files"][mcap_file.name]["analysis_results"] = None
            failed_analyses["all_failed"].append(mcap_file.name)
        else:
            # Process each analysis type in the result
            for analysis_type, analysis_result in result.items():
                if analysis_result is None:
                    failed_analyses[analysis_type].append(mcap_file.name)
                else:
                    successful_analyses[analysis_type].append(mcap_file.name)
                summary["analyzed_files"][mcap_file.name]["analysis_results"][
                    analysis_type
                ] = analysis_result

    # Add success/failure summaries
    summary["analysis_summary"] = {
        "total_files": len(mcap_files),
        "completely_failed_files": len(failed_analyses["all_failed"]),
        "analysis_types": {
            analysis_type: {
                "successful_files": len(successful_files),
                "failed_files": len(failed_analyses.get(analysis_type, [])),
                "success_rate": f"{len(successful_files)/(len(successful_files) + len(failed_analyses.get(analysis_type, []))):.2%}",
            }
            for analysis_type, successful_files in successful_analyses.items()
        },
    }

    # Add detailed failure information
    if failed_analyses:
        summary["failures"] = {
            analysis_type: file_list
            for analysis_type, file_list in failed_analyses.items()
            if file_list  # Only include analysis types that had failures
        }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete. Summary saved to: {summary_path}")

    # Print failure summary
    if failed_analyses:
        print("\nAnalysis Failures Summary:")
        for analysis_type, failed_files in failed_analyses.items():
            if analysis_type == "all_failed":
                print(f"\nCompletely failed files: {len(failed_files)}")
            else:
                print(f"\nFailed {analysis_type}: {len(failed_files)} files")
                print(
                    f"Success rate: {summary['analysis_summary']['analysis_types'][analysis_type]['success_rate']}"
                )
