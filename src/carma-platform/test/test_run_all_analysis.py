import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from run_all_analysis import find_mcap_files, run_all_analysis
from datetime import datetime
import json

"""
Usage: 
cd carma-analytics-fotda/src/carma-platform/
python3 -m pytest test
"""


def test_find_mcap_files(tmp_path):
    # Create mock MCAP files
    (tmp_path / "file1.mcap").touch()
    (tmp_path / "file2.mcap").touch()
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file3.mcap").touch()

    mcap_files = find_mcap_files(tmp_path)
    assert len(mcap_files) == 3
    assert all(file.suffix == ".mcap" for file in mcap_files)


def test_run_all_analysis(tmp_path):
    # Mock analysis function
    def mock_analysis(mcap_path, output_dir, stats_dir, data_dir, plots_dir):
        return {
            "mock_metric_passed": True,
            "mock_metric_failed": False,
            "mock_metric_error": None,
        }

    # Create mock MCAP file
    (tmp_path / "test.mcap").touch()

    # Create a fixed datetime for testing
    mock_date = datetime(2024, 1, 1)

    with patch("run_all_analysis.datetime") as mock_datetime:
        # Configure the mock to return a real datetime object
        mock_datetime.now.return_value = mock_date

        run_all_analysis(tmp_path, mock_analysis, tmp_path / "output", "mock_analysis")

    # Verify directory structure
    output_dir = tmp_path / "output" / "mock_analysis_20240101_000000"
    assert output_dir.exists()

    # Read and verify the JSON content
    summary_file = output_dir / "analysis_summary.json"
    assert summary_file.exists()

    with open(summary_file) as f:
        summary = json.load(f)

    # Verify the summary content
    assert summary["analysis_type"] == "mock_analysis"
    assert summary["total_files_analyzed"] == 1
    assert "metrics_summary" in summary
    assert "analyzed_files" in summary
    assert "test.mcap" in summary["analyzed_files"]
    assert summary["analyzed_files"]["test.mcap"]["metrics_results"] == {
        "mock_metric_passed": True,
        "mock_metric_failed": False,
        "mock_metric_error": None,
    }


def test_run_all_analysis_no_mcap_files(tmp_path):
    with pytest.raises(ValueError, match="No MCAP files found"):
        run_all_analysis(tmp_path, lambda: None)
