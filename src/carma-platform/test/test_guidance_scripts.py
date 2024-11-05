import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from guidance_scripts import (
    get_engage_time,
    run_crosstrack_analysis,
    run_turn_accuracy_analysis,
    run_acceleration_comfort_analysis,
    calculate_instant_acceleration,
    calculate_window_average,
)
from pytest import approx
import numpy as np

"""
Usage:
cd carma-analytics-fotda/src/carma-platform/
python3 -m pytest test
"""


@pytest.fixture
def mock_mcap_path():
    return Path("/path/to/mock.mcap")


def test_get_engage_time(mock_mcap_path):
    STARTUP = 1
    ACTIVE = 3
    ENGAGED = 4
    INACTIVE = 5
    SHUTDOWN = 0
    with patch("guidance_scripts.extract_mcap_data") as mock_extract:
        mock_extract.return_value = {
            "/guidance/state": (
                [0, 1, 2, 3, 4, 5],
                [STARTUP, ACTIVE, ENGAGED, ENGAGED, INACTIVE, SHUTDOWN],
            )
        }
        start_time, end_time = get_engage_time(mock_mcap_path)
        assert start_time == 2
        assert end_time == 4


def test_run_crosstrack_analysis(mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract, patch(
        "guidance_scripts.plt"
    ) as mock_plt:
        mock_extract.return_value = {
            "/guidance/route_state": (
                [0, 1, 2, 3],
                [1.0, 1.5, 0.5, 1.8],
            )
        }
        mock_plt.figure.return_value = MagicMock()

        is_passed, stats, plot_figure, cross_tracks, timestamps = (
            run_crosstrack_analysis(
                mock_mcap_path,
                error_threshold_to_pass_meter=2.0,
                start_time=0,
                end_time=3,
                save_stats_dir=None,
                save_data_dir=None,
                save_plot_dir=None,
            )
        )

        assert is_passed == True
        assert stats["minimum"] == approx(0.5000, rel=1e-2)
        assert stats["maximum"] == approx(1.8000, rel=1e-2)
        assert stats["median"] == approx(1.2500, rel=1e-2)
        assert plot_figure is not None
        assert cross_tracks == approx([1.0, 1.5, 0.5, 1.8])
        assert timestamps == approx([0, 1, 2, 3])


def test_run_turn_accuracy_analysis(mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract, patch(
        "guidance_scripts.plt"
    ) as mock_plt:
        mock_extract.return_value = {
            "/localization/current_pose": (
                [0, 1, 2],
                [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            ),
            "/guidance/plan_trajectory": ([0], [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]]),
        }
        mock_plt.figure.return_value = MagicMock()

        is_passed, stats, plot_figure, actual_path, planned_path, distances, timestamps = (
            run_turn_accuracy_analysis(
                mock_mcap_path,
                error_threshold_to_pass_meter=2.0,
                start_time=0,
                end_time=2,
            )
        )

        expected_actual_path = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        expected_planned_path = [[0.0, 0.0], [1.0, 1.0]]

        assert is_passed == True
        assert stats["minimum"] == approx(0.0, abs=1e-2)
        assert plot_figure is not None
        assert len(distances) == 3
        assert timestamps == approx([0, 1, 2])

        # Check that each point in actual_path matches expected_actual_path approximately
        for actual, expected in zip(actual_path, expected_actual_path):
            assert actual == approx(expected)

        # Check that each point in planned_path matches expected_planned_path approximately
        for planned, expected in zip(planned_path, expected_planned_path):
            assert planned == approx(expected)


def test_calculate_instant_acceleration():
    timestamps = np.array([0, 1, 2, 3])
    speeds = np.array([0, 1, 4, 9])

    accelerations, time_points = calculate_instant_acceleration(timestamps, speeds)

    assert len(accelerations) == len(time_points)
    assert len(accelerations) == len(timestamps) - 1
    assert accelerations == approx([1, 3, 5])
    assert time_points == approx([1, 2, 3])

def test_calculate_window_average():
    """Test the calculate_window_average function for averaging values"""
    timestamps = np.array([0, 0.5, 1.0, 1.5, 2.0])
    values = np.array([0, 1, 2, 3, 4])

    # Test average calculation
    avg_values, avg_timestamps = calculate_window_average(
        timestamps, values, window_size=1.0
    )

    assert len(avg_values) == len(avg_timestamps)
    # First window should include values [0, 1, 2] as they're within 1 second of t=0
    assert avg_values[0] == approx(1.0)  # Average of [0, 1, 2] in windows_size 1.0
    # Timestamps should start from original timestamps
    assert avg_timestamps[0] == approx(0.0)


def test_run_acceleration_comfort_analysis(mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract, patch(
        "guidance_scripts.plt"
    ) as mock_plt:
        # Mock plt.subplots to return a tuple of (figure, (ax1, ax2))
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        # Use more data points with 0.1s intervals to ensure window calculations work
        timestamps = np.arange(0, 3.1, 0.1)  # 0 to 3 seconds with 0.1s intervals
        speeds = np.array([
            0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,
            2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 2.8, 2.6, 2.4, 2.2,
            2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0
        ])

        mock_extract.return_value = {
            "/hardware_interface/vehicle_status": (timestamps, speeds)
        }

        (is_passed, instant_stats, avg_stats, plot_figure,
         accelerations, avg_accelerations, time_points, avg_timepoints) = (
            run_acceleration_comfort_analysis(
                mock_mcap_path,
                comfort_deceleration_threshold_to_pass=3.0,
                start_time=0,
                end_time=3,
            )
        )

        # Test instant acceleration stats
        assert is_passed == True  # accelerations should be within comfort threshold
        assert instant_stats["minimum"] == approx(-2.0, rel=1e-1)
        assert instant_stats["maximum"] == approx(2.0, rel=1e-1)
        assert instant_stats["mean"] == approx(0.0, abs=1e-1)

        # Test average acceleration stats
        assert len(avg_accelerations) > 0
        assert avg_stats["minimum"] == approx(-2.0, rel=1e-1)
        assert avg_stats["maximum"] == approx(2.0, rel=1e-1)
        assert avg_stats["mean"] == approx(-0.6, abs=1e-1)

        # Test plotting was called correctly
        assert plot_figure is not None
        mock_plt.subplots.assert_called_once_with(2, 1, figsize=(12, 12))

        # Test output lengths and types
        assert len(accelerations) == len(timestamps) - 1
        assert len(time_points) == len(timestamps) - 1
        assert len(avg_accelerations) > 0
        assert len(avg_timepoints) > 0
