import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from guidance_scripts import *
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

        (
            is_passed,
            stats,
            plot_figure,
            actual_path,
            planned_path,
            distances,
            timestamps,
        ) = run_turn_accuracy_analysis(
            mock_mcap_path,
            error_threshold_to_pass_meter=2.0,
            start_time=0,
            end_time=2,
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
        speeds = np.array(
            [
                0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
                1.2,
                1.4,
                1.6,
                1.8,
                2.0,
                2.2,
                2.4,
                2.6,
                2.8,
                3.0,
                2.8,
                2.6,
                2.4,
                2.2,
                2.0,
                1.8,
                1.6,
                1.4,
                1.2,
                1.0,
                0.8,
                0.6,
                0.4,
                0.2,
                0,
            ]
        )

        mock_extract.return_value = {
            "/hardware_interface/vehicle_status": (timestamps, speeds)
        }

        (
            is_passed,
            instant_stats,
            avg_stats,
            plot_figure,
            accelerations,
            avg_accelerations,
            time_points,
            avg_timepoints,
        ) = run_acceleration_comfort_analysis(
            mock_mcap_path,
            comfort_deceleration_threshold_to_pass=3.0,
            start_time=0,
            end_time=3,
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


def test_calculate_instant_lateral_values():
    """Test the calculate_instant_lateral_values function with known values"""
    # Test data
    long_velocities = np.array([1.0, 2.0, 3.0, 4.0])
    ang_velocities = np.array([0.1, 0.2, 0.3, 0.4])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])

    # Calculate values
    lateral_acc, lateral_jerk, acc_timestamps, jerk_timestamps = (
        calculate_instant_lateral_values(long_velocities, ang_velocities, timestamps)
    )

    # Test output lengths
    assert len(lateral_acc) == len(timestamps)
    assert len(lateral_jerk) == len(timestamps) - 1
    assert len(acc_timestamps) == len(timestamps)
    assert len(jerk_timestamps) == len(timestamps) - 1

    # Test acceleration calculations (v * ω)
    expected_acc = np.array([0.1, 0.4, 0.9, 1.6])
    np.testing.assert_array_almost_equal(lateral_acc, expected_acc)

    # Test jerk calculations (Δacc/Δt)
    expected_jerk = np.array([0.3, 0.5, 0.7])
    np.testing.assert_array_almost_equal(lateral_jerk, expected_jerk)


def test_calculate_instant_lateral_values_zero_input():
    """Test with zero inputs"""
    long_velocities = np.zeros(3)
    ang_velocities = np.zeros(3)
    timestamps = np.array([0.0, 1.0, 2.0])

    lateral_acc, lateral_jerk, acc_timestamps, jerk_timestamps = (
        calculate_instant_lateral_values(long_velocities, ang_velocities, timestamps)
    )

    assert np.all(lateral_acc == 0)
    assert np.all(lateral_jerk == 0)


def test_run_lateral_analysis(mock_mcap_path):
    """Test the run_lateral_analysis function with mocked data"""
    with patch("guidance_scripts.extract_mcap_data") as mock_extract, patch(
        "guidance_scripts.plt"
    ) as mock_plt:

        # Mock timestamps and twist messages
        timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        velocity_data = [
            (0.5, 0.1),  # (linear.x, angular.z)
            (1.0, 0.2),
            (1.5, 0.3),
            (2.0, 0.4),
            (2.5, 0.5),
        ]

        # Setup mock return value
        mock_extract.return_value = {
            "/hardware_interface/vehicle/twist": (timestamps, velocity_data)
        }

        # Mock plt.subplots to return figures and axes
        mock_fig_acc = MagicMock()
        mock_fig_jerk = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax3 = MagicMock()
        mock_ax4 = MagicMock()
        mock_plt.subplots.side_effect = [
            (mock_fig_acc, (mock_ax1, mock_ax2)),
            (mock_fig_jerk, (mock_ax3, mock_ax4)),
        ]

        # Run analysis
        (
            is_passed,
            acc_inst_stats,
            acc_avg_stats,
            jerk_inst_stats,
            jerk_avg_stats,
            figures,
            lateral_acc_inst,
            lateral_acc_avg,
            lateral_jerk_inst,
            lateral_jerk_avg,
            timestamps_out,
        ) = run_lateral_analysis(
            mock_mcap_path, acc_threshold_to_pass=2.0, jerk_threshold_to_pass=2.0
        )

        # Test that values were calculated correctly
        assert len(lateral_acc_inst) == len(timestamps)
        assert len(lateral_jerk_inst) == len(timestamps) - 1

        # Test statistics were calculated
        assert acc_inst_stats["minimum"] == approx(0.05, rel=1e-2)
        assert acc_inst_stats["maximum"] == approx(1.25, rel=1e-2)

        # Test figure generation
        assert len(figures) == 2
        assert isinstance(figures[0], MagicMock)  # acc figure
        assert isinstance(figures[1], MagicMock)  # jerk figure

        # Test pass/fail criteria
        assert isinstance(is_passed, bool)

        # Test output timestamps
        np.testing.assert_array_equal(timestamps_out, timestamps)


def test_run_lateral_analysis_exceeds_threshold(mock_mcap_path):
    """Test run_lateral_analysis when values exceed comfort thresholds"""
    with patch("guidance_scripts.extract_mcap_data") as mock_extract, patch(
        "guidance_scripts.plt"
    ) as mock_plt:

        # Setup mock data with high acceleration/jerk values
        timestamps = np.array([0.0, 0.5, 1.0, 1.5])
        velocity_data = [
            (2.5, 1.0),  # Will create high lateral acceleration
            (5.0, 1.0),
            (10.0, 1.0),
            (15.0, 1.0),
        ]
        mock_extract.return_value = {
            "/hardware_interface/vehicle/twist": (timestamps, velocity_data)
        }

        # Mock figure creation
        mock_plt.subplots.side_effect = [
            (MagicMock(), (MagicMock(), MagicMock())),
            (MagicMock(), (MagicMock(), MagicMock())),
        ]

        # Run analysis with low thresholds
        result = run_lateral_analysis(
            mock_mcap_path,
            acc_threshold_to_pass=1.0,  # Low threshold
            jerk_threshold_to_pass=1.0,  # Low threshold
        )

        # Test that analysis failed due to exceeded thresholds
        assert result[0] == False  # is_passed should be False


def test_run_guidance_steering_analysis(mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract, patch(
        "guidance_scripts.plt"
    ) as mock_plt:

        # Mock timestamps and steering angles
        timestamps = np.array([0, 1, 2, 3, 4])
        cmd_angles = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        actual_angles = np.array([0.12, 0.19, 0.31, 0.18, 0.11])

        mock_extract.return_value = {
            "/guidance/ctrl_cmd": (timestamps, cmd_angles),
            "/hardware_interface/vehicle_status": (timestamps, actual_angles),
        }

        # Mock plt.figure to return a MagicMock
        mock_plt.figure.return_value = MagicMock()

        # Run analysis
        is_passed, stats, plot_figure, error_angles, common_timestamps = (
            run_guidance_steering_analysis(
                mock_mcap_path,
                error_threshold_to_pass_radian=0.1,
                start_time=0,
                end_time=4,
            )
        )

        # Test output values
        assert is_passed == True  # Error should be within threshold
        assert stats["minimum"] == approx(0.01, rel=1e-2)  # Minimum error
        assert stats["maximum"] == approx(0.02, rel=1e-2)  # Maximum error
        assert stats["median"] == approx(0.015, rel=1e-2)  # Median error

        # Test that arrays have correct lengths
        assert len(error_angles) == len(common_timestamps)

        # Test that plot was created
        assert plot_figure is not None
        mock_plt.figure.assert_called_once_with(figsize=(15, 10))


def test_run_guidance_steering_analysis_fails_threshold(mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract, patch(
        "guidance_scripts.plt"
    ) as mock_plt:

        # Mock data with large steering errors
        timestamps = np.array([0, 1, 2])
        cmd_angles = np.array([0.1, 0.2, 0.3])
        actual_angles = np.array([0.3, 0.4, 0.5])  # Large differences

        mock_extract.return_value = {
            "/guidance/ctrl_cmd": (timestamps, cmd_angles),
            "/hardware_interface/vehicle_status": (timestamps, actual_angles),
        }

        mock_plt.figure.return_value = MagicMock()

        # Run analysis with strict threshold
        is_passed, stats, _, _, _ = run_guidance_steering_analysis(
            mock_mcap_path, error_threshold_to_pass_radian=0.1  # Strict threshold
        )

        assert is_passed == False  # Should fail due to large errors
        assert stats["median"] > 0.1  # Median error should exceed threshold


def test_run_steering_wheel_analysis(mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract, patch(
        "guidance_scripts.plt"
    ) as mock_plt:

        # Mock timestamps and steering values
        timestamps = np.array([0, 1, 2, 3, 4])
        cmd_values = np.array([0.5, 1.0, 1.5, 1.0, 0.5])
        actual_values = np.array([0.48, 1.02, 1.47, 1.03, 0.51])

        mock_extract.return_value = {
            "/hardware_interface/as/pacmod/parsed_tx/steer_rpt": (
                timestamps,
                actual_values,
                cmd_values
            ),
        }

        mock_plt.figure.return_value = MagicMock()

        # Run analysis
        is_passed, stats, plot_figure, error_values, timestamps = (
            run_steering_wheel_analysis(
                mock_mcap_path, error_threshold_to_pass=0.1, start_time=0, end_time=4
            )
        )

        # Test output values
        assert is_passed == True  # Error should be within threshold
        assert stats["minimum"] == approx(0.02, rel=1e-2)  # Minimum error
        assert stats["maximum"] == approx(0.03, rel=1e-2)  # Maximum error
        assert stats["median"] == approx(0.025, rel=1e-2)  # Median error

        # Test that arrays have correct lengths
        assert len(error_values) == len(timestamps)

        # Test that plot was created
        assert plot_figure is not None
        mock_plt.figure.assert_called_once_with(figsize=(15, 10))


def test_run_steering_wheel_analysis_fails_threshold(mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract, patch(
        "guidance_scripts.plt"
    ) as mock_plt:

        # Mock data with large steering errors
        timestamps = np.array([0, 1, 2])
        cmd_values = np.array([0.5, 1.0, 1.5])
        actual_values = np.array([0.7, 1.2, 1.7])  # Large differences

        mock_extract.return_value = {
            "/hardware_interface/as/pacmod/parsed_tx/steer_rpt": (
                timestamps,
                actual_values,
                cmd_values
            )
        }

        mock_plt.figure.return_value = MagicMock()

        # Run analysis with strict threshold
        is_passed, stats, _, _, _ = run_steering_wheel_analysis(
            mock_mcap_path, error_threshold_to_pass=0.1  # Strict threshold
        )

        assert is_passed == False  # Should fail due to large errors
        assert stats["median"] > 0.1  # Median error should exceed threshold


def test_get_planner_trajectory_intervals(mock_mcap_path):
    # Mock the extract_mcap_data function
    with patch("guidance_scripts.extract_mcap_data") as mock_extract:
        mock_extract.return_value = {
            "/guidance/plan_trajectory": (
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [
                    ("guidance/plugins/inlanecruising_plugin"),
                    ("guidance/plugins/inlanecruising_plugin"),
                    ("guidance/plugins/inlanecruising_plugin"),
                    ("guidance/plugins/cooperative_lanechange"),
                    ("guidance/plugins/cooperative_lanechange"),
                    ("guidance/plugins/inlanecruising_plugin"),
                    ("guidance/plugins/inlanecruising_plugin"),
                    ("guidance/plugins/cooperative_lanechange"),
                    ("guidance/plugins/cooperative_lanechange"),
                    ("guidance/plugins/inlanecruising_plugin"),
                ],
            )
        }

        # Call the function and assert the expected output
        intervals = get_planner_trajectory_intervals(
            mock_mcap_path, "guidance/plugins/inlanecruising_plugin"
        )
        assert intervals == [(0.0, 3.0), (5.0, 7.0), (9.0, 9.1)]
