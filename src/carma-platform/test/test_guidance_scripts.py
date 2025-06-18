import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from guidance_scripts import *
from pytest import approx
from types import SimpleNamespace
import numpy as np

import matplotlib
# Use Agg backend for matplotlib to avoid GUI issues in CI environments
matplotlib.use('Agg')

"""
Usage:
cd carma-analytics-fotda/src/carma-platform/
python3 -m pytest test
"""


@pytest.fixture
def mock_mcap_path():
    return Path("/path/to/mock.mcap")


def test_check_deceleration_for_geofence():
    # Test vehicle doesn't enter geofence
    assert check_deceleration_for_geofence(time_enter_geofence=None, accelerations=None, max_deceleration=None) is False

    # Test vehicle enters geofence with sufficient deceleration
    accelerations = [
        (1.0, 2.0),
        (2.0, -1.0),
        (3.0, -4.0),
        (4.0, -5.0),
        (5.0, -2.0),
        (6.0, -2.0),
        (7.0, -2.0),
        (8.0, -2.0),
        (9.0, -2.0),
        (10.0, -2.0),
        (11.0, -2.0)]
    max_deceleration = -4.0
    time_enter_geofence = 1.0

    assert check_deceleration_for_geofence(time_enter_geofence, accelerations, max_deceleration) is True

    # Test deceleration period never began
    accelerations = [[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
    assert check_deceleration_for_geofence(time_enter_geofence, accelerations, max_deceleration) is False

def test_check_acceleration_after_geofence():

    assert check_acceleration_after_geofence(time_exit_geofence=None, accelerations=None, min_average_acceleration=None, section_accelerations=None, max_section_acceleration=None) is False

    time_exit_geofence = 2.0
    min_average_acceleration = 2.0
    max_section_acceleration = 5.0
    section_accelerations = [
        (1.0, 2.0), (2.0, 1.0), (3.0, 4.0), (4.0, 5.0), (5.0, 2.0), (6.0, 2.0),
        (7.0, 2.0), (8.0, 2.0), (9.0, 2.0), (10.0, 2.0), (11.0, 2.0), (12.0, 2.0)
    ]

    # Test vehicle exits geofence with less than sufficient acceleration
    times = range (1,13)
    accel_values = [1.0] * 12
    accelerations = list(zip(times, accel_values))

    assert check_acceleration_after_geofence(time_exit_geofence, accelerations, min_average_acceleration, section_accelerations, max_section_acceleration) is False

    accel_values = [-2.0, 1.0, 4.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    accelerations = list(zip(times, accel_values))

    assert check_acceleration_after_geofence(time_exit_geofence, accelerations, min_average_acceleration, section_accelerations, max_section_acceleration) is True

def test_check_deceleration_for_geofence():
    time_enter_geofence = 1.0
    times = range (1,12)
    accel_values = [-1.0] * 11
    accelerations = list(zip(times, accel_values))
    max_deceleration = -3.0

    assert check_deceleration_for_geofence(time_enter_geofence, accelerations, max_deceleration) is True

def test_create_geofence_acceleration_plot(tmp_path):

    times = range (1,13)
    accel_values = [1, -2, -3, -4, -2] + [1.0] * 7
    accelerations = list(zip(times, accel_values))

    sec_accelerations = [
        (1.0, 2.0), (2.0, 1.0), (3.0, 4.0), (4.0, 5.0), (5.0, 2.0), (6.0, 2.0),
        (7.0, 2.0), (8.0, 2.0), (9.0, 2.0), (10.0, 2.0), (11.0, 2.0), (12.0, 2.0)
    ]

    time_enter_geofence = 2.0
    time_exit_geofence = 5.0
    save_dir = tmp_path

    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()

    with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, (mock_ax1, mock_ax2))), \
         patch("matplotlib.pyplot.savefig") as mock_savefig, \
         patch("matplotlib.pyplot.show") as mock_show:
        create_geofence_acceleration_plot(
            accelerations, sec_accelerations, time_enter_geofence, time_exit_geofence, save_plots_dir=save_dir
        )
        mock_savefig.assert_called_once()
        mock_show.assert_not_called()
        args, kwargs = mock_savefig.call_args
        assert "geofence_acceleration.png" in str(args[0])

def test_check_speed_before_before_workzone(mock_mcap_path):
    workzone_lanelet_id = 174
    start_time = 0
    end_time = 5
    advisory_speed_limit_ms = 15.0
    speed_limit_threshold_ms = 1.0

    def make_twist(x):
        linear = MagicMock()
        linear.x = x
        twist = MagicMock()
        twist.linear = linear
        return twist


    with patch("guidance_scripts.extract_mcap_data") as mock_extract:
        # Mock timestamps and twist messages
        timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        twists = [make_twist(x) for x in [16.0, 15.0, 15.0, 20.0, 20.0]]

        # Setup mock return value
        mock_extract.return_value = {
            "/hardware_interface/vehicle/twist": (timestamps, twists),
            "/guidance/route_state": (
                [0, 1, 2, 3, 4, 5],
                [172, 173, 174, 175, 176, 177],  # Lanelet IDs
            )
        }
        assert check_speed_before_workzone("mock_mcap_path", start_time, end_time, workzone_lanelet_id, advisory_speed_limit_ms, speed_limit_threshold_ms) is True

def test_check_time_to_begin_deceleration():
    #Test no speed limit changes
    assert check_time_to_begin_deceleration(speed_limit_changes=None, response_times=None, response_threshold=None, save_stats_dir=None, save_data_dir=None) is False

    #Test speed limit changes under response threshold
    speed_limit_change_times = [1.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    old_speed_limits = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    new_speed_limits = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    speed_limit_changes = zip(speed_limit_change_times, old_speed_limits, new_speed_limits)

    response_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    response_threshold = 4.0

    assert check_time_to_begin_deceleration(speed_limit_changes, response_times, response_threshold, None, None) is True

def test_find_accel_period():
    # Arguments required (accelerations, time_start, deceleration)
    # Tuple of lists with timestamps and accelerations/decelerations
    timestamps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    acceleration_values = [1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0]
    acceleration = zip(timestamps, acceleration_values)
    time_start = 1.0
    deceleration = True

    time_start_period, time_end_period, accels = find_accel_period(acceleration, time_start, deceleration)
    # Returns - time_start_period, time_end_period, accels
    assert time_start_period == 2.0
    assert time_end_period == 12.0
    assert accels[0] == approx(-2.0, rel=1e-2)

def test_check_lanechange_duration(mock_mcap_path, tmp_path):

    start_time = 0.0
    max_lanechange_duration = 5.0

    with patch("guidance_scripts.extract_mcap_data") as mock_extract:
        mock_extract.return_value = {
            "/guidance/plan_trajectory": (
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [
                    ("cooperative_lanechange"),
                    ("cooperative_lanechange"),
                    ("cooperative_lanechange"),
                    ("cooperative_lanechange"),
                    ("cooperative_lanechange")
                ],
            )
        }
        is_successful, stats = check_lanechange_duration(mock_mcap_path, start_time, max_lanechange_duration, None, None)
        assert is_successful is True
        assert stats["maximum"] == approx(4.0, rel=0.1)

        is_successful, stats = check_lanechange_duration(mock_mcap_path, start_time, max_lanechange_duration, tmp_path, tmp_path)
        assert is_successful is True


# def test_check_lanechange_lateral_velocity(mock_mcap_path, tmp_path):
#     # mcap_path: Path to MCAP file
#     # min_lat_velocity: Minimum lateral velocity value during lane change
#     # max_lat_velocity: Maximum lateral velocity value during lane change
#     # save_stats_dir
#     # save_data_dir
#     # save_plot_dir

#     min_lat_velocity = 0.5
#     max_lat_velocity = 2.0

#     def make_twist(x):
#         linear = MagicMock()
#         linear.x = x
#         twist = MagicMock()
#         twist.linear = linear
#         return twist

#     # Mock timestamps and twist messages
#     timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
#     twists = [make_twist(x) for x in [16.0, 15.0, 15.0, 20.0, 20.0]]

#     with patch("guidance_scripts.extract_mcap_data") as mock_extract:
#         mock_extract.return_value = {
#             "/guidance/plan_trajectory": (
#                 [0.0, 1.0, 2.0, 3.0, 4.0],
#                 [
#                     ("cooperative_lanechange"),
#                     ("cooperative_lanechange"),
#                     ("cooperative_lanechange"),
#                     ("cooperative_lanechange"),
#                     ("cooperative_lanechange")
#                 ]
#             ),
#             "/localization/current_pose": (
#                 [0, 1, 2, 3, 4],
#                 [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
#                 ),
#                 "/hardware_interface/vehicle/twist": (timestamps, twists)
#         }
#         check_lanechange_lateral_velocity(mock_mcap_path, min_lat_velocity, max_lat_velocity)


def test_check_steady_state_after_geofence(mock_mcap_path):

    time_begin_acceleration_after_geofence = 2.0
    time_end_engagement = 5.0
    original_speed_limit_ms = 15.0
    min_time_at_steady_state = 2.0
    threshold_speed_limit_offset = 0.89408  # 2 mph in m/s

    def make_twist(x):
        linear = MagicMock()
        linear.x = x
        twist = MagicMock()
        twist.linear = linear
        return twist

    # Mock timestamps and twist messages
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    twists = [make_twist(x) for x in [16.0, 15.0, 15.0, 20.0, 20.0]]


    with patch("guidance_scripts.extract_mcap_data") as mock_extract:
        mock_extract.return_value = {
        "/hardware_interface/vehicle/twist": (timestamps, twists)
        }

        assert check_steady_state_after_geofence(mock_mcap_path, time_begin_acceleration_after_geofence, time_end_engagement, original_speed_limit_ms, min_time_at_steady_state, threshold_speed_limit_offset) is True


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

@patch('matplotlib.pyplot.figure')
@patch('builtins.open', new_callable=mock_open)
@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
@patch('json.dump')
def test_run_crosstrack_analysis(mock_json_dump, mock_file, mock_savez, mock_mkdir, mock_plt, mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract:

        mock_extract.return_value = {
            "/guidance/route_state": (
                [0, 1, 2, 3],
                [1.0, 1.5, 0.5, 1.8],
            )
        }
        mock_plt.figure.return_value = MagicMock()

        fake_save_dir = "/fake/dir"

        is_passed, stats, plot_figure, cross_tracks, timestamps = (
            run_crosstrack_analysis(
                mock_mcap_path,
                error_threshold_to_pass_meter=2.0,
                start_time=0,
                end_time=3,
                save_stats_dir=fake_save_dir,
                save_data_dir=fake_save_dir,
                save_plot_dir=fake_save_dir,
            )
        )

        assert is_passed == True
        assert stats["minimum"] == approx(0.5000, rel=1e-2)
        assert stats["maximum"] == approx(1.8000, rel=1e-2)
        assert stats["median"] == approx(1.2500, rel=1e-2)
        assert plot_figure is not None
        assert cross_tracks == approx([1.0, 1.5, 0.5, 1.8])
        assert timestamps == approx([0, 1, 2, 3])

@patch('matplotlib.pyplot.figure')
@patch('builtins.open', new_callable=mock_open)
@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
@patch('json.dump')
def test_run_turn_accuracy_analysis(mock_json_dump, mock_file, mock_savez, mock_mkdir, mock_plt, mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract:

        mock_extract.return_value = {
            "/localization/current_pose": (
                [0, 1, 2],
                [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            ),
            "/guidance/plan_trajectory": ([0], [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]]),
        }
        mock_plt.figure.return_value = MagicMock()
        fake_save_dir = "/fake/dir"

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
            save_stats_dir=Path(fake_save_dir),
            save_data_dir=fake_save_dir,
            save_plot_dir=fake_save_dir
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

@patch('guidance_scripts.plt')
@patch('builtins.open', new_callable=mock_open)
@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
@patch('json.dump')
def test_run_acceleration_comfort_analysis(mock_json_dump, mock_file, mock_savez, mock_mkdir, mock_plt, mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract:
        # Mock plt.subplots to return a tuple of (figure, (ax1, ax2))
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        # Create fake save directory
        fake_save_dir = "/fake/dir"

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
            save_stats_dir=Path(fake_save_dir),
            save_data_dir=fake_save_dir,
            save_plot_dir=fake_save_dir
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

@patch('guidance_scripts.plt')
@patch('builtins.open', new_callable=mock_open)
@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
@patch('json.dump')
def test_run_lateral_analysis(mock_json_dump, mock_file, mock_savez, mock_mkdir, mock_plt, mock_mcap_path):
    """Test the run_lateral_analysis function with mocked data"""
    with patch("guidance_scripts.extract_mcap_data") as mock_extract:

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

        # Create fake save directory
        fake_save_dir = "/fake/dir"

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
            mock_mcap_path,
            acc_threshold_to_pass=2.0,
            jerk_threshold_to_pass=2.0,
            save_stats_dir=Path(fake_save_dir),
            save_data_dir=Path(fake_save_dir),
            save_plot_dir=Path(fake_save_dir)
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

@patch('guidance_scripts.plt')
@patch('builtins.open', new_callable=mock_open)
@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
@patch('json.dump')
def test_run_guidance_steering_analysis(mock_json_dump, mock_file, mock_savez, mock_mkdir, mock_plt, mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract:

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

        # Create fake save directory
        fake_save_dir = "/fake/dir"

        # Run analysis
        is_passed, stats, plot_figure, error_angles, common_timestamps = (
            run_guidance_steering_analysis(
                mock_mcap_path,
                error_threshold_to_pass_radian=0.1,
                start_time=0,
                end_time=4,
                save_stats_dir=Path(fake_save_dir),
                save_data_dir=Path(fake_save_dir),
                save_plot_dir=Path(fake_save_dir)
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

@patch('guidance_scripts.plt')
@patch('builtins.open', new_callable=mock_open)
@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
@patch('json.dump')
def test_run_steering_wheel_analysis(mock_json_dump, mock_file, mock_savez, mock_mkdir, mock_plt, mock_mcap_path):
    with patch("guidance_scripts.extract_mcap_data") as mock_extract:

        # Mock timestamps and steering values
        timestamps = np.array([0, 1, 2, 3, 4])
        values = [(0.5, 0.48), (1.0, 1.02), (1.5, 1.47), (1.0, 1.03), (0.5, 0.51)]

        mock_extract.return_value = {
            "/hardware_interface/as/pacmod/parsed_tx/steer_rpt": (
                timestamps,
                values
            ),
        }

        mock_plt.figure.return_value = MagicMock()

        # Create fake save directory
        fake_save_dir = "/fake/dir"

        # Run analysis
        is_passed, stats, plot_figure, error_values, timestamps = (
            run_steering_wheel_analysis(
                mock_mcap_path, error_threshold_to_pass=0.1,
                start_time=0,
                end_time=4,
                save_stats_dir=Path(fake_save_dir),
                save_data_dir=Path(fake_save_dir),
                save_plot_dir=Path(fake_save_dir)
            )
        )

        # Test output values
        assert is_passed == True  # Error should be within threshold
        assert stats["minimum"] == approx(0.01, rel=1e-2)  # Minimum error
        assert stats["maximum"] == approx(0.03, rel=1e-2)  # Maximum error
        assert stats["median"] == approx(0.02, rel=1e-2)  # Median error

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
        values = [(0.5, 0.7), (1.0, 1.2), (1.5, 1.7)]

        mock_extract.return_value = {
            "/hardware_interface/as/pacmod/parsed_tx/steer_rpt": (
                timestamps,
                values
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


def test_detect_speed_limit_changes():
    """Test detection of speed limit changes with threshold"""
    # Test data
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    speed_limits = np.array([10.0, 10.0, 15.0, 15.0, 9.0, 9.0])
    min_change = 0.5

    # Expected changes at t=2 and t=4
    changes = detect_speed_limit_changes(timestamps, speed_limits, min_change)

    assert len(changes) == 2
    assert changes[0][0] == 2.0  # First change timestamp
    assert changes[0][1] == 10.0  # Old limit
    assert changes[0][2] == 15.0  # New limit
    assert changes[1][0] == 4.0  # Second change timestamp
    assert changes[1][1] == 15.0  # Old limit
    assert changes[1][2] == 9.0   # New limit

    """Test that changes below threshold are ignored"""
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    speed_limits = np.array([10.0, 10.3, 10.4, 10.4])  # Small changes
    min_change = 0.5

    changes = detect_speed_limit_changes(timestamps, speed_limits, min_change)

    assert len(changes) == 0  # No changes should be detected

    """Test handling of zero values in speed limits"""
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    speed_limits = np.array([0.0, 0.0, 5.0, 5.0, 5.0])  # Start with zeros
    min_change = 0.5

    changes = detect_speed_limit_changes(timestamps, speed_limits, min_change)

    assert len(changes) == 0

def test_analyze_speed_responses():
    """Test basic response time calculation"""
    timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    cmd_velocities = np.array([10.0, 10.0, 10.1, 10.5, 10.5])  # Command changes at t=0.2
    velocities = np.array([10.0, 10.0, 10.0, 10.2, 10.4])  # Vehicle starts responding at t=0.3
    speed_limit_changes = [(0.1, 8.0, 11.0)]  # Speed limit changes at t=0.1
    steady_state_time = 0.1
    tolerance_pct = 0.05

    response_times, steady_state_periods = analyze_speed_responses(
        timestamps, cmd_velocities, velocities, speed_limit_changes,
        steady_state_time, tolerance_pct
    )

    assert len(response_times) == 1
    assert response_times[0] == pytest.approx(0.1, abs=1e-3)  # Response time from 0.1 to 0.2

    """Test handling of multiple speed limit changes"""
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    cmd_velocities = np.array([10.0, 10.0, 15.0, 15.0, 15.0, 8.0, 8.0, 8.0, 8.0, 8.0])  # Commands at t=2 and t=5
    velocities = np.array([10.0, 10.0, 10.0, 12.5, 15.0, 15.0, 15.0, 10.0, 8.0, 8.0])  # Vehicle responds gradually
    speed_limit_changes = [(1.0, 10.0, 15.0), (6.0, 15.0, 8.0)]
    steady_state_time = 1.0
    tolerance_pct = 0.05

    response_times, steady_state_periods = analyze_speed_responses(
        timestamps, cmd_velocities, velocities, speed_limit_changes,
        steady_state_time, tolerance_pct
    )

    assert len(response_times) == 2
    assert response_times[0] == pytest.approx(1.0, abs=1e-3)  # Response time for first change
    assert response_times[1] == pytest.approx(0.0, abs=1e-3)  # Response time for second change

    assert len(steady_state_periods) == 2
    # First steady state (at 15 m/s) from t=4 to t=5
    assert steady_state_periods[0][0] == pytest.approx(4.0, abs=1e-3)
    assert steady_state_periods[0][1] == pytest.approx(5.0, abs=1e-3)
    assert steady_state_periods[0][2] == pytest.approx(15.0, abs=1e-3)

    # Second steady state (at 8 m/s) from t=6 to t=7
    assert steady_state_periods[1][0] == pytest.approx(8.0, abs=1e-3)
    assert steady_state_periods[1][1] == pytest.approx(9.0, abs=1e-3)
    assert steady_state_periods[1][2] == pytest.approx(8.0, abs=1e-3)


@patch('matplotlib.pyplot.figure')
@patch('builtins.open', new_callable=mock_open)
@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
@patch('json.dump')
def test_run_speed_limit_change_response_analysis_basic(mock_json_dump, mock_savez, mock_mkdir, mock_file, mock_figure):
    """Test the main analysis function with basic mocked data"""
    with patch('guidance_scripts.extract_mcap_data') as mock_extract:
        # Create simple test data
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        velocities = np.array([10.0, 10.0, 12.0, 14.0, 15.0])
        speed_limits = np.array([10.0, 15.0, 15.0, 15.0, 15.0])
        cmd_velocities = np.array([10.0, 15.0, 15.0, 15.0, 15.0])

        # Setup mock return for extract_mcap_data
        mock_extract.return_value = {
            '/hardware_interface/vehicle/twist': (timestamps, velocities),
            '/guidance/route_state': (timestamps, speed_limits),
            '/guidance/ctrl_cmd': (timestamps, cmd_velocities)
        }

        # Create mock figure
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        # Create fake save directory
        fake_save_dir = "/fake/dir"

        # Run the analysis function
        result = run_speed_limit_change_response_analysis(
            mcap_path="test.mcap",
            response_time_threshold=0.2,
            steady_state_indication_time=1.0,
            speed_tolerance_pct=0.05,
            save_stats_dir=fake_save_dir,
            save_data_dir=fake_save_dir,
            save_plot_dir=fake_save_dir
        )

        # Check basic structure of results
        assert len(result) == 5
        passed, stats, fig, changes, response_times = result

        # Verify the function was called correctly
        mock_extract.assert_called_once()

        # Check that results have the right types
        assert isinstance(passed, bool)
        assert isinstance(stats, dict)
        assert fig == mock_fig
        assert isinstance(changes, list)
        assert isinstance(response_times, np.ndarray)


@patch('matplotlib.pyplot.figure')
def test_run_speed_limit_change_response_analysis_fail_case(mock_figure):
    """Test the analysis function with data that should fail the thresholds"""
    with patch('guidance_scripts.extract_mcap_data') as mock_extract:
        # Create test data with slow response times
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        velocities = np.array([10.0, 10.0, 10.0, 10.0, 11.0])  # Very slow to respond
        speed_limits = np.array([10.0, 15.0, 15.0, 15.0, 15.0])  # Change at t=1
        cmd_velocities = np.array([10.0, 10.0, 10.0, 15.0, 15.0])  # Command delayed until t=3

        mock_extract.return_value = {
            '/hardware_interface/vehicle/twist': (timestamps, velocities),
            '/guidance/route_state': (timestamps, speed_limits),
            '/guidance/ctrl_cmd': (timestamps, cmd_velocities)
        }

        # Mock figure
        mock_figure.return_value = MagicMock()

        # Run with strict thresholds
        passed, stats, _, _, _ = run_speed_limit_change_response_analysis(
            mcap_path="test.mcap",
            response_time_threshold=0.1,  # Strict threshold
            steady_state_indication_time=0.5,
            speed_tolerance_pct=0.05
        )

        # Test should fail
        assert passed == False

def test_get_lateral_velocities(mock_mcap_path):
    """Test Get Lateral Velocities functionality"""
    with patch('guidance_scripts.extract_mcap_data') as mock_extract:
        start_time=0
        end_time=4
        orientation_timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Test Scenario where vehicle starts straight, lane changes left (30 degrees), moves straight, lane changes right (30 degrees), moves straight at a constant 2 m/s
        # 30 Degrees -> z = sin(30deg / 2) = 0.259, w = cos(30deg / 2) = 0.966
        orientations = [
            SimpleNamespace(x=0, y=0, z=0, w=1),
            SimpleNamespace(x=0, y=0, z=0.259, w=0.966),
            SimpleNamespace(x=0, y=0, z=0, w=1),
            SimpleNamespace(x=0, y=0, z=-0.259, w=0.966),
            SimpleNamespace(x=0, y=0, z=0, w=1)
        ]
        twist_timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.5])
        twists = [
            SimpleNamespace(x=2, y=0, z=0),
            SimpleNamespace(x=2, y=0, z=0),
            SimpleNamespace(x=2, y=0, z=0),
            SimpleNamespace(x=2, y=0, z=0),
            SimpleNamespace(x=2, y=0, z=0)
        ]

        mock_extract.return_value = {
            '/localization/current_pose': (orientation_timestamps, orientations),
            '/hardware_interface/vehicle/twist': (twist_timestamps, twists)
        }

        lane_change_velocities = get_lateral_velocities(mock_mcap_path, start_time, end_time)

        assert len(lane_change_velocities) == 5
        assert lane_change_velocities[0][1] == lane_change_velocities[2][1] == lane_change_velocities[4][1] == 0
        assert lane_change_velocities[1][1] == approx(1.0, rel=1e-2)
        assert lane_change_velocities[3][1] == approx(-1.0, rel=1e-2)

    """Test get lateral velocities with no orientation data"""
    with patch('guidance_scripts.extract_mcap_data') as mock_extract:
        start_time=0
        end_time=4
        orientation_timestamps = np.array([])
        orientations = []
        twist_timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.5])
        twists = [
            SimpleNamespace(x=1, y=0, z=0),
            SimpleNamespace(x=2, y=0, z=0),
            SimpleNamespace(x=1.8, y=0, z=0),
            SimpleNamespace(x=1.9, y=0, z=0),
            SimpleNamespace(x=2.1, y=0, z=0)
        ]

        mock_extract.return_value = {
            '/localization/current_pose': (orientation_timestamps, orientations),
            '/hardware_interface/vehicle/twist': (twist_timestamps, twists)
        }

        lane_change_velocities = get_lateral_velocities(mock_mcap_path, start_time, end_time)

        assert not lane_change_velocities


@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
def test_check_reroute_duration(mock_savez, mock_mkdir, mock_mcap_path):
    fake_save_dir = '/fake/dir'
    max_route_update_duration = 3

    """Test successful check_reroute_duration() with no restricted lanes"""

    with patch('guidance_scripts.extract_mcap_data') as mock_extract:

        mock_extract.return_value = {
            "/message/incoming_geofence_control": (
                [1, 2],
                [
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 5})
                            ]
                        )
                    ),
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 0})
                            ]
                        )
                    )
                ]
            ),
            "/guidance/route": (
                [0, 3, 4],
                [[123,432,632], [123, 342, 241, 632], [123, 342, 142, 143, 632]]
            )
        }

        passed = check_reroute_duration(mock_mcap_path, max_route_update_duration, fake_save_dir)
        assert passed


        """Test passing check_reroute_duration() with restricted lane received before closed lane"""
        mock_extract.return_value = {
            "/message/incoming_geofence_control": (
                [1, 2],
                [
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 2})
                            ]
                        )
                    ),
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 5})
                            ]
                        )
                    )
                ]
            ),
            "/guidance/route": (
                [0, 3, 4],
                [[123,432,632], [123, 342, 241, 632], [123, 342, 142, 143, 632]]
            )
        }

        passed = check_reroute_duration(mock_mcap_path, max_route_update_duration, fake_save_dir)
        assert passed

        """Test passing check_reroute_duration() with restricted lane received after closed lane"""
        mock_extract.return_value = {
            "/message/incoming_geofence_control": (
                [1, 2],
                [
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 5})
                            ]
                        )
                    ),
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 2})
                            ]
                        )
                    )
                ]
            ),
            "/guidance/route": (
                [0, 3, 4],
                [[123,432,632], [123, 342, 241, 632], [123, 342, 142, 143, 632]]
            )
        }

        passed = check_reroute_duration(mock_mcap_path, max_route_update_duration, fake_save_dir)
        assert passed

        """Test failing check_reroute_duration() with restricted lane update after max_duration"""
        mock_extract.return_value = {
            "/message/incoming_geofence_control": (
                [1, 4],
                [
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 2})
                            ]
                        )
                    ),
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 5})
                            ]
                        )
                    )
                ]
            ),
            "/guidance/route": (
                [0, 5, 6],
                [[123,432,632], [123, 342, 241, 632], [123, 342, 142, 143, 632]]
            )
        }

        passed = check_reroute_duration(mock_mcap_path, max_route_update_duration, fake_save_dir)
        assert not passed

        """Test failing check_reroute_duration() with closed lane update after max_duration"""
        mock_extract.return_value = {
            "/message/incoming_geofence_control": (
                [1, 2],
                [
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 2})
                            ]
                        )
                    ),
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 5})
                            ]
                        )
                    )
                ]
            ),
            "/guidance/route": (
                [0, 3, 6],
                [[123,432,632], [123, 342, 241, 632], [123, 342, 142, 143, 632]]
            )
        }

        passed = check_reroute_duration(mock_mcap_path, max_route_update_duration, fake_save_dir)
        assert not passed

        """Test failing check_reroute_duration() with invalid number of route updates"""
        mock_extract.return_value = {
            "/message/incoming_geofence_control": (
                [1, 4],
                [
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 2})
                            ]
                        )
                    ),
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=5
                            ),
                            vclasses=[
                                SimpleNamespace(**{"vehicle_class": 5})
                            ]
                        )
                    )
                ]
            ),
            "/guidance/route": (
                [0],
                [[123,432,632]]
            )
        }

        passed = check_reroute_duration(mock_mcap_path, max_route_update_duration, fake_save_dir)
        assert not passed


@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
def test_check_speed_limits_in_geofence(mock_savez, mock_mkdir, mock_mcap_path):
    fake_save_dir = "/fake/dir"
    time_enter = 1
    time_exit = 5
    advisory_speed = 2

    """Test passing check_speed_limits_in_geofence"""

    with patch("guidance_scripts.extract_mcap_data") as mock_extract:

        mock_extract.side_effect = [
            # Incoming_geofence_control - Guidance_route_state
            {"/message/incoming_geofence_control": (
                [2,4],
                [
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=12,
                                maxspeed=2
                            )
                        )
                    )
                ]
            )},
            {"/guidance/route_state": (
                [3,4],
                [(2,2),(1.98,3)]
            )}
        ]

        passed = check_speed_limits_in_geofence(mock_mcap_path, time_enter, time_exit, advisory_speed, fake_save_dir)

        assert passed

        """Test failing check_speed_limits_in_geofence() with advisory speed not matched"""
        mock_extract.side_effect = [
            # Incoming_geofence_control - Guidance_route_state
            {"/message/incoming_geofence_control": (
                [2,4],
                [
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=12,
                                maxspeed=2
                            )
                        )
                    )
                ]
            )},
            {"/guidance/route_state": (
                [3,4],
                [(2,2),(3,3)]
            )}
        ]

        passed = check_speed_limits_in_geofence(mock_mcap_path, time_enter, time_exit, advisory_speed, fake_save_dir)

        assert not passed

        """Test failing check_speed_limits_in_geofence() with no time enter/exit geofence"""
        mock_extract.side_effect = [
            # Incoming_geofence_control - Guidance_route_state
            {"/message/incoming_geofence_control": (
                [2,4],
                [
                    SimpleNamespace(
                        params=SimpleNamespace(
                            detail=SimpleNamespace(
                                choice=12,
                                maxspeed=2
                            )
                        )
                    )
                ]
            )},
            {"/guidance/route_state": (
                [3,4],
                [(2,2),(3,3)]
            )}
        ]

        passed = check_speed_limits_in_geofence(mock_mcap_path, None, None, advisory_speed, fake_save_dir)

        assert not passed


@patch('guidance_scripts.Path.mkdir')
@patch('numpy.savez')
def test_check_geofence_in_reroute(mock_savez, mock_mkdir, mock_mcap_path):
    fake_save_dir = "/fake/dir"
    closed_lanelets = [3,5]

    """Test check_geofence_in_reroute() with closed lanelets in initial route and removed from reroute"""
    with patch("guidance_scripts.extract_mcap_data") as mock_extract:

        mock_extract.return_value = {
            "/guidance/route": (
                [1,2],
                [[1,3,5,7],[1,2,4,6,7]]
            )
        }

        fwz_1_passed, fwz_8_passed = check_geofence_in_reroute(mock_mcap_path, closed_lanelets, fake_save_dir)
        assert fwz_1_passed
        assert fwz_8_passed

        """Test check_geofence_in_reroute() with closed lanelets not in initial route and removed from reroute"""
        mock_extract.return_value = {
            "/guidance/route": (
                [1,2],
                [[1,2,4,6,7],[1,2,6,7]]
            )
        }

        fwz_1_passed, fwz_8_passed = check_geofence_in_reroute(mock_mcap_path, closed_lanelets, fake_save_dir)
        assert not fwz_1_passed
        assert fwz_8_passed

        """Test check_geofence_in_reroute() with closed lanelets in initial route and in reroute"""
        mock_extract.return_value = {
            "/guidance/route": (
                [1,2],
                [[1,3,5,7],[1,3,5,6,7]]
            )
        }

        fwz_1_passed, fwz_8_passed = check_geofence_in_reroute(mock_mcap_path, closed_lanelets, fake_save_dir)
        assert fwz_1_passed
        assert not fwz_8_passed

        """Test check_geofence_in_reroute() with closed lanelets not in initial route and present in reroute"""
        mock_extract.return_value = {
            "/guidance/route": (
                [1,2],
                [[1,2,4,6,7],[1,3,5,7]]
            )
        }

        fwz_1_passed, fwz_8_passed = check_geofence_in_reroute(mock_mcap_path, closed_lanelets, fake_save_dir)
        assert not fwz_1_passed
        assert not fwz_8_passed

        """Test check_geofence_in_reroute() with invalid number of routes"""
        mock_extract.return_value = {
            "/guidance/route": (
                [1],
                [[1,3,5,7]]
            )
        }

        fwz_1_passed, fwz_8_passed = check_geofence_in_reroute(mock_mcap_path, closed_lanelets, fake_save_dir)
        assert not fwz_1_passed
        assert not fwz_8_passed

        """Test check_geofence_in_reroute() with no closed lanelets"""
        mock_extract.return_value = {
            "/guidance/route": (
                [1,2],
                [[1,3,5,7],[1,2,4,6,7]]
            )
        }

        fwz_1_passed, fwz_8_passed = check_geofence_in_reroute(mock_mcap_path, [], fake_save_dir)
        assert not fwz_1_passed
        assert not fwz_8_passed


def test_get_route_original_speed(mock_mcap_path):

    with patch('guidance_scripts.extract_mcap_data') as mock_extract:

        mock_extract.return_value = {
            '/guidance/route_state': (
                [1,3,5],
                [2,4,6]
            )
        }

        speed_limit = get_route_original_speed(mock_mcap_path)
        assert speed_limit == 2

        mock_extract.return_value = {
            '/guidance/route_state': (
                [1,3,5],
                [6,4,2]
            )
        }

        speed_limit = get_route_original_speed(mock_mcap_path)
        assert speed_limit == 6


def test_get_geofence_entrance_and_exit_times(mock_mcap_path):

    """Test get_geofence_entrance_and_exit_times() with normal behavior"""
    with patch('guidance_scripts.extract_mcap_data') as mock_extract:

        mock_extract.return_value = {
            '/environment/active_geofence': (
                [1,2,3,4,5,6],
                [False, False, True, True, True, False]
            )
        }

        time_enter, time_exit, found = get_geofence_entrance_and_exit_times(mock_mcap_path)
        assert time_enter == 3
        assert time_exit == 6
        assert found

        """Test get_geofence_entrance_and_exit_times() with never entering geofence"""
        mock_extract.return_value = {
            '/environment/active_geofence': (
                [1,2,3,4,5,6],
                [False, False, False, False, False, False]
            )
        }

        time_enter, time_exit, found = get_geofence_entrance_and_exit_times(mock_mcap_path)
        assert time_enter == None
        assert time_exit == None
        assert not found
