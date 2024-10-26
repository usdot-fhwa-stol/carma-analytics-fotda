import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from guidance_scripts import get_engage_time, run_crosstrack_analysis
from pytest import approx

"""
Usage: 
cd carma-analytics-fotda/src/carma-platform/
python3 -m pytest test
"""


@pytest.fixture
def mock_mcap_path():
    return Path("/path/to/mock.mcap")


def test_get_engage_time(mock_mcap_path):
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
                [
                    1.0,
                    1.5,
                    0.5,
                    1.8,
                ],
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
        assert stats == {
            "minimum": approx(0.5000, rel=1e-2),
            "maximum": approx(1.8000, rel=1e-2),
            "median": approx(1.2500, rel=1e-2),
            "std_dev": approx(0.4950, rel=1e-2),
            "mean": approx(1.2000, rel=1e-2),
            "sample_count": approx(4, rel=1e-2),
            "rms": approx(1.2981, rel=1e-2),
            "start_time_since_recording": approx(0, rel=1e-2),
            "end_time_since_recording": approx(3, rel=1e-2),
        }
        assert plot_figure is not None
        assert cross_tracks == [1.0, 1.5, 0.5, 1.8]
        assert timestamps == [0, 1, 2, 3]
