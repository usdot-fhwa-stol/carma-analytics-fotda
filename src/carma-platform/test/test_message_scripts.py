import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from message_scripts import *
from pytest import approx
from types import SimpleNamespace
import numpy as np

"""
Usage:
cd carma-analytics-fotda/src/carma-platform/
python3 -m pytest test
"""

@pytest.fixture
def mock_mcap_path():
    return Path("/path/to/mock.mcap")

def test_passing_check_tcm_acknowledgement_delay(mock_mcap_path):
    fake_save_dir = '/fake/dir'
    max_delay_sec = 1
    mock_data = {
        '/message/incoming_geofence_control':
        [
            [142.25, 142.31, 144.76],
            [
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136,  69,  62, 167, 197,  69,  17], dtype=np.int64)
                    ),
                    msgnum=1
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136,  69,  62, 167, 197,  69,  17], dtype=np.int64)
                    ),
                    msgnum=2
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([253, 183,   5,  51, 158,  15,  76,  27])
                    ),
                    msgnum=1
                )
            ]
        ],
        '/message/outgoing_mobility_operation':
        [
            [142.28, 142.45, 144.77],
            [
                SimpleNamespace(
                    strategy="carma3/Geofence_Acknowledgement",
                    strategy_params='traffic_control_id:f688453ea7c54511, msgnum:1, acknowledgement:1, reason:Successfully processed TCM.'
                ),
                SimpleNamespace(
                    strategy="carma3/Geofence_Acknowledgement",
                    strategy_params='traffic_control_id:f688453ea7c54511, msgnum:2, acknowledgement:1, reason:Successfully processed TCM.'
                ),
                SimpleNamespace(
                    strategy="carma3/Geofence_Acknowledgement",
                    strategy_params='traffic_control_id:fdb705339e0f4c1b, msgnum:1, acknowledgement:1, reason:Dropping received TrafficControl message with already handled id: 008ba29f-41eb-4df1-cf0a-4e3a2ab11096'
                )
            ]
        ]
    }

    with patch("message_scripts.extract_mcap_data") as mock_extract, \
        patch('message_scripts.plt') as mock_plt, \
        patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("builtins.open", mock_open()) as mock_file, \
        patch("json.dump") as mock_json_dump:

        mock_extract.return_value = mock_data

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        passed, acknowledgements, plt_fig, stats = check_tcm_acknowledgement_delay(mock_mcap_path, max_delay_sec, fake_save_dir, fake_save_dir, fake_save_dir)
    
        assert passed
        assert len(acknowledgements) == 3
        assert plt_fig is not None
        assert stats["minimum"] == approx(0.01, rel=1e-2)
        assert stats["maximum"] == approx(0.14, rel=1e-2)
        assert stats["sample_count"] == 3


def test_failing_check_tcm_acknowledgement_delay(mock_mcap_path):
    max_delay_sec = 1
    mock_data = {
        '/message/incoming_geofence_control':
        [
            [142.25, 142.31, 144.76],
            [
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136,  69,  62, 167, 197,  69,  12], dtype=np.int64) # Changed value such that there is a TCM received that isn't acknowledged
                    ),
                    msgnum=1
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136,  69,  62, 167, 197,  69,  17], dtype=np.int64)
                    ),
                    msgnum=2
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([253, 183,   5,  51, 158,  15,  76,  27]) 
                    ),
                    msgnum=1
                )
            ]
        ],
        '/message/outgoing_mobility_operation':
        [
            [142.28, 143.45, 144.77],
            [
                SimpleNamespace(
                    strategy="carma3/Geofence_Acknowledgement",
                    strategy_params='traffic_control_id:f688453ea7c54511, msgnum:1, acknowledgement:1, reason:Successfully processed TCM.'
                ),
                SimpleNamespace(
                    strategy="carma3/Geofence_Acknowledgement",
                    strategy_params='traffic_control_id:f688453ea7c54511, msgnum:2, acknowledgement:1, reason:Successfully processed TCM.'
                ),
                SimpleNamespace(
                    strategy="carma3/Geofence_Acknowledgement",
                    strategy_params='traffic_control_id:fdb705339e0f4c1b, msgnum:1, acknowledgement:1, reason:Dropping received TrafficControl message with already handled id: 008ba29f-41eb-4df1-cf0a-4e3a2ab11096'
                )
            ]
        ]
    }

    with patch("message_scripts.extract_mcap_data") as mock_extract, \
        patch('message_scripts.plt') as mock_plt:

        mock_extract.return_value = mock_data

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        passed, acknowledgements, plt_fig, stats = check_tcm_acknowledgement_delay(mock_mcap_path, max_delay_sec)
    
        assert not passed
        assert len(acknowledgements) == 2
        assert plt_fig is not None
        assert stats["minimum"] == approx(0.01, rel=1e-2)
        assert stats["maximum"] == approx(1.14, rel=1e-2)
        assert stats["sample_count"] == 2


def test_passing_check_tcm_response_time(mock_mcap_path):
    fake_save_dir = '/fake/dir'
    expected_duration_sec = 1

    mock_data = {
        '/message/outgoing_geofence_request':
        [
            [141.89, 142.02, 144.24],
            [
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136, 69, 62, 167, 197, 69, 17], dtype=np.int64)
                    )
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136, 69, 62, 167, 197, 69, 17], dtype=np.int64)
                    )
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([253, 183,   5,  51, 158,  15,  76,  27], dtype=np.int64)
                    )
                )
            ]
        ],
        '/message/incoming_geofence_control':
        [
            [142.25, 142.31, 144.76],
            [
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136,  69,  62, 167, 197,  69,  17], dtype=np.int64)
                    ),
                    msgnum=1
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136,  69,  62, 167, 197,  69,  17], dtype=np.int64)
                    ),
                    msgnum=2
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([253, 183,   5,  51, 158,  15,  76,  27], dtype=np.int64)
                    ),
                    msgnum=1
                )
            ]
        ]
    }

    with patch('message_scripts.extract_mcap_data') as mock_extract, \
        patch('message_scripts.plt') as mock_plt, \
        patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("builtins.open", mock_open()) as mock_file, \
        patch("json.dump") as mock_json_dump:

        mock_extract.return_value = mock_data

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        passed = check_tcm_response_time(mock_mcap_path, expected_duration_sec, fake_save_dir, fake_save_dir, fake_save_dir)

        assert passed


def test_failing_check_tcm_response_time(mock_mcap_path):
    fake_save_dir = '/fake/dir'
    expected_duration_sec = 1

    mock_data = {
        '/message/outgoing_geofence_request':
        [
            [141.89, 142.02, 144.24],
            [
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136, 69, 62, 167, 197, 69, 17], dtype=np.int64)
                    )
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136, 69, 62, 167, 197, 69, 17], dtype=np.int64)
                    )
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([253, 183,   5,  51, 158,  15,  76,  27], dtype=np.int64)
                    )
                )
            ]
        ],
        '/message/incoming_geofence_control':
        [
            [142.25, 142.31, 145.76], # Will fail because the 3rd TCM is received more than expected_duration_sec after the related TCR was sent
            [
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136,  69,  62, 167, 197,  69,  17], dtype=np.int64)
                    ),
                    msgnum=1
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([246, 136,  69,  62, 167, 197,  69,  17], dtype=np.int64)
                    ),
                    msgnum=2
                ),
                SimpleNamespace(
                    reqid=SimpleNamespace(
                        id=np.array([253, 183,   5,  51, 158,  15,  76,  27], dtype=np.int64)
                    ),
                    msgnum=1
                )
            ]
        ]
    }

    with patch('message_scripts.extract_mcap_data') as mock_extract, \
        patch('message_scripts.plt') as mock_plt, \
        patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("builtins.open", mock_open()) as mock_file, \
        patch("json.dump") as mock_json_dump:

        mock_extract.return_value = mock_data

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        passed = check_tcm_response_time(mock_mcap_path, expected_duration_sec, fake_save_dir, fake_save_dir, None)

        assert not passed


def test_process_cc_logs_for_tcr_tcm_data():
    log_date = "2025-05-20"
    max_delay_sec = 0.1
    expected_rate_hz = 20

    expected_reqids = ['F688453EA7C54511', 'FDB705339E0F4C1B', 'F804AA710EA64E31', 'D3F3512874AC47A1']
    rate_tcm_ids = ['00305feb9eac2d077bd5f534833d0152', '008ba29f41eb4df1cf0a4e3a2ab11094']
    inf_rate_tcm_ids = ['00305feb9eac2d077bd5f534833d0153', '008ba29f41eb4df1cf0a4e3a2ab11095']
    
    # Fake files in the directory
    fake_files = ['file1.txt']
    fake_save_dir = '/fake/dir'

    sample_lines = [
        '[DEBUG 19:03:20.071 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlRequest port="33333" list="true"><reqid>F688453EA7C54511</reqid><reqseq>0</reqseq><scale>-1</scale><bounds><TrafficControlBounds><oldest>29107863</oldest><reflon>-771510186</reflon><reflat>389548654</reflat><offsets><OffsetPoint><deltax>3015</deltax><deltay>0</deltay></OffsetPoint><OffsetPoint><deltax>3015</deltax><deltay>1049</deltay></OffsetPoint><OffsetPoint><deltax>0</deltax><deltay>1049</deltay></OffsetPoint></offsets></TrafficControlBounds></bounds></TrafficControlRequest>',
        '[DEBUG 19:03:20.077 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F688453EA7C54511</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11096</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:20.078 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F688453EA7C54511</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0151</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:22.544 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlRequest port="33333" list="true"><reqid>FDB705339E0F4C1B</reqid><reqseq>0</reqseq><scale>-1</scale><bounds><TrafficControlBounds><oldest>29107863</oldest><reflon>-771510186</reflon><reflat>389548654</reflat><offsets><OffsetPoint><deltax>3015</deltax><deltay>0</deltay></OffsetPoint><OffsetPoint><deltax>3015</deltax><deltay>1049</deltay></OffsetPoint><OffsetPoint><deltax>0</deltax><deltay>1049</deltay></OffsetPoint></offsets></TrafficControlBounds></bounds></TrafficControlRequest>',
        '[DEBUG 19:03:22.547 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>FDB705339E0F4C1B</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11094</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:22.548 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>FDB705339E0F4C1B</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0152</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:22.647 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>FDB705339E0F4C1B</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11094</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:22.648 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>FDB705339E0F4C1B</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0152</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:25.601 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlRequest port="33333" list="true"><reqid>F804AA710EA64E31</reqid><reqseq>0</reqseq><scale>-1</scale><bounds><TrafficControlBounds><oldest>29107863</oldest><reflon>-771510186</reflon><reflat>389548654</reflat><offsets><OffsetPoint><deltax>3015</deltax><deltay>0</deltay></OffsetPoint><OffsetPoint><deltax>3015</deltax><deltay>1049</deltay></OffsetPoint><OffsetPoint><deltax>0</deltax><deltay>1049</deltay></OffsetPoint></offsets></TrafficControlBounds></bounds></TrafficControlRequest>',
        '[DEBUG 19:03:25.605 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F804AA710EA64E31</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11095</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:25.606 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F804AA710EA64E31</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0153</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:25.605 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F804AA710EA64E31</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11095</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:25.606 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F804AA710EA64E31</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0153</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:28.521 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlRequest port="33333" list="true"><reqid>D3F3512874AC47A1</reqid><reqseq>0</reqseq><scale>-1</scale><bounds><TrafficControlBounds><oldest>29107863</oldest><reflon>-771510186</reflon><reflat>389548654</reflat><offsets><OffsetPoint><deltax>3015</deltax><deltay>0</deltay></OffsetPoint><OffsetPoint><deltax>3015</deltax><deltay>1049</deltay></OffsetPoint><OffsetPoint><deltax>0</deltax><deltay>1049</deltay></OffsetPoint></offsets></TrafficControlBounds></bounds></TrafficControlRequest>',
        '[DEBUG 19:03:28.525 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>D3F3512874AC47A1</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11096</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:28.526 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>D3F3512874AC47A1</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0151</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>'
    ]

    file_contents = "\n".join(sample_lines)+"\n"

    with patch('os.listdir', return_value=fake_files), \
        patch('os.path.isfile', return_value=True), \
        patch('builtins.open', mock_open(read_data=file_contents)) as mock_file, \
        patch('message_scripts.plt') as mock_plt, \
        patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("json.dump") as mock_json_dump:

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        with np.errstate(invalid='ignore'): # Purposefully ignoring error state on this, the sample cc logs will cause a float('inf') value for the rate which causes warning on calculating error statistics
            all_results, _ = process_cc_logs_for_tcr_tcm_data(fake_save_dir, log_date, max_delay_sec, expected_rate_hz, fake_save_dir, fake_save_dir, fake_save_dir)

        for reqid in expected_reqids:
            assert reqid in all_results

        for reqid, data in all_results.items():
            assert isinstance(data, dict)

            for key in ["first_tcm_time", "response_delay", "tcr_time"]:
                assert key in data

            for tcm_id, tcm_data in data.items():
                if tcm_id in ["first_tcm_time", "response_delay", "tcr_time"]:
                    continue
                assert isinstance(tcm_data, dict)

                for field in ["count", "msgnum", "rate", "timestamps"]:
                    assert field in tcm_data

                if tcm_id in rate_tcm_ids:
                    assert tcm_data["rate"] == approx(expected_rate_hz, rel=0.05)

                if tcm_id in inf_rate_tcm_ids:
                    assert tcm_data["rate"] == float('inf')


def test_bad_data_process_cc_logs_for_tcr_tcm_data():
    log_date = "2025-05-20"
    max_delay_sec = 0.1
    expected_rate_hz = 0

    expected_reqids = ['F688453EA7C54511', 'FDB705339E0F4C1B', 'F804AA710EA64E31', 'D3F3512874AC47A1']
    
    # Fake files in the directory
    fake_files = ['file1.txt']
    fake_save_dir = '/fake/dir'

    sample_lines = [
        'line with out TCR / TCM written out',
        'TrafficControlRequest missing time stamp',
        '[DEBUG 15:20:05.383 [TcmReqServlet] <TrafficControlMessage> missing req id]',
        '[DEBUG 15:25:25.937 [TcmReqServlet] TrafficControlMessage missing <> on TCM/TCR <reqid>F688453EA7C54511</reqid>]',
        '[DEBUG 15:35:35.037 [TcmReqServlet] <TrafficControlMessage> missing tcm msg id <reqid>F688453EA7C54511</reqid>]',
        '[DEBUG 15:45:45.203 [TcmReqServlet] <TrafficControlMessage> missing tcm msgnum <reqid>F688453EA7C54511</reqid>]<id>008ba29f41eb4df1cf0a4e3a2ab11096</id>',
        # Regular log lines below - asserts should be similar to above test
        '[DEBUG 19:03:20.071 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlRequest port="33333" list="true"><reqid>F688453EA7C54511</reqid><reqseq>0</reqseq><scale>-1</scale><bounds><TrafficControlBounds><oldest>29107863</oldest><reflon>-771510186</reflon><reflat>389548654</reflat><offsets><OffsetPoint><deltax>3015</deltax><deltay>0</deltay></OffsetPoint><OffsetPoint><deltax>3015</deltax><deltay>1049</deltay></OffsetPoint><OffsetPoint><deltax>0</deltax><deltay>1049</deltay></OffsetPoint></offsets></TrafficControlBounds></bounds></TrafficControlRequest>',
        '[DEBUG 19:03:20.077 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F688453EA7C54511</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11096</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:20.078 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F688453EA7C54511</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0151</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:22.544 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlRequest port="33333" list="true"><reqid>FDB705339E0F4C1B</reqid><reqseq>0</reqseq><scale>-1</scale><bounds><TrafficControlBounds><oldest>29107863</oldest><reflon>-771510186</reflon><reflat>389548654</reflat><offsets><OffsetPoint><deltax>3015</deltax><deltay>0</deltay></OffsetPoint><OffsetPoint><deltax>3015</deltax><deltay>1049</deltay></OffsetPoint><OffsetPoint><deltax>0</deltax><deltay>1049</deltay></OffsetPoint></offsets></TrafficControlBounds></bounds></TrafficControlRequest>',
        '[DEBUG 19:03:22.547 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>FDB705339E0F4C1B</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11094</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:22.548 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>FDB705339E0F4C1B</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0152</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:25.601 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlRequest port="33333" list="true"><reqid>F804AA710EA64E31</reqid><reqseq>0</reqseq><scale>-1</scale><bounds><TrafficControlBounds><oldest>29107863</oldest><reflon>-771510186</reflon><reflat>389548654</reflat><offsets><OffsetPoint><deltax>3015</deltax><deltay>0</deltay></OffsetPoint><OffsetPoint><deltax>3015</deltax><deltay>1049</deltay></OffsetPoint><OffsetPoint><deltax>0</deltax><deltay>1049</deltay></OffsetPoint></offsets></TrafficControlBounds></bounds></TrafficControlRequest>',
        '[DEBUG 19:03:25.605 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F804AA710EA64E31</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11095</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:25.606 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>F804AA710EA64E31</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0153</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:28.521 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlRequest port="33333" list="true"><reqid>D3F3512874AC47A1</reqid><reqseq>0</reqseq><scale>-1</scale><bounds><TrafficControlBounds><oldest>29107863</oldest><reflon>-771510186</reflon><reflat>389548654</reflat><offsets><OffsetPoint><deltax>3015</deltax><deltay>0</deltay></OffsetPoint><OffsetPoint><deltax>3015</deltax><deltay>1049</deltay></OffsetPoint><OffsetPoint><deltax>0</deltax><deltay>1049</deltay></OffsetPoint></offsets></TrafficControlBounds></bounds></TrafficControlRequest>',
        '[DEBUG 19:03:28.525 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>D3F3512874AC47A1</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>1</msgnum><id>008ba29f41eb4df1cf0a4e3a2ab11096</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>008ba29f41eb4df1cf0a4e3a2ab11096</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29109288</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><closed><notopen/></closed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29109288</reftime><reflon>-771486221</reflon><reflat>389549122</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>5</x><y>0</y><width>1</width></PathNode><PathNode><x>-1491</x><y>94</y><width>8</width></PathNode><PathNode><x>-830</x><y>80</y><width>3</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>',
        '[DEBUG 19:03:28.526 [TcmReqServlet] - <?xml version="1.0" encoding="UTF-8"?><TrafficControlMessage><tcmV01><reqid>D3F3512874AC47A1</reqid><reqseq>0</reqseq><msgtot>2</msgtot><msgnum>2</msgnum><id>00305feb9eac2d077bd5f534833d0151</id><updated>0</updated><package><label>workzone</label><tcids><Id128b>00305feb9eac2d077bd5f534833d0151</Id128b></tcids></package><params><vclasses><micromobile/><motorcycle/><passenger-car/><light-truck-van/><bus/><two-axle-six-tire-single-unit-truck/><three-axle-single-unit-truck/><four-or-more-axle-single-unit-truck/><four-or-fewer-axle-single-trailer-truck/><five-axle-single-trailer-truck/><six-or-more-axle-single-trailer-truck/><five-or-fewer-axle-multi-trailer-truck/><six-axle-multi-trailer-truck/><seven-or-more-axle-multi-trailer-truck/></vclasses><schedule><start>29129391</start><end>153722867280912</end><dow>1111111</dow></schedule><regulatory><true/></regulatory><detail><maxspeed>45</maxspeed></detail></params><geometry><proj>epsg:3785</proj><datum>WGS84</datum><reftime>29129391</reftime><reflon>-771488786</reflon><reflat>389548996</reflat><refelv>0</refelv><heading>3312</heading><nodes><PathNode><x>1</x><y>0</y><width>-1</width></PathNode><PathNode><x>1489</x><y>-152</y><width>24</width></PathNode><PathNode><x>1496</x><y>-92</y><width>9</width></PathNode><PathNode><x>60</x><y>-3</y><width>2</width></PathNode></nodes></geometry></tcmV01></TrafficControlMessage>'
    
    ]

    file_contents = "\n".join(sample_lines)+"\n"

    with patch('os.listdir', return_value=fake_files), \
        patch('os.path.isfile', return_value=True), \
        patch('builtins.open', mock_open(read_data=file_contents)) as mock_file, \
        patch('message_scripts.plt') as mock_plt, \
        patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("json.dump") as mock_json_dump:

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        with np.errstate(invalid='ignore'): # Purposefully ignoring error state on this, the sample cc logs will cause a float('inf') value for the rate which causes warning on calculating error statistics
            all_results, _ = process_cc_logs_for_tcr_tcm_data(fake_save_dir, log_date, max_delay_sec, expected_rate_hz, fake_save_dir, fake_save_dir, None)

        for reqid in expected_reqids:
            assert reqid in all_results

        for reqid, data in all_results.items():
            assert isinstance(data, dict)

            for key in ["first_tcm_time", "response_delay", "tcr_time"]:
                assert key in data

            for tcm_id, tcm_data in data.items():
                if tcm_id in ["first_tcm_time", "response_delay", "tcr_time"]:
                    continue
                assert isinstance(tcm_data, dict)

                for field in ["count", "msgnum", "rate", "timestamps"]:
                    assert field in tcm_data

                assert tcm_data["rate"] == approx(expected_rate_hz, rel=0.05)


def test_passing_check_cc_response_delay():
    fake_save_dir = '/fake/dir'
    expected_delay_sec = 1

    cc_data = {'A': {'tcr_time': 1, 'first_tcm_time': 1.1, 'response_delay': 0.1, 'a1': {'timestamps': [1.1, 1.3, 1.5], 'msgnum': 1, 'count': 3, 'rate': 5}},
               'B': {'tcr_time': 1.5, 'first_Tcm_time': 2.4, 'response_delay': 0.9, 'b1': {'timestamps': [2.4], 'msgnum': 1, 'count': 1, 'rate': 0}}
    }

    with patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("builtins.open", mock_open()) as mock_file, \
        patch("json.dump") as mock_json_dump:
    
        passed = check_cc_response_delay(cc_data, expected_delay_sec, fake_save_dir, fake_save_dir)

        assert passed


def test_failing_check_cc_response_delay():
    fake_save_dir = '/fake/dir'
    expected_delay_sec = 1

    cc_data = {'A': {'tcr_time': 1, 'first_tcm_time': 1.1, 'response_delay': 0.1, 'a1': {'timestamps': [1.1, 1.3, 1.5], 'msgnum': 1, 'count': 3, 'rate': 5}},
               'B': {'tcr_time': 1.5, 'first_Tcm_time': 2.6, 'response_delay': 1.1, 'b1': {'timestamps': [2.6], 'msgnum': 1, 'count': 1, 'rate': 0}}
    }

    with patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("builtins.open", mock_open()) as mock_file, \
        patch("json.dump") as mock_json_dump:
    
        passed = check_cc_response_delay(cc_data, expected_delay_sec, fake_save_dir, fake_save_dir)

        assert not passed


def test_passing_check_tcm_broadcast_count():
    fake_save_dir = '/fake/dir'
    expected_count = 3

    cc_data = {'A': {'tcr_time': 1, 'first_tcm_time': 1.1, 'response_delay': 0.1, 'a1': {'timestamps': [1.1, 1.3, 1.5], 'msgnum': 1, 'count': 3, 'rate': 5}},
               'B': {'tcr_time': 1.5, 'first_tcm_time': 2.6, 'response_delay': 1.1, 'b1': {'timestamps': [2.6], 'msgnum': 2, 'count': 1, 'rate': 0}}
    }

    acknowledgements = [('A', 1, 1.1, 1.2), ('B', 2, 2.6, 3.2)]

    with patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("builtins.open", mock_open()) as mock_file, \
        patch("json.dump") as mock_json_dump:

        passed = check_tcm_broadcast_count(cc_data, acknowledgements, expected_count, fake_save_dir, fake_save_dir)

        assert passed


def test_failing_check_tcm_broadcast_count():
    fake_save_dir = '/fake/dir'
    expected_count = 2 # Causes failure on 'A' because the count is greater than expected

    cc_data = {'A': {'tcr_time': 1, 'first_tcm_time': 1.1, 'response_delay': 0.1, 'a1': {'timestamps': [1.1, 1.3, 1.5], 'msgnum': 1, 'count': 3, 'rate': 5}},
               'B': {'tcr_time': 1.5, 'first_tcm_time': 2.6, 'response_delay': 1.1, 'b1': {'timestamps': [2.6], 'msgnum': 2, 'count': 1, 'rate': 0}}
    }

    acknowledgements = [('A', 1, 1.1, 1.2)] # Causes failure on 'B' because count is less than expected AND was not acknowledged

    with patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("builtins.open", mock_open()) as mock_file, \
        patch("json.dump") as mock_json_dump:

        passed = check_tcm_broadcast_count(cc_data, acknowledgements, expected_count, fake_save_dir, fake_save_dir)

        assert not passed


def test_passing_check_tcm_broadcast_rate():
    fake_save_dir = '/fake/dir'
    expected_rate_hz = 5

    cc_data = {'A': {'tcr_time': 1, 'first_tcm_time': 1.1, 'response_delay': 0.1, 'a1': {'timestamps': [1.1, 1.3, 1.5], 'msgnum': 1, 'count': 3, 'rate': 5}},
               'B': {'tcr_time': 1.5, 'first_tcm_time': 2.6, 'response_delay': 1.1, 'b1': {'timestamps': [2.6], 'msgnum': 2, 'count': 1, 'rate': 0}}
    }

    acknowledgements = [('A', 1, 1.1, 1.2), ('B', 2, 2.6, 3.2)]

    with patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("builtins.open", mock_open()) as mock_file, \
        patch("json.dump") as mock_json_dump:

        passed = check_tcm_broadcast_rate(cc_data, acknowledgements, expected_rate_hz, fake_save_dir, fake_save_dir)

        assert passed


def test_failing_check_tcm_broadcast_rate():
    fake_save_dir = '/fake/dir'
    expected_rate_hz = 10 # Fails 'A' for having an incorrect rate (still accepted because acknowledged)

    cc_data = {'A': {'tcr_time': 1, 'first_tcm_time': 1.1, 'response_delay': 0.1, 'a1': {'timestamps': [1.1, 1.3, 1.5], 'msgnum': 1, 'count': 3, 'rate': 5}},
               'B': {'tcr_time': 1.5, 'first_tcm_time': 2.6, 'response_delay': 1.1, 'b1': {'timestamps': [2.6], 'msgnum': 2, 'count': 1, 'rate': 0}}
    }

    acknowledgements = [('A', 1, 1.1, 1.2)] # Fails 'B' because rate is wrong AND not acknowledged

    with patch.object(Path, "mkdir") as mock_mkdir, \
        patch("numpy.savez") as mock_savez, \
        patch("builtins.open", mock_open()) as mock_file, \
        patch("json.dump") as mock_json_dump:

        passed = check_tcm_broadcast_rate(cc_data, acknowledgements, expected_rate_hz, fake_save_dir, fake_save_dir)

        assert not passed

