import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from parse_ros2_bags import extract_mcap_data, get_earliest_timestamp

@pytest.fixture
def mock_reader():
    return MagicMock()

def test_get_earliest_timestamp(mock_reader):
    mock_reader.read_next.side_effect = [
        # Topic messages should generally arrive with timestamps in order
        ("/topic2", "msg2", 1000000000),
        ("/topic1", "msg1", 2000000000)
    ]
    
    result = get_earliest_timestamp(mock_reader)
    
    assert result == 1000000000
    assert mock_reader.set_filter.call_count == 2

@pytest.fixture
def mock_mcap_path():
    return Path("/path/to/mock.mcap")

def test_extract_mcap_data_file_not_found(mock_mcap_path):
    with pytest.raises(ValueError, match="MCAP file .* does not exist"):
        extract_mcap_data(mock_mcap_path, ["/test_topic"])

def test_extract_mcap_data_missing_extractor():
    with patch('os.path.exists', return_value=True):  # Mocking os.path.exists to always return True
        with pytest.raises(ValueError, match="Missing field extractors for topics"):
            extract_mcap_data(Path("existing.mcap"), ["/topic1", "/topic2"], field_extractors={})

@patch('parse_ros2_bags.open_bagfile')
@patch('parse_ros2_bags.deserialize_message')
@patch('parse_ros2_bags.get_message')
@patch('os.path.exists')
def test_extract_mcap_data_success(mock_exists, mock_get_message, mock_deserialize, mock_open_bagfile, mock_mcap_path):
    mock_exists.return_value = True
    mock_reader = MagicMock()
    mock_reader.has_next.side_effect = [True, True, False]
    mock_reader.read_next.side_effect = [
        ("/topic1", b"data1", 1000000000),
        ("/topic1", b"data2", 2000000000)
    ]
    mock_open_bagfile.return_value = (mock_reader, {"/topic1": "std_msgs/String"}, 0)
    mock_deserialize.side_effect = [MagicMock(data="message1"), MagicMock(data="message2")]

    result = extract_mcap_data(mock_mcap_path, ["/topic1"], field_extractors={"/topic1": lambda msg: msg.data})

    assert "/topic1" in result
    assert len(result["/topic1"][0]) == 2  # timestamps
    assert len(result["/topic1"][1]) == 2  # values
    assert result["/topic1"][1][0] == "message1"
    assert result["/topic1"][1][1] == "message2"
