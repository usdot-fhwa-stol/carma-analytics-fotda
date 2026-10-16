import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import cdasim_log_analyzer  # noqa: E402


def test_parse_comm_node_names_supports_structured_network_nodes():
    lines = [
        "CDAS_EVENT event=network_node_added node_id=carma_1 node_type=vehicle "
        "external_id=1 x=1.0 y=2.0 z=0.0 time_ns=1000000000",
        "CDAS_EVENT event=network_node_added node_id=traffic_light_468 node_type=rsu "
        "external_id=2 x=3.0 y=4.0 z=0.0 time_ns=1000000000",
    ]

    assert cdasim_log_analyzer.Parse_Comm_Node_Names(lines) == ["carma_1", "traffic_light_468"]
    assert cdasim_log_analyzer.Parse_Comm_Nodes_By_Type(lines, "vehicle") == ["carma_1"]
    assert cdasim_log_analyzer.Parse_Comm_Nodes_By_Type(lines, "rsu") == ["traffic_light_468"]
    assert cdasim_log_analyzer._rsu_nodes(lines, cdasim_log_analyzer.Parse_Comm_Node_Names(lines)) == [
        "traffic_light_468"
    ]


def test_parse_comm_node_names_keeps_legacy_fallbacks():
    lines = [
        "Adding vehicle ID[int=carma_1, ext=1] at projected_position=(1, 2) time=1",
        "Receive V2XMessage : Id(10) on Node rsu_1 at Time=2",
    ]

    assert cdasim_log_analyzer.Parse_Comm_Node_Names(lines) == ["carma_1", "rsu_1"]
