import json
from dataclasses import dataclass
from enum import Enum

class KafkaLogMessageType(Enum):
    """Enumeration used for indentifying the type of KafkaLogMessage
    """
    TimeSync="time_sync"
    DesiredPhasePlan="desired_phase_plan"
    SPAT="spat"
    TSCConfigState="tsc_config_state"
    BSM="bsm"
    MAP="map"
    MobilityOperation="mobility_operation"
    SchedulingPlan="scheduling_plan"
    SDSM="sdsm"
    DetectedObject="detected_object"
    VehicleStatusIntent="vehicle_status_intent"

@dataclass
class KafkaLogMessage:
    """Class used to store data for each Kafka Log Message
    """
    created: int
    json_message: dict
    msg_type: KafkaLogMessageType
    def __init__(self, created, json_message, msg_type):
        self.created = created
        self.json_message = json_message
        self.msg_type = msg_type



