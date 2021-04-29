import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job


## PARAMETERS: just change the run numbers on the next 2 lines, to execute the job on the corresponding dataset
glue_db = 'carma-core-ford-r9'
redshift_db_name = 'white_ford_r9'

## DO NOT change anything below this line, UNLESS to add a new topic to be processed
## @params: [TempDir, JOB_NAME]
args = getResolvedOptions(sys.argv, ['TempDir','JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

try:
    datasource0 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_result_ctrl_lateral_accel_csv', transformation_ctx = 'datasource0')
    applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'string'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping1')
    resolvechoice2 = ResolveChoice.apply(frame = applymapping1, choice = 'make_cols', transformation_ctx = 'resolvechoice2')
    dropnullfields3 = DropNullFields.apply(frame = resolvechoice2, transformation_ctx = 'dropnullfields3')
    datasink4 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields3, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_result_ctrl_lateral_accel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink4')
except:
    print('Failed')

try:
    datasource1 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_bestpos_csv', transformation_ctx = 'datasource1')
    applymapping2 = ApplyMapping.apply(frame = datasource1, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('solution_status', 'string', 'solution_status', 'string'), ('position_type', 'string', 'position_type', 'string'), ('lat', 'double', 'lat', 'double'), ('lon', 'double', 'lon', 'double'), ('height', 'double', 'height', 'double'), ('undulation', 'double', 'undulation', 'double'), ('datum_id', 'string', 'datum_id', 'string'), ('lat_sigma', 'double', 'lat_sigma', 'double'), ('lon_sigma', 'double', 'lon_sigma', 'double'), ('height_sigma', 'double', 'height_sigma', 'double'), ('base_station_id', 'string', 'base_station_id', 'string'), ('diff_age', 'double', 'diff_age', 'double'), ('solution_age', 'double', 'solution_age', 'double'), ('num_satellites_tracked', 'long', 'num_satellites_tracked', 'long'), ('num_satellites_used_in_solution', 'long', 'num_satellites_used_in_solution', 'long'), ('num_gps_and_glonass_l1_used_in_solution', 'long', 'num_gps_and_glonass_l1_used_in_solution', 'long'), ('num_gps_and_glonass_l1_and_l2_used_in_solution', 'long', 'num_gps_and_glonass_l1_and_l2_used_in_solution', 'long'), ('extended_solution_status', 'string', 'extended_solution_status', 'string'), ('original_mask', 'long', 'original_mask', 'long'), ('advance_rtk_verified', 'boolean', 'advance_rtk_verified', 'boolean'), ('psuedorange_iono_correction', 'string', 'psuedorange_iono_correction', 'string'), ('signal_mask', 'string', 'signal_mask', 'string'), ('gps_l1_used_in_solution', 'boolean', 'gps_l1_used_in_solution', 'boolean'), ('gps_l2_used_in_solution', 'boolean', 'gps_l2_used_in_solution', 'boolean'), ('gps_l3_used_in_solution', 'boolean', 'gps_l3_used_in_solution', 'boolean'), ('glonass_l1_used_in_solution', 'boolean', 'glonass_l1_used_in_solution', 'boolean'), ('glonass_l2_used_in_solution', 'boolean', 'glonass_l2_used_in_solution', 'boolean')], transformation_ctx = 'applymapping2')
    resolvechoice3 = ResolveChoice.apply(frame = applymapping2, choice = 'make_cols', transformation_ctx = 'resolvechoice3')
    dropnullfields4 = DropNullFields.apply(frame = resolvechoice3, transformation_ctx = 'dropnullfields4')
    datasink5 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields4, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_bestpos', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink5')
except:
    print('Failed')

try:
    datasource2 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_can_transmission_state_csv', transformation_ctx = 'datasource2')
    applymapping3 = ApplyMapping.apply(frame = datasource2, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('transmission_state', 'long', 'transmission_state', 'long')], transformation_ctx = 'applymapping3')
    resolvechoice4 = ResolveChoice.apply(frame = applymapping3, choice = 'make_cols', transformation_ctx = 'resolvechoice4')
    dropnullfields5 = DropNullFields.apply(frame = resolvechoice4, transformation_ctx = 'dropnullfields5')
    datasink6 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields5, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_can_transmission_state', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink6')
except:
    print('Failed')

try:
    datasource3 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_speed_pedals_csv', transformation_ctx = 'datasource3')
    applymapping4 = ApplyMapping.apply(frame = datasource3, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('mode', 'long', 'mode', 'long'), ('throttle', 'double', 'throttle', 'double'), ('brake', 'double', 'brake', 'double')], transformation_ctx = 'applymapping4')
    resolvechoice5 = ResolveChoice.apply(frame = applymapping4, choice = 'make_cols', transformation_ctx = 'resolvechoice5')
    dropnullfields6 = DropNullFields.apply(frame = resolvechoice5, transformation_ctx = 'dropnullfields6')
    datasink7 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields6, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_speed_pedals', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink7')
except:
    print('Failed')

try:
    datasource4 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_headlight_rpt_csv', transformation_ctx = 'datasource4')
    applymapping5 = ApplyMapping.apply(frame = datasource4, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enabled', 'boolean', 'enabled', 'boolean'), ('override_active', 'boolean', 'override_active', 'boolean'), ('command_output_fault', 'boolean', 'command_output_fault', 'boolean'), ('input_output_fault', 'boolean', 'input_output_fault', 'boolean'), ('output_reported_fault', 'boolean', 'output_reported_fault', 'boolean'), ('pacmod_fault', 'boolean', 'pacmod_fault', 'boolean'), ('vehicle_fault', 'boolean', 'vehicle_fault', 'boolean'), ('manual_input', 'long', 'manual_input', 'long'), ('command', 'long', 'command', 'long'), ('output', 'long', 'output', 'long')], transformation_ctx = 'applymapping5')
    resolvechoice6 = ResolveChoice.apply(frame = applymapping5, choice = 'make_cols', transformation_ctx = 'resolvechoice6')
    dropnullfields7 = DropNullFields.apply(frame = resolvechoice6, transformation_ctx = 'dropnullfields7')
    datasink8 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields7, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_headlight_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink8')
except:
    print('Failed')

try:
    datasource5 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_comms_inbound_binary_msg_csv', transformation_ctx = 'datasource5')
    applymapping6 = ApplyMapping.apply(frame = datasource5, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('messagetype', 'string', 'messagetype', 'string'), ('content', 'array', 'content', 'string')], transformation_ctx = 'applymapping6')
    resolvechoice7 = ResolveChoice.apply(frame = applymapping6, choice = 'make_cols', transformation_ctx = 'resolvechoice7')
    dropnullfields8 = DropNullFields.apply(frame = resolvechoice7, transformation_ctx = 'dropnullfields8')
    datasink9 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields8, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_comms_inbound_binary_msg', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink9')
except:
    print('Failed')

try:
    datasource6 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_steering_feedback_csv', transformation_ctx = 'datasource6')
    applymapping7 = ApplyMapping.apply(frame = datasource6, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('steering_wheel_angle', 'double', 'steering_wheel_angle', 'double')], transformation_ctx = 'applymapping7')
    resolvechoice8 = ResolveChoice.apply(frame = applymapping7, choice = 'make_cols', transformation_ctx = 'resolvechoice8')
    dropnullfields9 = DropNullFields.apply(frame = resolvechoice8, transformation_ctx = 'dropnullfields9')
    datasink10 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields9, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_steering_feedback', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink10')
except:
    print('Failed')

try:
    datasource7 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_gear_select_csv', transformation_ctx = 'datasource7')
    applymapping8 = ApplyMapping.apply(frame = datasource7, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('command', 'string', 'command', 'string'), ('gear', 'long', 'gear', 'long')], transformation_ctx = 'applymapping8')
    resolvechoice9 = ResolveChoice.apply(frame = applymapping8, choice = 'make_cols', transformation_ctx = 'resolvechoice9')
    dropnullfields10 = DropNullFields.apply(frame = resolvechoice9, transformation_ctx = 'dropnullfields10')
    datasink11 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields10, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_gear_select', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink11')
except:
    print('Failed')

try:
    datasource8 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_veh_interface_config_csv', transformation_ctx = 'datasource8')
    applymapping9 = ApplyMapping.apply(frame = datasource8, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string'), ('tire_radius', 'string', 'tire_radius', 'string')], transformation_ctx = 'applymapping9')
    resolvechoice10 = ResolveChoice.apply(frame = applymapping9, choice = 'make_cols', transformation_ctx = 'resolvechoice10')
    dropnullfields11 = DropNullFields.apply(frame = resolvechoice10, transformation_ctx = 'dropnullfields11')
    datasink12 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields11, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_veh_interface_config', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink12')
except:
    print('Failed')

try:
    datasource9 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_lat_lon_heading_rpt_csv', transformation_ctx = 'datasource9')
    applymapping10 = ApplyMapping.apply(frame = datasource9, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('latitude_degrees', 'long', 'latitude_degrees', 'long'), ('latitude_minutes', 'long', 'latitude_minutes', 'long'), ('latitude_seconds', 'long', 'latitude_seconds', 'long'), ('longitude_degrees', 'long', 'longitude_degrees', 'long'), ('longitude_minutes', 'long', 'longitude_minutes', 'long'), ('longitude_seconds', 'long', 'longitude_seconds', 'long'), ('heading', 'double', 'heading', 'double')], transformation_ctx = 'applymapping10')
    resolvechoice11 = ResolveChoice.apply(frame = applymapping10, choice = 'make_cols', transformation_ctx = 'resolvechoice11')
    dropnullfields12 = DropNullFields.apply(frame = resolvechoice11, transformation_ctx = 'dropnullfields12')
    datasink13 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields12, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_lat_lon_heading_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink13')
except:
    print('Failed')

try:
    datasource10 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='environment_active_geofence_csv', transformation_ctx = 'datasource10')
    applymapping11 = ApplyMapping.apply(frame = datasource10, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('is_on_active_geofence', 'boolean', 'is_on_active_geofence', 'boolean'), ('type', 'long', 'type', 'long'), ('value', 'double', 'value', 'double'), ('distance_to_next_geofence', 'double', 'distance_to_next_geofence', 'double')], transformation_ctx = 'applymapping11')
    resolvechoice12 = ResolveChoice.apply(frame = applymapping11, choice = 'make_cols', transformation_ctx = 'resolvechoice12')
    dropnullfields13 = DropNullFields.apply(frame = resolvechoice12, transformation_ctx = 'dropnullfields13')
    datasink14 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields13, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.environment_active_geofence', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink14')
except:
    print('Failed')

try:
    datasource11 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='localization_ekf_twist_csv', transformation_ctx = 'datasource11')
    applymapping12 = ApplyMapping.apply(frame = datasource11, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('twist', 'string', 'twist', 'string'), ('linear', 'string', 'linear', 'string'), ('x', 'string', 'x', 'string'), ('y', 'double', 'y', 'double'), ('z', 'string', 'z', 'string'), ('angular', 'string', 'angular', 'string')], transformation_ctx = 'applymapping12')
    resolvechoice13 = ResolveChoice.apply(frame = applymapping12, choice = 'make_cols', transformation_ctx = 'resolvechoice13')
    dropnullfields14 = DropNullFields.apply(frame = resolvechoice13, transformation_ctx = 'dropnullfields14')
    datasink15 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields14, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.localization_ekf_twist', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink15')
except:
    print('Failed')

try:
    datasource12 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_steer_rpt_csv', transformation_ctx = 'datasource12')
    applymapping13 = ApplyMapping.apply(frame = datasource12, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enabled', 'boolean', 'enabled', 'boolean'), ('override_active', 'boolean', 'override_active', 'boolean'), ('command_output_fault', 'boolean', 'command_output_fault', 'boolean'), ('input_output_fault', 'boolean', 'input_output_fault', 'boolean'), ('output_reported_fault', 'boolean', 'output_reported_fault', 'boolean'), ('pacmod_fault', 'boolean', 'pacmod_fault', 'boolean'), ('vehicle_fault', 'boolean', 'vehicle_fault', 'boolean'), ('manual_input', 'double', 'manual_input', 'double'), ('command', 'double', 'command', 'double'), ('output', 'double', 'output', 'double')], transformation_ctx = 'applymapping13')
    resolvechoice14 = ResolveChoice.apply(frame = applymapping13, choice = 'make_cols', transformation_ctx = 'resolvechoice14')
    dropnullfields15 = DropNullFields.apply(frame = resolvechoice14, transformation_ctx = 'dropnullfields15')
    datasink16 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields15, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_steer_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink16')
except:
    print('Failed')

try:
    datasource13 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_velocity_accel_cov_csv', transformation_ctx = 'datasource13')
    applymapping14 = ApplyMapping.apply(frame = datasource13, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('velocity', 'double', 'velocity', 'double'), ('accleration', 'double', 'accleration', 'double'), ('covariance', 'double', 'covariance', 'double')], transformation_ctx = 'applymapping14')
    resolvechoice15 = ResolveChoice.apply(frame = applymapping14, choice = 'make_cols', transformation_ctx = 'resolvechoice15')
    dropnullfields16 = DropNullFields.apply(frame = resolvechoice15, transformation_ctx = 'dropnullfields16')
    datasink17 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields16, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_velocity_accel_cov', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink17')
except:
    print('Failed')

try:
    datasource14 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_steering_command_echo_csv', transformation_ctx = 'datasource14')
    applymapping15 = ApplyMapping.apply(frame = datasource14, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('steering_wheel_angle', 'double', 'steering_wheel_angle', 'double')], transformation_ctx = 'applymapping15')
    resolvechoice16 = ResolveChoice.apply(frame = applymapping15, choice = 'make_cols', transformation_ctx = 'resolvechoice16')
    dropnullfields17 = DropNullFields.apply(frame = resolvechoice16, transformation_ctx = 'dropnullfields17')
    datasink18 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields17, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_steering_command_echo', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink18')
except:
    print('Failed')

try:
    datasource15 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_wheel_speed_rpt_csv', transformation_ctx = 'datasource15')
    applymapping16 = ApplyMapping.apply(frame = datasource15, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('front_left_wheel_speed', 'double', 'front_left_wheel_speed', 'double'), ('front_right_wheel_speed', 'double', 'front_right_wheel_speed', 'double'), ('rear_left_wheel_speed', 'double', 'rear_left_wheel_speed', 'double'), ('rear_right_wheel_speed', 'double', 'rear_right_wheel_speed', 'double')], transformation_ctx = 'applymapping16')
    resolvechoice17 = ResolveChoice.apply(frame = applymapping16, choice = 'make_cols', transformation_ctx = 'resolvechoice17')
    dropnullfields18 = DropNullFields.apply(frame = resolvechoice17, transformation_ctx = 'dropnullfields18')
    datasink19 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields18, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_wheel_speed_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink19')
except:
    print('Failed')

try:
    datasource16 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_fix_csv', transformation_ctx = 'datasource16')
    applymapping17 = ApplyMapping.apply(frame = datasource16, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('status', 'long', 'status', 'long'), ('service', 'long', 'service', 'long'), ('latitude', 'double', 'latitude', 'double'), ('longitude', 'double', 'longitude', 'double'), ('altitude', 'double', 'altitude', 'double'), ('position_covariance', 'array', 'position_covariance', 'string'), ('position_covariance_type', 'long', 'position_covariance_type', 'long')], transformation_ctx = 'applymapping17')
    resolvechoice18 = ResolveChoice.apply(frame = applymapping17, choice = 'make_cols', transformation_ctx = 'resolvechoice18')
    dropnullfields19 = DropNullFields.apply(frame = resolvechoice18, transformation_ctx = 'dropnullfields19')
    datasink20 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields19, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_fix', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink20')
except:
    print('Failed')

try:
    datasource17 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_arbitrated_steering_commands_csv', transformation_ctx = 'datasource17')
    applymapping18 = ApplyMapping.apply(frame = datasource17, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('mode', 'long', 'mode', 'long'), ('curvature', 'double', 'curvature', 'double'), ('max_curvature_rate', 'double', 'max_curvature_rate', 'double')], transformation_ctx = 'applymapping18')
    resolvechoice19 = ResolveChoice.apply(frame = applymapping18, choice = 'make_cols', transformation_ctx = 'resolvechoice19')
    dropnullfields20 = DropNullFields.apply(frame = resolvechoice19, transformation_ctx = 'dropnullfields20')
    datasink21 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields20, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_arbitrated_steering_commands', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink21')
except:
    print('Failed')

try:
    datasource18 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_global_rpt_csv', transformation_ctx = 'datasource18')
    applymapping19 = ApplyMapping.apply(frame = datasource18, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enabled', 'boolean', 'enabled', 'boolean'), ('override_active', 'boolean', 'override_active', 'boolean'), ('fault_active', 'boolean', 'fault_active', 'boolean'), ('config_fault_active', 'boolean', 'config_fault_active', 'boolean'), ('user_can_timeout', 'boolean', 'user_can_timeout', 'boolean'), ('brake_can_timeout', 'boolean', 'brake_can_timeout', 'boolean'), ('steering_can_timeout', 'boolean', 'steering_can_timeout', 'boolean'), ('vehicle_can_timeout', 'boolean', 'vehicle_can_timeout', 'boolean'), ('subsystem_can_timeout', 'boolean', 'subsystem_can_timeout', 'boolean'), ('user_can_read_errors', 'boolean', 'user_can_read_errors', 'boolean')], transformation_ctx = 'applymapping19')
    resolvechoice20 = ResolveChoice.apply(frame = applymapping19, choice = 'make_cols', transformation_ctx = 'resolvechoice20')
    dropnullfields21 = DropNullFields.apply(frame = resolvechoice20, transformation_ctx = 'dropnullfields21')
    datasink22 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields21, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_global_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink22')
except:
    print('Failed')

try:
    datasource19 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_inspva_csv', transformation_ctx = 'datasource19')
    applymapping20 = ApplyMapping.apply(frame = datasource19, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('week', 'long', 'week', 'long'), ('seconds', 'double', 'seconds', 'double'), ('latitude', 'double', 'latitude', 'double'), ('longitude', 'double', 'longitude', 'double'), ('height', 'double', 'height', 'double'), ('north_velocity', 'double', 'north_velocity', 'double'), ('east_velocity', 'string', 'east_velocity', 'string'), ('up_velocity', 'string', 'up_velocity', 'string'), ('roll', 'double', 'roll', 'double'), ('pitch', 'double', 'pitch', 'double'), ('azimuth', 'double', 'azimuth', 'double'), ('status', 'string', 'status', 'string')], transformation_ctx = 'applymapping20')
    resolvechoice21 = ResolveChoice.apply(frame = applymapping20, choice = 'make_cols', transformation_ctx = 'resolvechoice21')
    dropnullfields22 = DropNullFields.apply(frame = resolvechoice21, transformation_ctx = 'dropnullfields22')
    datasink23 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields22, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_inspva', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink23')
except:
    print('Failed')

try:
    datasource20 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_brake_aux_rpt_csv', transformation_ctx = 'datasource20')
    applymapping21 = ApplyMapping.apply(frame = datasource20, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('raw_pedal_pos', 'double', 'raw_pedal_pos', 'double'), ('raw_pedal_pos_is_valid', 'boolean', 'raw_pedal_pos_is_valid', 'boolean'), ('raw_pedal_force', 'double', 'raw_pedal_force', 'double'), ('raw_pedal_force_is_valid', 'boolean', 'raw_pedal_force_is_valid', 'boolean'), ('raw_brake_pressure', 'double', 'raw_brake_pressure', 'double'), ('raw_brake_pressure_is_valid', 'boolean', 'raw_brake_pressure_is_valid', 'boolean'), ('brake_on_off', 'boolean', 'brake_on_off', 'boolean'), ('brake_on_off_is_valid', 'boolean', 'brake_on_off_is_valid', 'boolean'), ('user_interaction', 'boolean', 'user_interaction', 'boolean'), ('user_interaction_is_valid', 'boolean', 'user_interaction_is_valid', 'boolean')], transformation_ctx = 'applymapping21')
    resolvechoice22 = ResolveChoice.apply(frame = applymapping21, choice = 'make_cols', transformation_ctx = 'resolvechoice22')
    dropnullfields23 = DropNullFields.apply(frame = resolvechoice22, transformation_ctx = 'dropnullfields23')
    datasink24 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields23, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_brake_aux_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink24')
except:
    print('Failed')

try:
    datasource21 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_velodyne_nodelet_manager_bond_csv', transformation_ctx = 'datasource21')
    applymapping22 = ApplyMapping.apply(frame = datasource21, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('id', 'string', 'id', 'string'), ('instance_id', 'string', 'instance_id', 'string'), ('active', 'boolean', 'active', 'boolean'), ('heartbeat_timeout', 'double', 'heartbeat_timeout', 'double'), ('heartbeat_period', 'double', 'heartbeat_period', 'double')], transformation_ctx = 'applymapping22')
    resolvechoice23 = ResolveChoice.apply(frame = applymapping22, choice = 'make_cols', transformation_ctx = 'resolvechoice23')
    dropnullfields24 = DropNullFields.apply(frame = resolvechoice23, transformation_ctx = 'dropnullfields24')
    datasink25 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields24, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_velodyne_nodelet_manager_bond', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink25')
except:
    print('Failed')

try:
    datasource22 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_steering_model_config_csv', transformation_ctx = 'datasource22')
    applymapping23 = ApplyMapping.apply(frame = datasource22, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string'), ('velocity_timeout', 'string', 'velocity_timeout', 'string'), ('slip_map', 'string', 'slip_map', 'string'), ('initiation_rl_map_speed', 'string', 'initiation_rl_map_speed', 'string'), ('initiation_rl_map_fast', 'string', 'initiation_rl_map_fast', 'string')], transformation_ctx = 'applymapping23')
    resolvechoice24 = ResolveChoice.apply(frame = applymapping23, choice = 'make_cols', transformation_ctx = 'resolvechoice24')
    dropnullfields25 = DropNullFields.apply(frame = resolvechoice24, transformation_ctx = 'dropnullfields25')
    datasink26 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields25, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_steering_model_config', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink26')
except:
    print('Failed')

try:
    datasource23 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_vehicle_twist_csv', transformation_ctx = 'datasource23')
    applymapping24 = ApplyMapping.apply(frame = datasource23, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('twist', 'string', 'twist', 'string'), ('linear', 'string', 'linear', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'string', 'z', 'string'), ('angular', 'string', 'angular', 'string')], transformation_ctx = 'applymapping24')
    resolvechoice25 = ResolveChoice.apply(frame = applymapping24, choice = 'make_cols', transformation_ctx = 'resolvechoice25')
    dropnullfields26 = DropNullFields.apply(frame = resolvechoice25, transformation_ctx = 'dropnullfields26')
    datasink27 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields26, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_vehicle_twist', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink27')
except:
    print('Failed')

try:
    datasource24 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_corrimudata_csv', transformation_ctx = 'datasource24')
    applymapping25 = ApplyMapping.apply(frame = datasource24, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('pitch_rate', 'string', 'pitch_rate', 'string'), ('roll_rate', 'string', 'roll_rate', 'string'), ('yaw_rate', 'string', 'yaw_rate', 'string'), ('lateral_acceleration', 'string', 'lateral_acceleration', 'string'), ('longitudinal_acceleration', 'string', 'longitudinal_acceleration', 'string'), ('vertical_acceleration', 'string', 'vertical_acceleration', 'string')], transformation_ctx = 'applymapping25')
    resolvechoice26 = ResolveChoice.apply(frame = applymapping25, choice = 'make_cols', transformation_ctx = 'resolvechoice26')
    dropnullfields27 = DropNullFields.apply(frame = resolvechoice26, transformation_ctx = 'dropnullfields27')
    datasink28 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields27, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_corrimudata', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink28')
except:
    print('Failed')

try:
    datasource25 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_controller_robot_status_csv', transformation_ctx = 'datasource25')
    applymapping26 = ApplyMapping.apply(frame = datasource25, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('robot_active', 'boolean', 'robot_active', 'boolean'), ('robot_enabled', 'boolean', 'robot_enabled', 'boolean'), ('torque', 'double', 'torque', 'double'), ('torque_validity', 'boolean', 'torque_validity', 'boolean'), ('brake_decel', 'double', 'brake_decel', 'double'), ('brake_decel_validity', 'boolean', 'brake_decel_validity', 'boolean'), ('throttle_effort', 'double', 'throttle_effort', 'double'), ('throttle_effort_validity', 'boolean', 'throttle_effort_validity', 'boolean'), ('braking_effort', 'double', 'braking_effort', 'double'), ('braking_effort_validity', 'boolean', 'braking_effort_validity', 'boolean')], transformation_ctx = 'applymapping26')
    resolvechoice27 = ResolveChoice.apply(frame = applymapping26, choice = 'make_cols', transformation_ctx = 'resolvechoice27')
    dropnullfields28 = DropNullFields.apply(frame = resolvechoice27, transformation_ctx = 'dropnullfields28')
    datasink29 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields28, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_controller_robot_status', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink29')
except:
    print('Failed')

try:
    datasource26 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_route_event_csv', transformation_ctx = 'datasource26')
    applymapping27 = ApplyMapping.apply(frame = datasource26, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('event', 'long', 'event', 'long')], transformation_ctx = 'applymapping27')
    resolvechoice28 = ResolveChoice.apply(frame = applymapping27, choice = 'make_cols', transformation_ctx = 'resolvechoice28')
    dropnullfields29 = DropNullFields.apply(frame = resolvechoice28, transformation_ctx = 'dropnullfields29')
    datasink30 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields29, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_route_event', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink30')
except:
    print('Failed')

try:
    datasource27 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_gear_feedback_csv', transformation_ctx = 'datasource27')
    applymapping28 = ApplyMapping.apply(frame = datasource27, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('current_gear', 'string', 'current_gear', 'string'), ('gear', 'long', 'gear', 'long')], transformation_ctx = 'applymapping28')
    resolvechoice29 = ResolveChoice.apply(frame = applymapping28, choice = 'make_cols', transformation_ctx = 'resolvechoice29')
    dropnullfields30 = DropNullFields.apply(frame = resolvechoice29, transformation_ctx = 'dropnullfields30')
    datasink31 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields30, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_gear_feedback', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink31')
except:
    print('Failed')

try:
    datasource28 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_curvature_feedback_csv', transformation_ctx = 'datasource28')
    applymapping29 = ApplyMapping.apply(frame = datasource28, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('curvature', 'string', 'curvature', 'string')], transformation_ctx = 'applymapping29')
    resolvechoice30 = ResolveChoice.apply(frame = applymapping29, choice = 'make_cols', transformation_ctx = 'resolvechoice30')
    dropnullfields31 = DropNullFields.apply(frame = resolvechoice30, transformation_ctx = 'dropnullfields31')
    datasink32 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields31, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_curvature_feedback', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink32')
except:
    print('Failed')

try:
    datasource29 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_vehicle_platform_csv', transformation_ctx = 'datasource29')
    applymapping30 = ApplyMapping.apply(frame = datasource29, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping30')
    resolvechoice31 = ResolveChoice.apply(frame = applymapping30, choice = 'make_cols', transformation_ctx = 'resolvechoice31')
    dropnullfields32 = DropNullFields.apply(frame = resolvechoice31, transformation_ctx = 'dropnullfields32')
    datasink33 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields32, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_vehicle_platform', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink33')
except:
    print('Failed')

try:
    datasource30 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_can_steering_wheel_angle_csv', transformation_ctx = 'datasource30')
    applymapping31 = ApplyMapping.apply(frame = datasource30, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping31')
    resolvechoice32 = ResolveChoice.apply(frame = applymapping31, choice = 'make_cols', transformation_ctx = 'resolvechoice32')
    dropnullfields33 = DropNullFields.apply(frame = resolvechoice32, transformation_ctx = 'dropnullfields33')
    datasink34 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields33, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_can_steering_wheel_angle', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink34')
except:
    print('Failed')

try:
    datasource31 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_as_tx_enabled_csv', transformation_ctx = 'datasource31')
    applymapping32 = ApplyMapping.apply(frame = datasource31, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'boolean', 'data', 'boolean')], transformation_ctx = 'applymapping32')
    resolvechoice33 = ResolveChoice.apply(frame = applymapping32, choice = 'make_cols', transformation_ctx = 'resolvechoice33')
    dropnullfields34 = DropNullFields.apply(frame = resolvechoice33, transformation_ctx = 'dropnullfields34')
    datasink35 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields34, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_as_tx_enabled', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink35')
except:
    print('Failed')

try:
    datasource32 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_gnss_fix_fused_csv', transformation_ctx = 'datasource32')
    applymapping33 = ApplyMapping.apply(frame = datasource32, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('status', 'long', 'status', 'long'), ('satellites_used', 'long', 'satellites_used', 'long'), ('satellite_used_prn', 'array', 'satellite_used_prn', 'string'), ('satellites_visible', 'long', 'satellites_visible', 'long'), ('satellite_visible_prn', 'array', 'satellite_visible_prn', 'string'), ('satellite_visible_z', 'array', 'satellite_visible_z', 'string'), ('satellite_visible_azimuth', 'array', 'satellite_visible_azimuth', 'string'), ('satellite_visible_snr', 'array', 'satellite_visible_snr', 'string'), ('motion_source', 'long', 'motion_source', 'long'), ('orientation_source', 'long', 'orientation_source', 'long'), ('position_source', 'long', 'position_source', 'long'), ('latitude', 'double', 'latitude', 'double'), ('longitude', 'double', 'longitude', 'double'), ('altitude', 'double', 'altitude', 'double'), ('track', 'double', 'track', 'double'), ('speed', 'double', 'speed', 'double'), ('climb', 'string', 'climb', 'string'), ('pitch', 'double', 'pitch', 'double'), ('roll', 'double', 'roll', 'double'), ('dip', 'double', 'dip', 'double'), ('time', 'double', 'time', 'double'), ('gdop', 'double', 'gdop', 'double'), ('pdop', 'double', 'pdop', 'double'), ('hdop', 'double', 'hdop', 'double'), ('vdop', 'double', 'vdop', 'double'), ('tdop', 'double', 'tdop', 'double'), ('err', 'double', 'err', 'double'), ('err_horz', 'double', 'err_horz', 'double'), ('err_vert', 'double', 'err_vert', 'double'), ('err_track', 'double', 'err_track', 'double'), ('err_speed', 'string', 'err_speed', 'string'), ('err_climb', 'double', 'err_climb', 'double'), ('err_time', 'double', 'err_time', 'double'), ('err_pitch', 'double', 'err_pitch', 'double'), ('err_roll', 'double', 'err_roll', 'double'), ('err_dip', 'double', 'err_dip', 'double'), ('position_covariance', 'array', 'position_covariance', 'string'), ('position_covariance_type', 'long', 'position_covariance_type', 'long')], transformation_ctx = 'applymapping33')
    resolvechoice34 = ResolveChoice.apply(frame = applymapping33, choice = 'make_cols', transformation_ctx = 'resolvechoice34')
    dropnullfields35 = DropNullFields.apply(frame = resolvechoice34, transformation_ctx = 'dropnullfields35')
    datasink36 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields35, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_gnss_fix_fused', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink36')
except:
    print('Failed')

try:
    datasource33 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_plugin_discovery_csv', transformation_ctx = 'datasource33')
    applymapping34 = ApplyMapping.apply(frame = datasource33, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('name', 'string', 'name', 'string'), ('versionid', 'string', 'versionid', 'string'), ('type', 'long', 'type', 'long'), ('available', 'boolean', 'available', 'boolean'), ('activated', 'boolean', 'activated', 'boolean'), ('capability', 'string', 'capability', 'string')], transformation_ctx = 'applymapping34')
    resolvechoice35 = ResolveChoice.apply(frame = applymapping34, choice = 'make_cols', transformation_ctx = 'resolvechoice35')
    dropnullfields36 = DropNullFields.apply(frame = resolvechoice35, transformation_ctx = 'dropnullfields36')
    datasink37 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields36, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_plugin_discovery', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink37')
except:
    print('Failed')

try:
    datasource34 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_insstdev_csv', transformation_ctx = 'datasource34')
    applymapping35 = ApplyMapping.apply(frame = datasource34, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('latitude_dev', 'double', 'latitude_dev', 'double'), ('longitude_dev', 'double', 'longitude_dev', 'double'), ('height_dev', 'double', 'height_dev', 'double'), ('north_velocity_dev', 'double', 'north_velocity_dev', 'double'), ('east_velocity_dev', 'double', 'east_velocity_dev', 'double'), ('up_velocity_dev', 'double', 'up_velocity_dev', 'double'), ('roll_dev', 'double', 'roll_dev', 'double'), ('pitch_dev', 'double', 'pitch_dev', 'double'), ('azimuth_dev', 'double', 'azimuth_dev', 'double'), ('extended_solution_status', 'string', 'extended_solution_status', 'string'), ('original_mask', 'long', 'original_mask', 'long'), ('advance_rtk_verified', 'boolean', 'advance_rtk_verified', 'boolean'), ('psuedorange_iono_correction', 'string', 'psuedorange_iono_correction', 'string'), ('time_since_update', 'long', 'time_since_update', 'long')], transformation_ctx = 'applymapping35')
    resolvechoice36 = ResolveChoice.apply(frame = applymapping35, choice = 'make_cols', transformation_ctx = 'resolvechoice36')
    dropnullfields37 = DropNullFields.apply(frame = resolvechoice36, transformation_ctx = 'dropnullfields37')
    datasink38 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields37, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_insstdev', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink38')
except:
    print('Failed')

try:
    datasource35 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_as_rx_shift_cmd_csv', transformation_ctx = 'datasource35')
    applymapping36 = ApplyMapping.apply(frame = datasource35, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enable', 'boolean', 'enable', 'boolean'), ('ignore_overrides', 'boolean', 'ignore_overrides', 'boolean'), ('clear_override', 'boolean', 'clear_override', 'boolean'), ('clear_faults', 'boolean', 'clear_faults', 'boolean'), ('command', 'long', 'command', 'long')], transformation_ctx = 'applymapping36')
    resolvechoice37 = ResolveChoice.apply(frame = applymapping36, choice = 'make_cols', transformation_ctx = 'resolvechoice37')
    dropnullfields38 = DropNullFields.apply(frame = resolvechoice37, transformation_ctx = 'dropnullfields38')
    datasink39 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields38, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_as_rx_shift_cmd', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink39')
except:
    print('Failed')

try:
    datasource36 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='localization_ekf_pose_with_covariance_csv', transformation_ctx = 'datasource36')
    applymapping37 = ApplyMapping.apply(frame = datasource36, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('pose', 'string', 'pose', 'string'), ('position', 'string', 'position', 'string'), ('x', 'string', 'x', 'string'), ('y', 'string', 'y', 'string'), ('z', 'double', 'z', 'double'), ('orientation', 'string', 'orientation', 'string'), ('w', 'double', 'w', 'double'), ('covariance', 'array', 'covariance', 'string')], transformation_ctx = 'applymapping37')
    resolvechoice38 = ResolveChoice.apply(frame = applymapping37, choice = 'make_cols', transformation_ctx = 'resolvechoice38')
    dropnullfields39 = DropNullFields.apply(frame = resolvechoice38, transformation_ctx = 'dropnullfields39')
    datasink40 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields39, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.localization_ekf_pose_with_covariance', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink40')
except:
    print('Failed')

try:
    datasource37 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_brake_feedback_csv', transformation_ctx = 'datasource37')
    applymapping38 = ApplyMapping.apply(frame = datasource37, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('brake_pedal', 'double', 'brake_pedal', 'double')], transformation_ctx = 'applymapping38')
    resolvechoice39 = ResolveChoice.apply(frame = applymapping38, choice = 'make_cols', transformation_ctx = 'resolvechoice39')
    dropnullfields40 = DropNullFields.apply(frame = resolvechoice39, transformation_ctx = 'dropnullfields40')
    datasink41 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields40, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_brake_feedback', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink41')
except:
    print('Failed')

try:
    datasource38 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_speed_model_config_csv', transformation_ctx = 'datasource38')
    applymapping39 = ApplyMapping.apply(frame = datasource38, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string'), ('min_throttle', 'string', 'min_throttle', 'string'), ('max_brake', 'string', 'max_brake', 'string'), ('command_filter', 'string', 'command_filter', 'string'), ('negative_p_gain_map', 'string', 'negative_p_gain_map', 'string'), ('max_p_term_map', 'string', 'max_p_term_map', 'string'), ('negative_i_gain_map', 'string', 'negative_i_gain_map', 'string'), ('max_i_term_map', 'string', 'max_i_term_map', 'string'), ('error_based_max_i_term_map', 'string', 'error_based_max_i_term_map', 'string'), ('positive_d_gain_map', 'string', 'positive_d_gain_map', 'string'), ('min_d_term_map', 'string', 'min_d_term_map', 'string'), ('d_gain_braking', 'string', 'd_gain_braking', 'string'), ('brake_ramp_on_min', 'string', 'brake_ramp_on_min', 'string'), ('stopped_ramp_rate', 'string', 'stopped_ramp_rate', 'string'), ('brake_mode_exit_i_term', 'string', 'brake_mode_exit_i_term', 'string'), ('brake_mode_enter_i_term_init', 'string', 'brake_mode_enter_i_term_init', 'string'), ('brake_mode_i_boost_map', 'string', 'brake_mode_i_boost_map', 'string'), ('brake_mode_p_min', 'string', 'brake_mode_p_min', 'string'), ('brake_mode_i_min', 'string', 'brake_mode_i_min', 'string'), ('brake_mode_d_max', 'string', 'brake_mode_d_max', 'string'), ('throttle_increasing_rate_limit_map', 'string', 'throttle_increasing_rate_limit_map', 'string'), ('quick_stop_mode_max', 'string', 'quick_stop_mode_max', 'string')], transformation_ctx = 'applymapping39')
    resolvechoice40 = ResolveChoice.apply(frame = applymapping39, choice = 'make_cols', transformation_ctx = 'resolvechoice40')
    dropnullfields41 = DropNullFields.apply(frame = resolvechoice40, transformation_ctx = 'dropnullfields41')
    datasink42 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields41, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_speed_model_config', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink42')
except:
    print('Failed')

try:
    datasource39 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_date_time_rpt_csv', transformation_ctx = 'datasource39')
    applymapping40 = ApplyMapping.apply(frame = datasource39, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('year', 'long', 'year', 'long'), ('month', 'long', 'month', 'long'), ('day', 'long', 'day', 'long'), ('hour', 'long', 'hour', 'long'), ('minute', 'long', 'minute', 'long'), ('second', 'long', 'second', 'long')], transformation_ctx = 'applymapping40')
    resolvechoice41 = ResolveChoice.apply(frame = applymapping40, choice = 'make_cols', transformation_ctx = 'resolvechoice41')
    dropnullfields42 = DropNullFields.apply(frame = resolvechoice41, transformation_ctx = 'dropnullfields42')
    datasink43 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields42, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_date_time_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink43')
except:
    print('Failed')

try:
    datasource40 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_can_rx_csv', transformation_ctx = 'datasource40')
    applymapping41 = ApplyMapping.apply(frame = datasource40, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('id', 'long', 'id', 'long'), ('is_rtr', 'boolean', 'is_rtr', 'boolean'), ('is_extended', 'boolean', 'is_extended', 'boolean'), ('is_error', 'boolean', 'is_error', 'boolean'), ('dlc', 'long', 'dlc', 'long'), ('data', 'array', 'data', 'string')], transformation_ctx = 'applymapping41')
    resolvechoice42 = ResolveChoice.apply(frame = applymapping41, choice = 'make_cols', transformation_ctx = 'resolvechoice42')
    dropnullfields43 = DropNullFields.apply(frame = resolvechoice42, transformation_ctx = 'dropnullfields43')
    datasink44 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields43, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_can_rx', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink44')
except:
    print('Failed')

try:
    datasource41 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_state_csv', transformation_ctx = 'datasource41')
    applymapping42 = ApplyMapping.apply(frame = datasource41, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('state', 'long', 'state', 'long')], transformation_ctx = 'applymapping42')
    resolvechoice43 = ResolveChoice.apply(frame = applymapping42, choice = 'make_cols', transformation_ctx = 'resolvechoice43')
    dropnullfields44 = DropNullFields.apply(frame = resolvechoice43, transformation_ctx = 'dropnullfields44')
    datasink45 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields44, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_state', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink45')
except:
    print('Failed')

try:
    datasource42 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_inscov_csv', transformation_ctx = 'datasource42')
    applymapping43 = ApplyMapping.apply(frame = datasource42, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('week', 'long', 'week', 'long'), ('seconds', 'double', 'seconds', 'double'), ('position_covariance', 'array', 'position_covariance', 'string'), ('attitude_covariance', 'array', 'attitude_covariance', 'string'), ('velocity_covariance', 'array', 'velocity_covariance', 'string')], transformation_ctx = 'applymapping43')
    resolvechoice44 = ResolveChoice.apply(frame = applymapping43, choice = 'make_cols', transformation_ctx = 'resolvechoice44')
    dropnullfields45 = DropNullFields.apply(frame = resolvechoice44, transformation_ctx = 'dropnullfields45')
    datasink46 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields45, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_inscov', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink46')
except:
    print('Failed')

try:
    datasource43 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_accel_aux_rpt_csv', transformation_ctx = 'datasource43')
    applymapping44 = ApplyMapping.apply(frame = datasource43, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('raw_pedal_pos', 'double', 'raw_pedal_pos', 'double'), ('raw_pedal_pos_is_valid', 'boolean', 'raw_pedal_pos_is_valid', 'boolean'), ('raw_pedal_force', 'double', 'raw_pedal_force', 'double'), ('raw_pedal_force_is_valid', 'boolean', 'raw_pedal_force_is_valid', 'boolean'), ('user_interaction', 'boolean', 'user_interaction', 'boolean'), ('user_interaction_is_valid', 'boolean', 'user_interaction_is_valid', 'boolean'), ('brake_interlock_active', 'boolean', 'brake_interlock_active', 'boolean'), ('brake_interlock_active_is_valid', 'boolean', 'brake_interlock_active_is_valid', 'boolean')], transformation_ctx = 'applymapping44')
    resolvechoice45 = ResolveChoice.apply(frame = applymapping44, choice = 'make_cols', transformation_ctx = 'resolvechoice45')
    dropnullfields46 = DropNullFields.apply(frame = resolvechoice45, transformation_ctx = 'dropnullfields46')
    datasink47 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields46, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_accel_aux_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink47')
except:
    print('Failed')

try:
    datasource44 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_steering_wheel_csv', transformation_ctx = 'datasource44')
    applymapping45 = ApplyMapping.apply(frame = datasource44, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('mode', 'long', 'mode', 'long'), ('angle', 'double', 'angle', 'double'), ('angle_velocity', 'double', 'angle_velocity', 'double')], transformation_ctx = 'applymapping45')
    resolvechoice46 = ResolveChoice.apply(frame = applymapping45, choice = 'make_cols', transformation_ctx = 'resolvechoice46')
    dropnullfields47 = DropNullFields.apply(frame = resolvechoice46, transformation_ctx = 'dropnullfields47')
    datasink48 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields47, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_steering_wheel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink48')
except:
    print('Failed')

try:
    datasource45 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_gear_command_echo_csv', transformation_ctx = 'datasource45')
    applymapping46 = ApplyMapping.apply(frame = datasource45, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('command', 'string', 'command', 'string'), ('gear', 'long', 'gear', 'long')], transformation_ctx = 'applymapping46')
    resolvechoice47 = ResolveChoice.apply(frame = applymapping46, choice = 'make_cols', transformation_ctx = 'resolvechoice47')
    dropnullfields48 = DropNullFields.apply(frame = resolvechoice47, transformation_ctx = 'dropnullfields48')
    datasink49 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields48, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_gear_command_echo', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink49')
except:
    print('Failed')

try:
    datasource46 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_can_brake_position_csv', transformation_ctx = 'datasource46')
    applymapping47 = ApplyMapping.apply(frame = datasource46, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping47')
    resolvechoice48 = ResolveChoice.apply(frame = applymapping47, choice = 'make_cols', transformation_ctx = 'resolvechoice48')
    dropnullfields49 = DropNullFields.apply(frame = resolvechoice48, transformation_ctx = 'dropnullfields49')
    datasink50 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields49, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_can_brake_position', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink50')
except:
    print('Failed')

try:
    datasource47 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_gpgga_csv', transformation_ctx = 'datasource47')
    applymapping48 = ApplyMapping.apply(frame = datasource47, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('message_id', 'string', 'message_id', 'string'), ('utc_seconds', 'double', 'utc_seconds', 'double'), ('lat', 'double', 'lat', 'double'), ('lon', 'double', 'lon', 'double'), ('lat_dir', 'string', 'lat_dir', 'string'), ('lon_dir', 'string', 'lon_dir', 'string'), ('gps_qual', 'long', 'gps_qual', 'long'), ('num_sats', 'long', 'num_sats', 'long'), ('hdop', 'double', 'hdop', 'double'), ('alt', 'double', 'alt', 'double'), ('altitude_units', 'string', 'altitude_units', 'string'), ('undulation', 'double', 'undulation', 'double'), ('undulation_units', 'string', 'undulation_units', 'string'), ('diff_age', 'long', 'diff_age', 'long'), ('station_id', 'string', 'station_id', 'string')], transformation_ctx = 'applymapping48')
    resolvechoice49 = ResolveChoice.apply(frame = applymapping48, choice = 'make_cols', transformation_ctx = 'resolvechoice49')
    dropnullfields50 = DropNullFields.apply(frame = resolvechoice49, transformation_ctx = 'dropnullfields50')
    datasink51 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields50, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_gpgga', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink51')
except:
    print('Failed')

try:
    datasource48 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_module_states_csv', transformation_ctx = 'datasource48')
    applymapping49 = ApplyMapping.apply(frame = datasource48, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('name', 'string', 'name', 'string'), ('state', 'string', 'state', 'string'), ('info', 'string', 'info', 'string')], transformation_ctx = 'applymapping49')
    resolvechoice50 = ResolveChoice.apply(frame = applymapping49, choice = 'make_cols', transformation_ctx = 'resolvechoice50')
    dropnullfields51 = DropNullFields.apply(frame = resolvechoice50, transformation_ctx = 'dropnullfields51')
    datasink52 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields51, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_module_states', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink52')
except:
    print('Failed')

try:
    datasource49 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_component_rpt_csv', transformation_ctx = 'datasource49')
    applymapping50 = ApplyMapping.apply(frame = datasource49, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('component_type', 'long', 'component_type', 'long'), ('component_func', 'long', 'component_func', 'long'), ('counter', 'long', 'counter', 'long'), ('complement', 'long', 'complement', 'long'), ('config_fault', 'boolean', 'config_fault', 'boolean')], transformation_ctx = 'applymapping50')
    resolvechoice51 = ResolveChoice.apply(frame = applymapping50, choice = 'make_cols', transformation_ctx = 'resolvechoice51')
    dropnullfields52 = DropNullFields.apply(frame = resolvechoice51, transformation_ctx = 'dropnullfields52')
    datasink53 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields52, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_component_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink53')
except:
    print('Failed')

try:
    datasource50 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_route_state_csv', transformation_ctx = 'datasource50')
    applymapping51 = ApplyMapping.apply(frame = datasource50, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('routeid', 'string', 'routeid', 'string'), ('state', 'long', 'state', 'long'), ('cross_track', 'double', 'cross_track', 'double'), ('down_track', 'double', 'down_track', 'double'), ('lanelet_downtrack', 'string', 'lanelet_downtrack', 'string'), ('lanelet_id', 'long', 'lanelet_id', 'long'), ('speed_limit', 'double', 'speed_limit', 'double')], transformation_ctx = 'applymapping51')
    resolvechoice52 = ResolveChoice.apply(frame = applymapping51, choice = 'make_cols', transformation_ctx = 'resolvechoice52')
    dropnullfields53 = DropNullFields.apply(frame = resolvechoice52, transformation_ctx = 'dropnullfields53')
    datasink54 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields53, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_route_state', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink54')
except:
    print('Failed')

try:
    datasource51 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_search_circle_mark_csv', transformation_ctx = 'datasource51')
    applymapping52 = ApplyMapping.apply(frame = datasource51, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('ns', 'string', 'ns', 'string'), ('id', 'long', 'id', 'long'), ('type', 'long', 'type', 'long'), ('action', 'long', 'action', 'long'), ('pose', 'string', 'pose', 'string'), ('position', 'string', 'position', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'double', 'z', 'double'), ('orientation', 'string', 'orientation', 'string'), ('w', 'double', 'w', 'double'), ('scale', 'string', 'scale', 'string'), ('color', 'string', 'color', 'string'), ('r', 'double', 'r', 'double'), ('g', 'double', 'g', 'double'), ('b', 'double', 'b', 'double'), ('a', 'double', 'a', 'double'), ('lifetime', 'string', 'lifetime', 'string'), ('frame_locked', 'boolean', 'frame_locked', 'boolean'), ('points', 'array', 'points', 'string'), ('colors', 'array', 'colors', 'string'), ('text', 'string', 'text', 'string'), ('mesh_resource', 'string', 'mesh_resource', 'string'), ('mesh_use_embedded_materials', 'boolean', 'mesh_use_embedded_materials', 'boolean')], transformation_ctx = 'applymapping52')
    resolvechoice53 = ResolveChoice.apply(frame = applymapping52, choice = 'make_cols', transformation_ctx = 'resolvechoice53')
    dropnullfields54 = DropNullFields.apply(frame = resolvechoice53, transformation_ctx = 'dropnullfields54')
    datasink55 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields54, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_search_circle_mark', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink55')
except:
    print('Failed')

try:
    datasource52 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_as_rx_steer_cmd_csv', transformation_ctx = 'datasource52')
    applymapping53 = ApplyMapping.apply(frame = datasource52, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enable', 'boolean', 'enable', 'boolean'), ('ignore_overrides', 'boolean', 'ignore_overrides', 'boolean'), ('clear_override', 'boolean', 'clear_override', 'boolean'), ('clear_faults', 'boolean', 'clear_faults', 'boolean'), ('command', 'double', 'command', 'double'), ('rotation_rate', 'double', 'rotation_rate', 'double')], transformation_ctx = 'applymapping53')
    resolvechoice54 = ResolveChoice.apply(frame = applymapping53, choice = 'make_cols', transformation_ctx = 'resolvechoice54')
    dropnullfields55 = DropNullFields.apply(frame = resolvechoice54, transformation_ctx = 'dropnullfields55')
    datasink56 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields55, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_as_rx_steer_cmd', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink56')
except:
    print('Failed')

try:
    datasource53 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_horn_rpt_csv', transformation_ctx = 'datasource53')
    applymapping54 = ApplyMapping.apply(frame = datasource53, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enabled', 'boolean', 'enabled', 'boolean'), ('override_active', 'boolean', 'override_active', 'boolean'), ('command_output_fault', 'boolean', 'command_output_fault', 'boolean'), ('input_output_fault', 'boolean', 'input_output_fault', 'boolean'), ('output_reported_fault', 'boolean', 'output_reported_fault', 'boolean'), ('pacmod_fault', 'boolean', 'pacmod_fault', 'boolean'), ('vehicle_fault', 'boolean', 'vehicle_fault', 'boolean'), ('manual_input', 'boolean', 'manual_input', 'boolean'), ('command', 'boolean', 'command', 'boolean'), ('output', 'boolean', 'output', 'boolean')], transformation_ctx = 'applymapping54')
    resolvechoice55 = ResolveChoice.apply(frame = applymapping54, choice = 'make_cols', transformation_ctx = 'resolvechoice55')
    dropnullfields56 = DropNullFields.apply(frame = resolvechoice55, transformation_ctx = 'dropnullfields56')
    datasink57 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields56, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_horn_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink57')
except:
    print('Failed')

try:
    datasource54 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_result_ctrl_lateral_jerk_csv', transformation_ctx = 'datasource54')
    applymapping55 = ApplyMapping.apply(frame = datasource54, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping55')
    resolvechoice56 = ResolveChoice.apply(frame = applymapping55, choice = 'make_cols', transformation_ctx = 'resolvechoice56')
    dropnullfields57 = DropNullFields.apply(frame = resolvechoice56, transformation_ctx = 'dropnullfields57')
    datasink58 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields57, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_result_ctrl_lateral_jerk', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink58')
except:
    print('Failed')

try:
    datasource55 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_nodelet_manager_bond_csv', transformation_ctx = 'datasource55')
    applymapping56 = ApplyMapping.apply(frame = datasource55, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('id', 'string', 'id', 'string'), ('instance_id', 'string', 'instance_id', 'string'), ('active', 'boolean', 'active', 'boolean'), ('heartbeat_timeout', 'double', 'heartbeat_timeout', 'double'), ('heartbeat_period', 'double', 'heartbeat_period', 'double')], transformation_ctx = 'applymapping56')
    resolvechoice57 = ResolveChoice.apply(frame = applymapping56, choice = 'make_cols', transformation_ctx = 'resolvechoice57')
    dropnullfields58 = DropNullFields.apply(frame = resolvechoice57, transformation_ctx = 'dropnullfields58')
    datasink59 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields58, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_nodelet_manager_bond', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink59')
except:
    print('Failed')

try:
    datasource56 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_yaw_rate_rpt_csv', transformation_ctx = 'datasource56')
    applymapping57 = ApplyMapping.apply(frame = datasource56, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('yaw_rate', 'double', 'yaw_rate', 'double')], transformation_ctx = 'applymapping57')
    resolvechoice58 = ResolveChoice.apply(frame = applymapping57, choice = 'make_cols', transformation_ctx = 'resolvechoice58')
    dropnullfields59 = DropNullFields.apply(frame = resolvechoice58, transformation_ctx = 'dropnullfields59')
    datasink60 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields59, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_yaw_rate_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink60')
except:
    print('Failed')

try:
    datasource57 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_gps_csv', transformation_ctx = 'datasource57')
    applymapping58 = ApplyMapping.apply(frame = datasource57, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('status', 'long', 'status', 'long'), ('satellites_used', 'long', 'satellites_used', 'long'), ('satellite_used_prn', 'array', 'satellite_used_prn', 'string'), ('satellites_visible', 'long', 'satellites_visible', 'long'), ('satellite_visible_prn', 'array', 'satellite_visible_prn', 'string'), ('satellite_visible_z', 'array', 'satellite_visible_z', 'string'), ('satellite_visible_azimuth', 'array', 'satellite_visible_azimuth', 'string'), ('satellite_visible_snr', 'array', 'satellite_visible_snr', 'string'), ('motion_source', 'long', 'motion_source', 'long'), ('orientation_source', 'long', 'orientation_source', 'long'), ('position_source', 'long', 'position_source', 'long'), ('latitude', 'double', 'latitude', 'double'), ('longitude', 'double', 'longitude', 'double'), ('altitude', 'double', 'altitude', 'double'), ('track', 'double', 'track', 'double'), ('speed', 'double', 'speed', 'double'), ('climb', 'double', 'climb', 'double'), ('pitch', 'double', 'pitch', 'double'), ('roll', 'double', 'roll', 'double'), ('dip', 'double', 'dip', 'double'), ('time', 'double', 'time', 'double'), ('gdop', 'double', 'gdop', 'double'), ('pdop', 'double', 'pdop', 'double'), ('hdop', 'double', 'hdop', 'double'), ('vdop', 'double', 'vdop', 'double'), ('tdop', 'double', 'tdop', 'double'), ('err', 'double', 'err', 'double'), ('err_horz', 'double', 'err_horz', 'double'), ('err_vert', 'double', 'err_vert', 'double'), ('err_track', 'double', 'err_track', 'double'), ('err_speed', 'double', 'err_speed', 'double'), ('err_climb', 'double', 'err_climb', 'double'), ('err_time', 'double', 'err_time', 'double'), ('err_pitch', 'double', 'err_pitch', 'double'), ('err_roll', 'double', 'err_roll', 'double'), ('err_dip', 'double', 'err_dip', 'double'), ('position_covariance', 'array', 'position_covariance', 'string'), ('position_covariance_type', 'long', 'position_covariance_type', 'long')], transformation_ctx = 'applymapping58')
    resolvechoice59 = ResolveChoice.apply(frame = applymapping58, choice = 'make_cols', transformation_ctx = 'resolvechoice59')
    dropnullfields60 = DropNullFields.apply(frame = resolvechoice59, transformation_ctx = 'dropnullfields60')
    datasink61 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields60, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_gps', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink61')
except:
    print('Failed')

try:
    datasource58 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_limitation_debug_twist_lateral_accel_csv', transformation_ctx = 'datasource58')
    applymapping59 = ApplyMapping.apply(frame = datasource58, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping59')
    resolvechoice60 = ResolveChoice.apply(frame = applymapping59, choice = 'make_cols', transformation_ctx = 'resolvechoice60')
    dropnullfields61 = DropNullFields.apply(frame = resolvechoice60, transformation_ctx = 'dropnullfields61')
    datasink62 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields61, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_limitation_debug_twist_lateral_accel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink62')
except:
    print('Failed')

try:
    datasource59 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_inspvax_csv', transformation_ctx = 'datasource59')
    applymapping60 = ApplyMapping.apply(frame = datasource59, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('ins_status', 'string', 'ins_status', 'string'), ('position_type', 'string', 'position_type', 'string'), ('latitude', 'double', 'latitude', 'double'), ('longitude', 'double', 'longitude', 'double'), ('altitude', 'double', 'altitude', 'double'), ('undulation', 'double', 'undulation', 'double'), ('north_velocity', 'double', 'north_velocity', 'double'), ('east_velocity', 'string', 'east_velocity', 'string'), ('up_velocity', 'string', 'up_velocity', 'string'), ('roll', 'double', 'roll', 'double'), ('pitch', 'double', 'pitch', 'double'), ('azimuth', 'double', 'azimuth', 'double'), ('latitude_std', 'double', 'latitude_std', 'double'), ('longitude_std', 'double', 'longitude_std', 'double'), ('altitude_std', 'double', 'altitude_std', 'double'), ('north_velocity_std', 'double', 'north_velocity_std', 'double'), ('east_velocity_std', 'double', 'east_velocity_std', 'double'), ('up_velocity_std', 'double', 'up_velocity_std', 'double'), ('roll_std', 'double', 'roll_std', 'double'), ('pitch_std', 'double', 'pitch_std', 'double'), ('azimuth_std', 'double', 'azimuth_std', 'double'), ('extended_status', 'string', 'extended_status', 'string'), ('original_mask', 'long', 'original_mask', 'long'), ('advance_rtk_verified', 'boolean', 'advance_rtk_verified', 'boolean'), ('psuedorange_iono_correction', 'string', 'psuedorange_iono_correction', 'string'), ('seconds_since_update', 'long', 'seconds_since_update', 'long')], transformation_ctx = 'applymapping60')
    resolvechoice61 = ResolveChoice.apply(frame = applymapping60, choice = 'make_cols', transformation_ctx = 'resolvechoice61')
    dropnullfields62 = DropNullFields.apply(frame = resolvechoice61, transformation_ctx = 'dropnullfields62')
    datasink63 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields62, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_inspvax', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink63')
except:
    print('Failed')

try:
    datasource60 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_as_rx_turn_cmd_csv', transformation_ctx = 'datasource60')
    applymapping61 = ApplyMapping.apply(frame = datasource60, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enable', 'boolean', 'enable', 'boolean'), ('ignore_overrides', 'boolean', 'ignore_overrides', 'boolean'), ('clear_override', 'boolean', 'clear_override', 'boolean'), ('clear_faults', 'boolean', 'clear_faults', 'boolean'), ('command', 'long', 'command', 'long')], transformation_ctx = 'applymapping61')
    resolvechoice62 = ResolveChoice.apply(frame = applymapping61, choice = 'make_cols', transformation_ctx = 'resolvechoice62')
    dropnullfields63 = DropNullFields.apply(frame = resolvechoice62, transformation_ctx = 'dropnullfields63')
    datasink64 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields63, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_as_rx_turn_cmd', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink64')
except:
    print('Failed')

try:
    datasource61 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_result_twist_lateral_jerk_csv', transformation_ctx = 'datasource61')
    applymapping62 = ApplyMapping.apply(frame = datasource61, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping62')
    resolvechoice63 = ResolveChoice.apply(frame = applymapping62, choice = 'make_cols', transformation_ctx = 'resolvechoice63')
    dropnullfields64 = DropNullFields.apply(frame = resolvechoice63, transformation_ctx = 'dropnullfields64')
    datasink65 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields64, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_result_twist_lateral_jerk', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink65')
except:
    print('Failed')

try:
    datasource62 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_as_tx_vehicle_speed_csv', transformation_ctx = 'datasource62')
    applymapping63 = ApplyMapping.apply(frame = datasource62, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping63')
    resolvechoice64 = ResolveChoice.apply(frame = applymapping63, choice = 'make_cols', transformation_ctx = 'resolvechoice64')
    dropnullfields65 = DropNullFields.apply(frame = resolvechoice64, transformation_ctx = 'dropnullfields65')
    datasink66 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields65, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_as_tx_vehicle_speed', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink66')
except:
    print('Failed')

try:
    datasource63 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_turn_rpt_csv', transformation_ctx = 'datasource63')
    applymapping64 = ApplyMapping.apply(frame = datasource63, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enabled', 'boolean', 'enabled', 'boolean'), ('override_active', 'boolean', 'override_active', 'boolean'), ('command_output_fault', 'boolean', 'command_output_fault', 'boolean'), ('input_output_fault', 'boolean', 'input_output_fault', 'boolean'), ('output_reported_fault', 'boolean', 'output_reported_fault', 'boolean'), ('pacmod_fault', 'boolean', 'pacmod_fault', 'boolean'), ('vehicle_fault', 'boolean', 'vehicle_fault', 'boolean'), ('manual_input', 'long', 'manual_input', 'long'), ('command', 'long', 'command', 'long'), ('output', 'long', 'output', 'long')], transformation_ctx = 'applymapping64')
    resolvechoice65 = ResolveChoice.apply(frame = applymapping64, choice = 'make_cols', transformation_ctx = 'resolvechoice65')
    dropnullfields66 = DropNullFields.apply(frame = resolvechoice65, transformation_ctx = 'dropnullfields66')
    datasink67 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields66, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_turn_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink67')
except:
    print('Failed')

try:
    datasource64 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_as_rx_accel_cmd_csv', transformation_ctx = 'datasource64')
    applymapping65 = ApplyMapping.apply(frame = datasource64, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enable', 'boolean', 'enable', 'boolean'), ('ignore_overrides', 'boolean', 'ignore_overrides', 'boolean'), ('clear_override', 'boolean', 'clear_override', 'boolean'), ('clear_faults', 'boolean', 'clear_faults', 'boolean'), ('command', 'double', 'command', 'double')], transformation_ctx = 'applymapping65')
    resolvechoice66 = ResolveChoice.apply(frame = applymapping65, choice = 'make_cols', transformation_ctx = 'resolvechoice66')
    dropnullfields67 = DropNullFields.apply(frame = resolvechoice66, transformation_ctx = 'dropnullfields67')
    datasink68 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields67, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_as_rx_accel_cmd', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink68')
except:
    print('Failed')

try:
    datasource65 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_gprmc_csv', transformation_ctx = 'datasource65')
    applymapping66 = ApplyMapping.apply(frame = datasource65, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('message_id', 'string', 'message_id', 'string'), ('utc_seconds', 'double', 'utc_seconds', 'double'), ('position_status', 'string', 'position_status', 'string'), ('lat', 'double', 'lat', 'double'), ('lon', 'double', 'lon', 'double'), ('lat_dir', 'string', 'lat_dir', 'string'), ('lon_dir', 'string', 'lon_dir', 'string'), ('speed', 'double', 'speed', 'double'), ('track', 'double', 'track', 'double'), ('date', 'string', 'date', 'string'), ('mag_var', 'double', 'mag_var', 'double'), ('mag_var_direction', 'string', 'mag_var_direction', 'string'), ('mode_indicator', 'string', 'mode_indicator', 'string')], transformation_ctx = 'applymapping66')
    resolvechoice67 = ResolveChoice.apply(frame = applymapping66, choice = 'make_cols', transformation_ctx = 'resolvechoice67')
    dropnullfields68 = DropNullFields.apply(frame = resolvechoice67, transformation_ctx = 'dropnullfields68')
    datasink69 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields68, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_gprmc', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink69')
except:
    print('Failed')

try:
    datasource66 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_as_rx_brake_cmd_csv', transformation_ctx = 'datasource66')
    applymapping67 = ApplyMapping.apply(frame = datasource66, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enable', 'boolean', 'enable', 'boolean'), ('ignore_overrides', 'boolean', 'ignore_overrides', 'boolean'), ('clear_override', 'boolean', 'clear_override', 'boolean'), ('clear_faults', 'boolean', 'clear_faults', 'boolean'), ('command', 'double', 'command', 'double')], transformation_ctx = 'applymapping67')
    resolvechoice68 = ResolveChoice.apply(frame = applymapping67, choice = 'make_cols', transformation_ctx = 'resolvechoice68')
    dropnullfields69 = DropNullFields.apply(frame = resolvechoice68, transformation_ctx = 'dropnullfields69')
    datasink70 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields69, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_as_rx_brake_cmd', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink70')
except:
    print('Failed')

try:
    datasource67 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_throttle_command_echo_csv', transformation_ctx = 'datasource67')
    applymapping68 = ApplyMapping.apply(frame = datasource67, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('throttle_pedal', 'double', 'throttle_pedal', 'double')], transformation_ctx = 'applymapping68')
    resolvechoice69 = ResolveChoice.apply(frame = applymapping68, choice = 'make_cols', transformation_ctx = 'resolvechoice69')
    dropnullfields70 = DropNullFields.apply(frame = resolvechoice69, transformation_ctx = 'dropnullfields70')
    datasink71 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields70, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_throttle_command_echo', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink71')
except:
    print('Failed')

try:
    datasource68 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_arbitrated_speed_commands_csv', transformation_ctx = 'datasource68')
    applymapping69 = ApplyMapping.apply(frame = datasource68, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('mode', 'long', 'mode', 'long'), ('speed', 'double', 'speed', 'double'), ('acceleration_limit', 'double', 'acceleration_limit', 'double'), ('deceleration_limit', 'double', 'deceleration_limit', 'double')], transformation_ctx = 'applymapping69')
    resolvechoice70 = ResolveChoice.apply(frame = applymapping69, choice = 'make_cols', transformation_ctx = 'resolvechoice70')
    dropnullfields71 = DropNullFields.apply(frame = resolvechoice70, transformation_ctx = 'dropnullfields71')
    datasink72 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields71, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_arbitrated_speed_commands', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink72')
except:
    print('Failed')

try:
    datasource69 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_route_csv', transformation_ctx = 'datasource69')
    applymapping70 = ApplyMapping.apply(frame = datasource69, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('route_id', 'string', 'route_id', 'string'), ('route_version', 'long', 'route_version', 'long'), ('route_name', 'string', 'route_name', 'string'), ('shortest_path_lanelet_ids', 'array', 'shortest_path_lanelet_ids', 'string'), ('route_path_lanelet_ids', 'array', 'route_path_lanelet_ids', 'string'), ('end_point', 'string', 'end_point', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'double', 'z', 'double')], transformation_ctx = 'applymapping70')
    resolvechoice71 = ResolveChoice.apply(frame = applymapping70, choice = 'make_cols', transformation_ctx = 'resolvechoice71')
    dropnullfields72 = DropNullFields.apply(frame = resolvechoice71, transformation_ctx = 'dropnullfields72')
    datasink73 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields72, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_route', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink73')
except:
    print('Failed')

try:
    datasource70 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_brake_command_echo_csv', transformation_ctx = 'datasource70')
    applymapping71 = ApplyMapping.apply(frame = datasource70, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('brake_pedal', 'double', 'brake_pedal', 'double')], transformation_ctx = 'applymapping71')
    resolvechoice72 = ResolveChoice.apply(frame = applymapping71, choice = 'make_cols', transformation_ctx = 'resolvechoice72')
    dropnullfields73 = DropNullFields.apply(frame = resolvechoice72, transformation_ctx = 'dropnullfields73')
    datasink74 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields73, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_brake_command_echo', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink74')
except:
    print('Failed')

try:
    datasource71 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_vin_rpt_csv', transformation_ctx = 'datasource71')
    applymapping72 = ApplyMapping.apply(frame = datasource71, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('mfg_code', 'string', 'mfg_code', 'string'), ('mfg', 'string', 'mfg', 'string'), ('model_year_code', 'string', 'model_year_code', 'string'), ('model_year', 'long', 'model_year', 'long'), ('serial', 'long', 'serial', 'long')], transformation_ctx = 'applymapping72')
    resolvechoice73 = ResolveChoice.apply(frame = applymapping72, choice = 'make_cols', transformation_ctx = 'resolvechoice73')
    dropnullfields74 = DropNullFields.apply(frame = resolvechoice73, transformation_ctx = 'dropnullfields74')
    datasink75 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields74, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_vin_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink75')
except:
    print('Failed')

try:
    datasource72 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_shift_aux_rpt_csv', transformation_ctx = 'datasource72')
    applymapping73 = ApplyMapping.apply(frame = datasource72, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('between_gears', 'boolean', 'between_gears', 'boolean'), ('between_gears_is_valid', 'boolean', 'between_gears_is_valid', 'boolean'), ('stay_in_neutral_mode', 'boolean', 'stay_in_neutral_mode', 'boolean'), ('stay_in_neutral_mode_is_valid', 'boolean', 'stay_in_neutral_mode_is_valid', 'boolean'), ('brake_interlock_active', 'boolean', 'brake_interlock_active', 'boolean'), ('brake_interlock_active_is_valid', 'boolean', 'brake_interlock_active_is_valid', 'boolean'), ('speed_interlock_active', 'boolean', 'speed_interlock_active', 'boolean'), ('speed_interlock_active_is_valid', 'boolean', 'speed_interlock_active_is_valid', 'boolean'), ('gear_number_avail', 'boolean', 'gear_number_avail', 'boolean'), ('gear_number', 'long', 'gear_number', 'long')], transformation_ctx = 'applymapping73')
    resolvechoice74 = ResolveChoice.apply(frame = applymapping73, choice = 'make_cols', transformation_ctx = 'resolvechoice74')
    dropnullfields75 = DropNullFields.apply(frame = resolvechoice74, transformation_ctx = 'dropnullfields75')
    datasink76 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields75, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_shift_aux_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink76')
except:
    print('Failed')

try:
    datasource73 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_accel_rpt_csv', transformation_ctx = 'datasource73')
    applymapping74 = ApplyMapping.apply(frame = datasource73, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enabled', 'boolean', 'enabled', 'boolean'), ('override_active', 'boolean', 'override_active', 'boolean'), ('command_output_fault', 'boolean', 'command_output_fault', 'boolean'), ('input_output_fault', 'boolean', 'input_output_fault', 'boolean'), ('output_reported_fault', 'boolean', 'output_reported_fault', 'boolean'), ('pacmod_fault', 'boolean', 'pacmod_fault', 'boolean'), ('vehicle_fault', 'boolean', 'vehicle_fault', 'boolean'), ('manual_input', 'double', 'manual_input', 'double'), ('command', 'double', 'command', 'double'), ('output', 'double', 'output', 'double')], transformation_ctx = 'applymapping74')
    resolvechoice75 = ResolveChoice.apply(frame = applymapping74, choice = 'make_cols', transformation_ctx = 'resolvechoice75')
    dropnullfields76 = DropNullFields.apply(frame = resolvechoice75, transformation_ctx = 'dropnullfields76')
    datasink77 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields76, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_accel_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink77')
except:
    print('Failed')

try:
    datasource74 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_vehicle_engage_csv', transformation_ctx = 'datasource74')
    applymapping75 = ApplyMapping.apply(frame = datasource74, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'boolean', 'data', 'boolean')], transformation_ctx = 'applymapping75')
    resolvechoice76 = ResolveChoice.apply(frame = applymapping75, choice = 'make_cols', transformation_ctx = 'resolvechoice76')
    dropnullfields77 = DropNullFields.apply(frame = resolvechoice76, transformation_ctx = 'dropnullfields77')
    datasink78 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields77, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_vehicle_engage', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink78')
except:
    print('Failed')

try:
    datasource75 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_vehicle_status_csv', transformation_ctx = 'datasource75')
    applymapping76 = ApplyMapping.apply(frame = datasource75, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('tm', 'string', 'tm', 'string'), ('drivemode', 'long', 'drivemode', 'long'), ('steeringmode', 'long', 'steeringmode', 'long'), ('gearshift', 'long', 'gearshift', 'long'), ('speed', 'double', 'speed', 'double'), ('drivepedal', 'long', 'drivepedal', 'long'), ('brakepedal', 'long', 'brakepedal', 'long'), ('angle', 'string', 'angle', 'string'), ('lamp', 'long', 'lamp', 'long'), ('light', 'long', 'light', 'long')], transformation_ctx = 'applymapping76')
    resolvechoice77 = ResolveChoice.apply(frame = applymapping76, choice = 'make_cols', transformation_ctx = 'resolvechoice77')
    dropnullfields78 = DropNullFields.apply(frame = resolvechoice77, transformation_ctx = 'dropnullfields78')
    datasink79 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields78, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_vehicle_status', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink79')
except:
    print('Failed')

try:
    datasource76 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='localization_ekf_localizer_debug_measured_pose_csv', transformation_ctx = 'datasource76')
    applymapping77 = ApplyMapping.apply(frame = datasource76, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('pose', 'string', 'pose', 'string'), ('position', 'string', 'position', 'string'), ('x', 'string', 'x', 'string'), ('y', 'string', 'y', 'string'), ('z', 'double', 'z', 'double'), ('orientation', 'string', 'orientation', 'string'), ('w', 'double', 'w', 'double')], transformation_ctx = 'applymapping77')
    resolvechoice78 = ResolveChoice.apply(frame = applymapping77, choice = 'make_cols', transformation_ctx = 'resolvechoice78')
    dropnullfields79 = DropNullFields.apply(frame = resolvechoice78, transformation_ctx = 'dropnullfields79')
    datasink80 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields79, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.localization_ekf_localizer_debug_measured_pose', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink80')
except:
    print('Failed')

try:
    datasource77 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_dbw_enabled_feedback_csv', transformation_ctx = 'datasource77')
    applymapping78 = ApplyMapping.apply(frame = datasource77, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'boolean', 'data', 'boolean')], transformation_ctx = 'applymapping78')
    resolvechoice79 = ResolveChoice.apply(frame = applymapping78, choice = 'make_cols', transformation_ctx = 'resolvechoice79')
    dropnullfields80 = DropNullFields.apply(frame = resolvechoice79, transformation_ctx = 'dropnullfields80')
    datasink81 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields80, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_dbw_enabled_feedback', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink81')
except:
    print('Failed')

try:
    datasource78 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_raw_csv', transformation_ctx = 'datasource78')
    applymapping79 = ApplyMapping.apply(frame = datasource78, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('twist', 'string', 'twist', 'string'), ('linear', 'string', 'linear', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'string', 'z', 'string'), ('angular', 'string', 'angular', 'string')], transformation_ctx = 'applymapping79')
    resolvechoice80 = ResolveChoice.apply(frame = applymapping79, choice = 'make_cols', transformation_ctx = 'resolvechoice80')
    dropnullfields81 = DropNullFields.apply(frame = resolvechoice80, transformation_ctx = 'dropnullfields81')
    datasink82 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields81, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_raw', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink82')
except:
    print('Failed')

try:
    datasource79 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_plugins_stopandwaitplugin_jerk_val_csv', transformation_ctx = 'datasource79')
    applymapping80 = ApplyMapping.apply(frame = datasource79, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping80')
    resolvechoice81 = ResolveChoice.apply(frame = applymapping80, choice = 'make_cols', transformation_ctx = 'resolvechoice81')
    dropnullfields82 = DropNullFields.apply(frame = resolvechoice81, transformation_ctx = 'dropnullfields82')
    datasink83 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields82, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_plugins_stopandwaitplugin_jerk_val', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink83')
except:
    print('Failed')

try:
    datasource80 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_vehicle_cmd_csv', transformation_ctx = 'datasource80')
    applymapping81 = ApplyMapping.apply(frame = datasource80, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('steer_cmd', 'string', 'steer_cmd', 'string'), ('steer', 'long', 'steer', 'long'), ('accel_cmd', 'string', 'accel_cmd', 'string'), ('accel', 'long', 'accel', 'long'), ('brake_cmd', 'string', 'brake_cmd', 'string'), ('brake', 'long', 'brake', 'long'), ('lamp_cmd', 'string', 'lamp_cmd', 'string'), ('l', 'long', 'l', 'long'), ('r', 'long', 'r', 'long'), ('gear', 'long', 'gear', 'long'), ('mode', 'long', 'mode', 'long'), ('twist_cmd', 'string', 'twist_cmd', 'string'), ('twist', 'string', 'twist', 'string'), ('linear', 'string', 'linear', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'double', 'z', 'double'), ('angular', 'string', 'angular', 'string'), ('ctrl_cmd', 'string', 'ctrl_cmd', 'string'), ('linear_velocity', 'double', 'linear_velocity', 'double'), ('linear_acceleration', 'double', 'linear_acceleration', 'double'), ('steering_angle', 'double', 'steering_angle', 'double'), ('emergency', 'long', 'emergency', 'long')], transformation_ctx = 'applymapping81')
    resolvechoice82 = ResolveChoice.apply(frame = applymapping81, choice = 'make_cols', transformation_ctx = 'resolvechoice82')
    dropnullfields83 = DropNullFields.apply(frame = resolvechoice82, transformation_ctx = 'dropnullfields83')
    datasink84 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields83, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_vehicle_cmd', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink84')
except:
    print('Failed')

try:
    datasource81 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_lightbar_light_bar_status_csv', transformation_ctx = 'datasource81')
    applymapping82 = ApplyMapping.apply(frame = datasource81, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('green_solid', 'long', 'green_solid', 'long'), ('yellow_solid', 'long', 'yellow_solid', 'long'), ('right_arrow', 'long', 'right_arrow', 'long'), ('left_arrow', 'long', 'left_arrow', 'long'), ('sides_solid', 'long', 'sides_solid', 'long'), ('flash', 'long', 'flash', 'long'), ('green_flash', 'long', 'green_flash', 'long'), ('takedown', 'long', 'takedown', 'long')], transformation_ctx = 'applymapping82')
    resolvechoice83 = ResolveChoice.apply(frame = applymapping82, choice = 'make_cols', transformation_ctx = 'resolvechoice83')
    dropnullfields84 = DropNullFields.apply(frame = resolvechoice83, transformation_ctx = 'dropnullfields84')
    datasink85 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields84, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_lightbar_light_bar_status', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink85')
except:
    print('Failed')

try:
    datasource82 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_shift_rpt_csv', transformation_ctx = 'datasource82')
    applymapping83 = ApplyMapping.apply(frame = datasource82, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enabled', 'boolean', 'enabled', 'boolean'), ('override_active', 'boolean', 'override_active', 'boolean'), ('command_output_fault', 'boolean', 'command_output_fault', 'boolean'), ('input_output_fault', 'boolean', 'input_output_fault', 'boolean'), ('output_reported_fault', 'boolean', 'output_reported_fault', 'boolean'), ('pacmod_fault', 'boolean', 'pacmod_fault', 'boolean'), ('vehicle_fault', 'boolean', 'vehicle_fault', 'boolean'), ('manual_input', 'long', 'manual_input', 'long'), ('command', 'long', 'command', 'long'), ('output', 'long', 'output', 'long')], transformation_ctx = 'applymapping83')
    resolvechoice84 = ResolveChoice.apply(frame = applymapping83, choice = 'make_cols', transformation_ctx = 'resolvechoice84')
    dropnullfields85 = DropNullFields.apply(frame = resolvechoice84, transformation_ctx = 'dropnullfields85')
    datasink86 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields85, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_shift_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink86')
except:
    print('Failed')

try:
    datasource83 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_cmd_csv', transformation_ctx = 'datasource83')
    applymapping84 = ApplyMapping.apply(frame = datasource83, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('twist', 'string', 'twist', 'string'), ('linear', 'string', 'linear', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'string', 'z', 'string'), ('angular', 'string', 'angular', 'string')], transformation_ctx = 'applymapping84')
    resolvechoice85 = ResolveChoice.apply(frame = applymapping84, choice = 'make_cols', transformation_ctx = 'resolvechoice85')
    dropnullfields86 = DropNullFields.apply(frame = resolvechoice85, transformation_ctx = 'dropnullfields86')
    datasink87 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields86, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_cmd', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink87')
except:
    print('Failed')

try:
    datasource84 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_limitation_debug_ctrl_lateral_jerk_csv', transformation_ctx = 'datasource84')
    applymapping85 = ApplyMapping.apply(frame = datasource84, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping85')
    resolvechoice86 = ResolveChoice.apply(frame = applymapping85, choice = 'make_cols', transformation_ctx = 'resolvechoice86')
    dropnullfields87 = DropNullFields.apply(frame = resolvechoice86, transformation_ctx = 'dropnullfields87')
    datasink88 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields87, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_limitation_debug_ctrl_lateral_jerk', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink88')
except:
    print('Failed')

try:
    datasource85 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_limitation_debug_ctrl_lateral_accel_csv', transformation_ctx = 'datasource85')
    applymapping86 = ApplyMapping.apply(frame = datasource85, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping86')
    resolvechoice87 = ResolveChoice.apply(frame = applymapping86, choice = 'make_cols', transformation_ctx = 'resolvechoice87')
    dropnullfields88 = DropNullFields.apply(frame = resolvechoice87, transformation_ctx = 'dropnullfields88')
    datasink89 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields88, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_limitation_debug_ctrl_lateral_accel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink89')
except:
    print('Failed')

try:
    datasource86 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_bestvel_csv', transformation_ctx = 'datasource86')
    applymapping87 = ApplyMapping.apply(frame = datasource86, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('solution_status', 'string', 'solution_status', 'string'), ('velocity_type', 'string', 'velocity_type', 'string'), ('latency', 'double', 'latency', 'double'), ('age', 'double', 'age', 'double'), ('horizontal_speed', 'string', 'horizontal_speed', 'string'), ('track_ground', 'double', 'track_ground', 'double'), ('vertical_speed', 'string', 'vertical_speed', 'string')], transformation_ctx = 'applymapping87')
    resolvechoice88 = ResolveChoice.apply(frame = applymapping87, choice = 'make_cols', transformation_ctx = 'resolvechoice88')
    dropnullfields89 = DropNullFields.apply(frame = resolvechoice88, transformation_ctx = 'dropnullfields89')
    datasink90 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields89, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_bestvel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink90')
except:
    print('Failed')

try:
    datasource87 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_vehicle_speed_rpt_csv', transformation_ctx = 'datasource87')
    applymapping88 = ApplyMapping.apply(frame = datasource87, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('vehicle_speed', 'double', 'vehicle_speed', 'double'), ('vehicle_speed_valid', 'boolean', 'vehicle_speed_valid', 'boolean'), ('vehicle_speed_raw', 'array', 'vehicle_speed_raw', 'string')], transformation_ctx = 'applymapping88')
    resolvechoice89 = ResolveChoice.apply(frame = applymapping88, choice = 'make_cols', transformation_ctx = 'resolvechoice89')
    dropnullfields90 = DropNullFields.apply(frame = resolvechoice89, transformation_ctx = 'dropnullfields90')
    datasink91 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields90, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_vehicle_speed_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink91')
except:
    print('Failed')

try:
    datasource88 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_comms_outbound_binary_msg_csv', transformation_ctx = 'datasource88')
    applymapping89 = ApplyMapping.apply(frame = datasource88, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('messagetype', 'string', 'messagetype', 'string'), ('content', 'array', 'content', 'string')], transformation_ctx = 'applymapping89')
    resolvechoice90 = ResolveChoice.apply(frame = applymapping89, choice = 'make_cols', transformation_ctx = 'resolvechoice90')
    dropnullfields91 = DropNullFields.apply(frame = resolvechoice90, transformation_ctx = 'dropnullfields91')
    datasink92 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields91, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_comms_outbound_binary_msg', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink92')
except:
    print('Failed')

try:
    datasource89 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_veh_controller_config_csv', transformation_ctx = 'datasource89')
    applymapping90 = ApplyMapping.apply(frame = datasource89, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string'), ('ignore_override_time', 'string', 'ignore_override_time', 'string')], transformation_ctx = 'applymapping90')
    resolvechoice91 = ResolveChoice.apply(frame = applymapping90, choice = 'make_cols', transformation_ctx = 'resolvechoice91')
    dropnullfields92 = DropNullFields.apply(frame = resolvechoice91, transformation_ctx = 'dropnullfields92')
    datasink93 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields92, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_veh_controller_config', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink93')
except:
    print('Failed')

try:
    datasource90 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_result_twist_lateral_accel_csv', transformation_ctx = 'datasource90')
    applymapping91 = ApplyMapping.apply(frame = datasource90, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping91')
    resolvechoice92 = ResolveChoice.apply(frame = applymapping91, choice = 'make_cols', transformation_ctx = 'resolvechoice92')
    dropnullfields93 = DropNullFields.apply(frame = resolvechoice92, transformation_ctx = 'dropnullfields93')
    datasink94 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields93, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_result_twist_lateral_accel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink94')
except:
    print('Failed')

try:
    datasource91 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_lightbar_manager_indicator_control_csv', transformation_ctx = 'datasource91')
    applymapping92 = ApplyMapping.apply(frame = datasource91, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('green_solid_owner', 'string', 'green_solid_owner', 'string'), ('green_flash_owner', 'string', 'green_flash_owner', 'string'), ('yellow_sides_owner', 'string', 'yellow_sides_owner', 'string'), ('yellow_dim_owner', 'string', 'yellow_dim_owner', 'string'), ('yellow_flash_owner', 'string', 'yellow_flash_owner', 'string'), ('yellow_arrow_left_owner', 'string', 'yellow_arrow_left_owner', 'string'), ('yellow_arrow_right_owner', 'string', 'yellow_arrow_right_owner', 'string'), ('yellow_arrow_out_owner', 'string', 'yellow_arrow_out_owner', 'string')], transformation_ctx = 'applymapping92')
    resolvechoice93 = ResolveChoice.apply(frame = applymapping92, choice = 'make_cols', transformation_ctx = 'resolvechoice93')
    dropnullfields94 = DropNullFields.apply(frame = resolvechoice93, transformation_ctx = 'dropnullfields94')
    datasink95 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields94, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_lightbar_manager_indicator_control', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink95')
except:
    print('Failed')

try:
    datasource92 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_turn_signal_command_csv', transformation_ctx = 'datasource92')
    applymapping93 = ApplyMapping.apply(frame = datasource92, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('mode', 'long', 'mode', 'long'), ('turn_signal', 'long', 'turn_signal', 'long')], transformation_ctx = 'applymapping93')
    resolvechoice94 = ResolveChoice.apply(frame = applymapping93, choice = 'make_cols', transformation_ctx = 'resolvechoice94')
    dropnullfields95 = DropNullFields.apply(frame = resolvechoice94, transformation_ctx = 'dropnullfields95')
    datasink96 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields95, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_turn_signal_command_csvhardware_interface_turn_signal_command', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink96')
except:
    print('Failed')

try:
    datasource93 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_brake_rpt_csv', transformation_ctx = 'datasource93')
    applymapping94 = ApplyMapping.apply(frame = datasource93, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('enabled', 'boolean', 'enabled', 'boolean'), ('override_active', 'boolean', 'override_active', 'boolean'), ('command_output_fault', 'boolean', 'command_output_fault', 'boolean'), ('input_output_fault', 'boolean', 'input_output_fault', 'boolean'), ('output_reported_fault', 'boolean', 'output_reported_fault', 'boolean'), ('pacmod_fault', 'boolean', 'pacmod_fault', 'boolean'), ('vehicle_fault', 'boolean', 'vehicle_fault', 'boolean'), ('manual_input', 'double', 'manual_input', 'double'), ('command', 'double', 'command', 'double'), ('output', 'double', 'output', 'double')], transformation_ctx = 'applymapping94')
    resolvechoice95 = ResolveChoice.apply(frame = applymapping94, choice = 'make_cols', transformation_ctx = 'resolvechoice95')
    dropnullfields96 = DropNullFields.apply(frame = resolvechoice95, transformation_ctx = 'dropnullfields96')
    datasink97 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields96, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_brake_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink97')
except:
    print('Failed')

try:
    datasource94 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_imu_raw_csv', transformation_ctx = 'datasource94')
    applymapping95 = ApplyMapping.apply(frame = datasource94, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('orientation', 'string', 'orientation', 'string'), ('x', 'string', 'x', 'string'), ('y', 'string', 'y', 'string'), ('z', 'string', 'z', 'string'), ('w', 'double', 'w', 'double'), ('orientation_covariance', 'array', 'orientation_covariance', 'string'), ('angular_velocity', 'string', 'angular_velocity', 'string'), ('angular_velocity_covariance', 'array', 'angular_velocity_covariance', 'string'), ('linear_acceleration', 'string', 'linear_acceleration', 'string'), ('linear_acceleration_covariance', 'array', 'linear_acceleration_covariance', 'string')], transformation_ctx = 'applymapping95')
    resolvechoice96 = ResolveChoice.apply(frame = applymapping95, choice = 'make_cols', transformation_ctx = 'resolvechoice96')
    dropnullfields97 = DropNullFields.apply(frame = resolvechoice96, transformation_ctx = 'dropnullfields97')
    datasink98 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields97, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_imu_raw', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink98')
except:
    print('Failed')

try:
    datasource95 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_driver_discovery_csv', transformation_ctx = 'datasource95')
    applymapping96 = ApplyMapping.apply(frame = datasource95, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('name', 'string', 'name', 'string'), ('status', 'long', 'status', 'long'), ('can', 'boolean', 'can', 'boolean'), ('radar', 'boolean', 'radar', 'boolean'), ('gnss', 'boolean', 'gnss', 'boolean'), ('lidar', 'boolean', 'lidar', 'boolean'), ('roadway_sensor', 'boolean', 'roadway_sensor', 'boolean'), ('comms', 'boolean', 'comms', 'boolean'), ('controller', 'boolean', 'controller', 'boolean'), ('camera', 'boolean', 'camera', 'boolean'), ('imu', 'boolean', 'imu', 'boolean'), ('trailer_angle_sensor', 'boolean', 'trailer_angle_sensor', 'boolean'), ('lightbar', 'boolean', 'lightbar', 'boolean')], transformation_ctx = 'applymapping96')
    resolvechoice97 = ResolveChoice.apply(frame = applymapping96, choice = 'make_cols', transformation_ctx = 'resolvechoice97')
    dropnullfields98 = DropNullFields.apply(frame = resolvechoice97, transformation_ctx = 'dropnullfields98')
    datasink99 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields98, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_driver_discovery', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink99')
except:
    print('Failed')

try:
    datasource96 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_parsed_tx_steer_aux_rpt_csv', transformation_ctx = 'datasource96')
    applymapping97 = ApplyMapping.apply(frame = datasource96, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('raw_position', 'double', 'raw_position', 'double'), ('raw_position_is_valid', 'boolean', 'raw_position_is_valid', 'boolean'), ('raw_torque', 'double', 'raw_torque', 'double'), ('raw_torque_is_valid', 'boolean', 'raw_torque_is_valid', 'boolean'), ('rotation_rate', 'double', 'rotation_rate', 'double'), ('rotation_rate_is_valid', 'boolean', 'rotation_rate_is_valid', 'boolean'), ('user_interaction', 'boolean', 'user_interaction', 'boolean'), ('user_interaction_is_valid', 'boolean', 'user_interaction_is_valid', 'boolean')], transformation_ctx = 'applymapping97')
    resolvechoice98 = ResolveChoice.apply(frame = applymapping97, choice = 'make_cols', transformation_ctx = 'resolvechoice98')
    dropnullfields99 = DropNullFields.apply(frame = resolvechoice98, transformation_ctx = 'dropnullfields99')
    datasink100 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields99, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_parsed_tx_steer_aux_rpt', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink100')
except:
    print('Failed')

try:
    datasource97 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_pacmod_can_tx_csv', transformation_ctx = 'datasource97')
    applymapping98 = ApplyMapping.apply(frame = datasource97, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('id', 'long', 'id', 'long'), ('is_rtr', 'boolean', 'is_rtr', 'boolean'), ('is_extended', 'boolean', 'is_extended', 'boolean'), ('is_error', 'boolean', 'is_error', 'boolean'), ('dlc', 'long', 'dlc', 'long'), ('data', 'array', 'data', 'string')], transformation_ctx = 'applymapping98')
    resolvechoice99 = ResolveChoice.apply(frame = applymapping98, choice = 'make_cols', transformation_ctx = 'resolvechoice99')
    dropnullfields100 = DropNullFields.apply(frame = resolvechoice99, transformation_ctx = 'dropnullfields100')
    datasink101 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields100, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_pacmod_can_tx', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink101')
except:
    print('Failed')

try:
    datasource98 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_dual_antenna_heading_csv', transformation_ctx = 'datasource98')
    applymapping99 = ApplyMapping.apply(frame = datasource98, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('solution_status', 'string', 'solution_status', 'string'), ('position_type', 'string', 'position_type', 'string'), ('baseline_length', 'double', 'baseline_length', 'double'), ('heading', 'double', 'heading', 'double'), ('pitch', 'double', 'pitch', 'double'), ('heading_sigma', 'double', 'heading_sigma', 'double'), ('pitch_sigma', 'double', 'pitch_sigma', 'double'), ('station_id', 'string', 'station_id', 'string'), ('num_satellites_tracked', 'long', 'num_satellites_tracked', 'long'), ('num_satellites_used_in_solution', 'long', 'num_satellites_used_in_solution', 'long'), ('num_satellites_above_elevation_mask_angle', 'long', 'num_satellites_above_elevation_mask_angle', 'long'), ('num_satellites_above_elevation_mask_angle_l2', 'long', 'num_satellites_above_elevation_mask_angle_l2', 'long'), ('solution_source', 'long', 'solution_source', 'long'), ('extended_solution_status', 'string', 'extended_solution_status', 'string'), ('original_mask', 'long', 'original_mask', 'long'), ('advance_rtk_verified', 'boolean', 'advance_rtk_verified', 'boolean'), ('psuedorange_iono_correction', 'string', 'psuedorange_iono_correction', 'string'), ('signal_mask', 'string', 'signal_mask', 'string'), ('gps_l1_used_in_solution', 'boolean', 'gps_l1_used_in_solution', 'boolean'), ('gps_l2_used_in_solution', 'boolean', 'gps_l2_used_in_solution', 'boolean'), ('gps_l3_used_in_solution', 'boolean', 'gps_l3_used_in_solution', 'boolean'), ('glonass_l1_used_in_solution', 'boolean', 'glonass_l1_used_in_solution', 'boolean'), ('glonass_l2_used_in_solution', 'boolean', 'glonass_l2_used_in_solution', 'boolean')], transformation_ctx = 'applymapping99')
    resolvechoice100 = ResolveChoice.apply(frame = applymapping99, choice = 'make_cols', transformation_ctx = 'resolvechoice100')
    dropnullfields101 = DropNullFields.apply(frame = resolvechoice100, transformation_ctx = 'dropnullfields101')
    datasink102 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields101, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_dual_antenna_heading', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink102')
except:
    print('Failed')

try:
    datasource99 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_limitation_debug_twist_lateral_jerk_csv', transformation_ctx = 'datasource99')
    applymapping100 = ApplyMapping.apply(frame = datasource99, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping100')
    resolvechoice101 = ResolveChoice.apply(frame = applymapping100, choice = 'make_cols', transformation_ctx = 'resolvechoice101')
    dropnullfields102 = DropNullFields.apply(frame = resolvechoice101, transformation_ctx = 'dropnullfields102')
    datasink103 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields102, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_limitation_debug_twist_lateral_jerk', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink103')
except:
    print('Failed')

try:
    datasource100 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_result_ctrl_lateral_accel_csv', transformation_ctx = 'datasource100')
    applymapping101 = ApplyMapping.apply(frame = datasource100, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping101')
    resolvechoice102 = ResolveChoice.apply(frame = applymapping101, choice = 'make_cols', transformation_ctx = 'resolvechoice102')
    dropnullfields103 = DropNullFields.apply(frame = resolvechoice102, transformation_ctx = 'dropnullfields103')
    datasink104 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields103, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_result_ctrl_lateral_accel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink104')
except:
    print('Failed')

try:
    datasource101 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_bestpos_csv', transformation_ctx = 'datasource101')
    applymapping102 = ApplyMapping.apply(frame = datasource101, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('solution_status', 'string', 'solution_status', 'string'), ('position_type', 'string', 'position_type', 'string'), ('lat', 'double', 'lat', 'double'), ('lon', 'double', 'lon', 'double'), ('height', 'double', 'height', 'double'), ('undulation', 'double', 'undulation', 'double'), ('datum_id', 'string', 'datum_id', 'string'), ('lat_sigma', 'double', 'lat_sigma', 'double'), ('lon_sigma', 'double', 'lon_sigma', 'double'), ('height_sigma', 'double', 'height_sigma', 'double'), ('base_station_id', 'string', 'base_station_id', 'string'), ('diff_age', 'double', 'diff_age', 'double'), ('solution_age', 'double', 'solution_age', 'double'), ('num_satellites_tracked', 'long', 'num_satellites_tracked', 'long'), ('num_satellites_used_in_solution', 'long', 'num_satellites_used_in_solution', 'long'), ('num_gps_and_glonass_l1_used_in_solution', 'long', 'num_gps_and_glonass_l1_used_in_solution', 'long'), ('num_gps_and_glonass_l1_and_l2_used_in_solution', 'long', 'num_gps_and_glonass_l1_and_l2_used_in_solution', 'long'), ('extended_solution_status', 'string', 'extended_solution_status', 'string'), ('original_mask', 'long', 'original_mask', 'long'), ('advance_rtk_verified', 'boolean', 'advance_rtk_verified', 'boolean'), ('psuedorange_iono_correction', 'string', 'psuedorange_iono_correction', 'string'), ('signal_mask', 'string', 'signal_mask', 'string'), ('gps_l1_used_in_solution', 'boolean', 'gps_l1_used_in_solution', 'boolean'), ('gps_l2_used_in_solution', 'boolean', 'gps_l2_used_in_solution', 'boolean'), ('gps_l3_used_in_solution', 'boolean', 'gps_l3_used_in_solution', 'boolean'), ('glonass_l1_used_in_solution', 'boolean', 'glonass_l1_used_in_solution', 'boolean'), ('glonass_l2_used_in_solution', 'boolean', 'glonass_l2_used_in_solution', 'boolean')], transformation_ctx = 'applymapping102')
    resolvechoice103 = ResolveChoice.apply(frame = applymapping102, choice = 'make_cols', transformation_ctx = 'resolvechoice103')
    dropnullfields104 = DropNullFields.apply(frame = resolvechoice103, transformation_ctx = 'dropnullfields104')
    datasink105 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields104, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_bestpos', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink105')
except:
    print('Failed')

try:
    datasource102 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_can_transmission_state_csv', transformation_ctx = 'datasource102')
    applymapping103 = ApplyMapping.apply(frame = datasource102, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('transmission_state', 'long', 'transmission_state', 'long')], transformation_ctx = 'applymapping103')
    resolvechoice104 = ResolveChoice.apply(frame = applymapping103, choice = 'make_cols', transformation_ctx = 'resolvechoice104')
    dropnullfields105 = DropNullFields.apply(frame = resolvechoice104, transformation_ctx = 'dropnullfields105')
    datasink106 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields105, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_can_transmission_state', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink106')
except:
    print('Failed')

try:
    datasource103 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_route_state_csv', transformation_ctx = 'datasource103')
    applymapping104 = ApplyMapping.apply(frame = datasource103, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('routeid', 'string', 'routeid', 'string'), ('state', 'long', 'state', 'long'), ('cross_track', 'double', 'cross_track', 'double'), ('down_track', 'double', 'down_track', 'double'), ('lanelet_downtrack', 'string', 'lanelet_downtrack', 'string'), ('lanelet_id', 'long', 'lanelet_id', 'long'), ('speed_limit', 'double', 'speed_limit', 'double')], transformation_ctx = 'applymapping104')
    resolvechoice105 = ResolveChoice.apply(frame = applymapping104, choice = 'make_cols', transformation_ctx = 'resolvechoice105')
    dropnullfields106 = DropNullFields.apply(frame = resolvechoice105, transformation_ctx = 'dropnullfields106')
    datasink107 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields106, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_route_state', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink107')
except:
    print('Failed')

try:
    datasource104 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_comms_inbound_binary_msg_csv', transformation_ctx = 'datasource104')
    applymapping105 = ApplyMapping.apply(frame = datasource104, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('messagetype', 'string', 'messagetype', 'string'), ('content', 'array', 'content', 'string')], transformation_ctx = 'applymapping105')
    resolvechoice106 = ResolveChoice.apply(frame = applymapping105, choice = 'make_cols', transformation_ctx = 'resolvechoice106')
    dropnullfields107 = DropNullFields.apply(frame = resolvechoice106, transformation_ctx = 'dropnullfields107')
    datasink108 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields107, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_comms_inbound_binary_msg', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink108')
except:
    print('Failed')

try:
    datasource105 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_search_circle_mark_csv', transformation_ctx = 'datasource105')
    applymapping106 = ApplyMapping.apply(frame = datasource105, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('ns', 'string', 'ns', 'string'), ('id', 'long', 'id', 'long'), ('type', 'long', 'type', 'long'), ('action', 'long', 'action', 'long'), ('pose', 'string', 'pose', 'string'), ('position', 'string', 'position', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'double', 'z', 'double'), ('orientation', 'string', 'orientation', 'string'), ('w', 'double', 'w', 'double'), ('scale', 'string', 'scale', 'string'), ('color', 'string', 'color', 'string'), ('r', 'double', 'r', 'double'), ('g', 'double', 'g', 'double'), ('b', 'double', 'b', 'double'), ('a', 'double', 'a', 'double'), ('lifetime', 'string', 'lifetime', 'string'), ('frame_locked', 'boolean', 'frame_locked', 'boolean'), ('points', 'array', 'points', 'string'), ('colors', 'array', 'colors', 'string'), ('text', 'string', 'text', 'string'), ('mesh_resource', 'string', 'mesh_resource', 'string'), ('mesh_use_embedded_materials', 'boolean', 'mesh_use_embedded_materials', 'boolean')], transformation_ctx = 'applymapping106')
    resolvechoice107 = ResolveChoice.apply(frame = applymapping106, choice = 'make_cols', transformation_ctx = 'resolvechoice107')
    dropnullfields108 = DropNullFields.apply(frame = resolvechoice107, transformation_ctx = 'dropnullfields108')
    datasink109 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields108, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_search_circle_mark', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink109')
except:
    print('Failed')

try:
    datasource106 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_result_ctrl_lateral_jerk_csv', transformation_ctx = 'datasource106')
    applymapping107 = ApplyMapping.apply(frame = datasource106, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping107')
    resolvechoice108 = ResolveChoice.apply(frame = applymapping107, choice = 'make_cols', transformation_ctx = 'resolvechoice108')
    dropnullfields109 = DropNullFields.apply(frame = resolvechoice108, transformation_ctx = 'dropnullfields109')
    datasink110 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields109, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_result_ctrl_lateral_jerk', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink110')
except:
    print('Failed')

try:
    datasource107 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='environment_node_status_csv', transformation_ctx = 'datasource107')
    applymapping108 = ApplyMapping.apply(frame = datasource107, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('node_name', 'string', 'node_name', 'string'), ('node_activated', 'boolean', 'node_activated', 'boolean'), ('status', 'array', 'status', 'string')], transformation_ctx = 'applymapping108')
    resolvechoice109 = ResolveChoice.apply(frame = applymapping108, choice = 'make_cols', transformation_ctx = 'resolvechoice109')
    dropnullfields110 = DropNullFields.apply(frame = resolvechoice109, transformation_ctx = 'dropnullfields110')
    datasink111 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields110, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.environment_node_status', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink111')
except:
    print('Failed')

try:
    datasource108 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_limitation_debug_twist_lateral_accel_csv', transformation_ctx = 'datasource108')
    applymapping109 = ApplyMapping.apply(frame = datasource108, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping109')
    resolvechoice110 = ResolveChoice.apply(frame = applymapping109, choice = 'make_cols', transformation_ctx = 'resolvechoice110')
    dropnullfields111 = DropNullFields.apply(frame = resolvechoice110, transformation_ctx = 'dropnullfields111')
    datasink112 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields111, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_limitation_debug_twist_lateral_accel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink112')
except:
    print('Failed')

try:
    datasource109 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_node_status_csv', transformation_ctx = 'datasource109')
    applymapping110 = ApplyMapping.apply(frame = datasource109, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('node_name', 'string', 'node_name', 'string'), ('node_activated', 'boolean', 'node_activated', 'boolean'), ('status', 'array', 'status', 'string')], transformation_ctx = 'applymapping110')
    resolvechoice111 = ResolveChoice.apply(frame = applymapping110, choice = 'make_cols', transformation_ctx = 'resolvechoice111')
    dropnullfields112 = DropNullFields.apply(frame = resolvechoice111, transformation_ctx = 'dropnullfields112')
    datasink113 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields112, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_node_status', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink113')
except:
    print('Failed')

try:
    datasource110 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_result_twist_lateral_jerk_csv', transformation_ctx = 'datasource110')
    applymapping111 = ApplyMapping.apply(frame = datasource110, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping111')
    resolvechoice112 = ResolveChoice.apply(frame = applymapping111, choice = 'make_cols', transformation_ctx = 'resolvechoice112')
    dropnullfields113 = DropNullFields.apply(frame = resolvechoice112, transformation_ctx = 'dropnullfields113')
    datasink114 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields113, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_result_twist_lateral_jerk', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink114')
except:
    print('Failed')

try:
    datasource111 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='environment_active_geofence_csv', transformation_ctx = 'datasource111')
    applymapping112 = ApplyMapping.apply(frame = datasource111, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('is_on_active_geofence', 'boolean', 'is_on_active_geofence', 'boolean'), ('type', 'long', 'type', 'long'), ('value', 'double', 'value', 'double'), ('distance_to_next_geofence', 'double', 'distance_to_next_geofence', 'double')], transformation_ctx = 'applymapping112')
    resolvechoice113 = ResolveChoice.apply(frame = applymapping112, choice = 'make_cols', transformation_ctx = 'resolvechoice113')
    dropnullfields114 = DropNullFields.apply(frame = resolvechoice113, transformation_ctx = 'dropnullfields114')
    datasink115 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields114, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.environment_active_geofence', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink115')
except:
    print('Failed')

try:
    datasource112 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_fix_csv', transformation_ctx = 'datasource112')
    applymapping113 = ApplyMapping.apply(frame = datasource112, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('status', 'long', 'status', 'long'), ('service', 'long', 'service', 'long'), ('latitude', 'double', 'latitude', 'double'), ('longitude', 'double', 'longitude', 'double'), ('altitude', 'double', 'altitude', 'double'), ('position_covariance', 'array', 'position_covariance', 'string'), ('position_covariance_type', 'long', 'position_covariance_type', 'long')], transformation_ctx = 'applymapping113')
    resolvechoice114 = ResolveChoice.apply(frame = applymapping113, choice = 'make_cols', transformation_ctx = 'resolvechoice114')
    dropnullfields115 = DropNullFields.apply(frame = resolvechoice114, transformation_ctx = 'dropnullfields115')
    datasink116 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields115, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_fix', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink116')
except:
    print('Failed')

try:
    datasource113 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_arbitrated_steering_commands_csv', transformation_ctx = 'datasource113')
    applymapping114 = ApplyMapping.apply(frame = datasource113, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('mode', 'long', 'mode', 'long'), ('curvature', 'double', 'curvature', 'double'), ('max_curvature_rate', 'double', 'max_curvature_rate', 'double')], transformation_ctx = 'applymapping114')
    resolvechoice115 = ResolveChoice.apply(frame = applymapping114, choice = 'make_cols', transformation_ctx = 'resolvechoice115')
    dropnullfields116 = DropNullFields.apply(frame = resolvechoice115, transformation_ctx = 'dropnullfields116')
    datasink117 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields116, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_arbitrated_steering_commands', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink117')
except:
    print('Failed')

try:
    datasource114 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_arbitrated_speed_commands_csv', transformation_ctx = 'datasource114')
    applymapping115 = ApplyMapping.apply(frame = datasource114, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('mode', 'long', 'mode', 'long'), ('speed', 'double', 'speed', 'double'), ('acceleration_limit', 'double', 'acceleration_limit', 'double'), ('deceleration_limit', 'double', 'deceleration_limit', 'double')], transformation_ctx = 'applymapping115')
    resolvechoice116 = ResolveChoice.apply(frame = applymapping115, choice = 'make_cols', transformation_ctx = 'resolvechoice116')
    dropnullfields117 = DropNullFields.apply(frame = resolvechoice116, transformation_ctx = 'dropnullfields117')
    datasink118 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields117, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_arbitrated_speed_commands', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink118')
except:
    print('Failed')

try:
    datasource115 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_route_csv', transformation_ctx = 'datasource115')
    applymapping116 = ApplyMapping.apply(frame = datasource115, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('route_id', 'string', 'route_id', 'string'), ('route_version', 'long', 'route_version', 'long'), ('route_name', 'string', 'route_name', 'string'), ('shortest_path_lanelet_ids', 'array', 'shortest_path_lanelet_ids', 'string'), ('route_path_lanelet_ids', 'array', 'route_path_lanelet_ids', 'string'), ('end_point', 'string', 'end_point', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'double', 'z', 'double')], transformation_ctx = 'applymapping116')
    resolvechoice117 = ResolveChoice.apply(frame = applymapping116, choice = 'make_cols', transformation_ctx = 'resolvechoice117')
    dropnullfields118 = DropNullFields.apply(frame = resolvechoice117, transformation_ctx = 'dropnullfields118')
    datasink119 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields118, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_route', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink119')
except:
    print('Failed')

try:
    datasource116 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_brake_command_echo_csv', transformation_ctx = 'datasource116')
    applymapping117 = ApplyMapping.apply(frame = datasource116, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('brake_pedal', 'double', 'brake_pedal', 'double')], transformation_ctx = 'applymapping117')
    resolvechoice118 = ResolveChoice.apply(frame = applymapping117, choice = 'make_cols', transformation_ctx = 'resolvechoice118')
    dropnullfields119 = DropNullFields.apply(frame = resolvechoice118, transformation_ctx = 'dropnullfields119')
    datasink120 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields119, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_brake_command_echo', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink120')
except:
    print('Failed')

try:
    datasource117 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_corrimudata_csv', transformation_ctx = 'datasource117')
    applymapping118 = ApplyMapping.apply(frame = datasource117, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('pitch_rate', 'string', 'pitch_rate', 'string'), ('roll_rate', 'string', 'roll_rate', 'string'), ('yaw_rate', 'string', 'yaw_rate', 'string'), ('lateral_acceleration', 'string', 'lateral_acceleration', 'string'), ('longitudinal_acceleration', 'string', 'longitudinal_acceleration', 'string'), ('vertical_acceleration', 'string', 'vertical_acceleration', 'string')], transformation_ctx = 'applymapping118')
    resolvechoice119 = ResolveChoice.apply(frame = applymapping118, choice = 'make_cols', transformation_ctx = 'resolvechoice119')
    dropnullfields120 = DropNullFields.apply(frame = resolvechoice119, transformation_ctx = 'dropnullfields120')
    datasink121 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields120, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_corrimudata', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink121')
except:
    print('Failed')

try:
    datasource118 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_controller_robot_status_csv', transformation_ctx = 'datasource118')
    applymapping119 = ApplyMapping.apply(frame = datasource118, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('robot_active', 'boolean', 'robot_active', 'boolean'), ('robot_enabled', 'boolean', 'robot_enabled', 'boolean'), ('torque', 'double', 'torque', 'double'), ('torque_validity', 'boolean', 'torque_validity', 'boolean'), ('brake_decel', 'double', 'brake_decel', 'double'), ('brake_decel_validity', 'boolean', 'brake_decel_validity', 'boolean'), ('throttle_effort', 'double', 'throttle_effort', 'double'), ('throttle_effort_validity', 'boolean', 'throttle_effort_validity', 'boolean'), ('braking_effort', 'double', 'braking_effort', 'double'), ('braking_effort_validity', 'boolean', 'braking_effort_validity', 'boolean')], transformation_ctx = 'applymapping119')
    resolvechoice120 = ResolveChoice.apply(frame = applymapping119, choice = 'make_cols', transformation_ctx = 'resolvechoice120')
    dropnullfields121 = DropNullFields.apply(frame = resolvechoice120, transformation_ctx = 'dropnullfields121')
    datasink122 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields121, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_controller_robot_status', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink122')
except:
    print('Failed')

try:
    datasource119 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_dbw_enabled_feedback_csv', transformation_ctx = 'datasource119')
    applymapping120 = ApplyMapping.apply(frame = datasource119, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'boolean', 'data', 'boolean')], transformation_ctx = 'applymapping120')
    resolvechoice121 = ResolveChoice.apply(frame = applymapping120, choice = 'make_cols', transformation_ctx = 'resolvechoice121')
    dropnullfields122 = DropNullFields.apply(frame = resolvechoice121, transformation_ctx = 'dropnullfields122')
    datasink123 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields122, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_dbw_enabled_feedback', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink123')
except:
    print('Failed')

try:
    datasource120 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_angular_gravity_csv', transformation_ctx = 'datasource120')
    applymapping121 = ApplyMapping.apply(frame = datasource120, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping121')
    resolvechoice122 = ResolveChoice.apply(frame = applymapping121, choice = 'make_cols', transformation_ctx = 'resolvechoice122')
    dropnullfields123 = DropNullFields.apply(frame = resolvechoice122, transformation_ctx = 'dropnullfields123')
    datasink124 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields123, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_angular_gravity', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink124')
except:
    print('Failed')

try:
    datasource121 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_next_waypoint_mark_csv', transformation_ctx = 'datasource121')
    applymapping122 = ApplyMapping.apply(frame = datasource121, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('ns', 'string', 'ns', 'string'), ('id', 'long', 'id', 'long'), ('type', 'long', 'type', 'long'), ('action', 'long', 'action', 'long'), ('pose', 'string', 'pose', 'string'), ('position', 'string', 'position', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'double', 'z', 'double'), ('orientation', 'string', 'orientation', 'string'), ('w', 'double', 'w', 'double'), ('scale', 'string', 'scale', 'string'), ('color', 'string', 'color', 'string'), ('r', 'double', 'r', 'double'), ('g', 'double', 'g', 'double'), ('b', 'double', 'b', 'double'), ('a', 'double', 'a', 'double'), ('lifetime', 'string', 'lifetime', 'string'), ('frame_locked', 'boolean', 'frame_locked', 'boolean'), ('points', 'array', 'points', 'string'), ('colors', 'array', 'colors', 'string'), ('text', 'string', 'text', 'string'), ('mesh_resource', 'string', 'mesh_resource', 'string'), ('mesh_use_embedded_materials', 'boolean', 'mesh_use_embedded_materials', 'boolean')], transformation_ctx = 'applymapping122')
    resolvechoice123 = ResolveChoice.apply(frame = applymapping122, choice = 'make_cols', transformation_ctx = 'resolvechoice123')
    dropnullfields124 = DropNullFields.apply(frame = resolvechoice123, transformation_ctx = 'dropnullfields124')
    datasink125 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields124, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_next_waypoint_mark', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink125')
except:
    print('Failed')

try:
    datasource122 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_raw_csv', transformation_ctx = 'datasource122')
    applymapping123 = ApplyMapping.apply(frame = datasource122, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('twist', 'string', 'twist', 'string'), ('linear', 'string', 'linear', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'string', 'z', 'string'), ('angular', 'string', 'angular', 'string')], transformation_ctx = 'applymapping123')
    resolvechoice124 = ResolveChoice.apply(frame = applymapping123, choice = 'make_cols', transformation_ctx = 'resolvechoice124')
    dropnullfields125 = DropNullFields.apply(frame = resolvechoice124, transformation_ctx = 'dropnullfields125')
    datasink126 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields125, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_raw', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink126')
except:
    print('Failed')

try:
    datasource123 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_route_event_csv', transformation_ctx = 'datasource123')
    applymapping124 = ApplyMapping.apply(frame = datasource123, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('event', 'long', 'event', 'long')], transformation_ctx = 'applymapping124')
    resolvechoice125 = ResolveChoice.apply(frame = applymapping124, choice = 'make_cols', transformation_ctx = 'resolvechoice125')
    dropnullfields126 = DropNullFields.apply(frame = resolvechoice125, transformation_ctx = 'dropnullfields126')
    datasink127 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields126, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_route_event', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink127')
except:
    print('Failed')

try:
    datasource124 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_plugins_stopandwaitplugin_jerk_val_csv', transformation_ctx = 'datasource124')
    applymapping125 = ApplyMapping.apply(frame = datasource124, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping125')
    resolvechoice126 = ResolveChoice.apply(frame = applymapping125, choice = 'make_cols', transformation_ctx = 'resolvechoice126')
    dropnullfields127 = DropNullFields.apply(frame = resolvechoice126, transformation_ctx = 'dropnullfields127')
    datasink128 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields127, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_plugins_stopandwaitplugin_jerk_val', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink128')
except:
    print('Failed')

try:
    datasource125 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_curvature_feedback_csv', transformation_ctx = 'datasource125')
    applymapping126 = ApplyMapping.apply(frame = datasource125, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('curvature', 'string', 'curvature', 'string')], transformation_ctx = 'applymapping126')
    resolvechoice127 = ResolveChoice.apply(frame = applymapping126, choice = 'make_cols', transformation_ctx = 'resolvechoice127')
    dropnullfields128 = DropNullFields.apply(frame = resolvechoice127, transformation_ctx = 'dropnullfields128')
    datasink129 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields128, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_curvature_feedback', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink129')
except:
    print('Failed')

try:
    datasource126 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_cmd_csv', transformation_ctx = 'datasource126')
    applymapping127 = ApplyMapping.apply(frame = datasource126, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('twist', 'string', 'twist', 'string'), ('linear', 'string', 'linear', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'string', 'z', 'string'), ('angular', 'string', 'angular', 'string')], transformation_ctx = 'applymapping127')
    resolvechoice128 = ResolveChoice.apply(frame = applymapping127, choice = 'make_cols', transformation_ctx = 'resolvechoice128')
    dropnullfields129 = DropNullFields.apply(frame = resolvechoice128, transformation_ctx = 'dropnullfields129')
    datasink130 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields129, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_cmd', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink130')
except:
    print('Failed')

try:
    datasource127 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_limitation_debug_ctrl_lateral_jerk_csv', transformation_ctx = 'datasource127')
    applymapping128 = ApplyMapping.apply(frame = datasource127, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping128')
    resolvechoice129 = ResolveChoice.apply(frame = applymapping128, choice = 'make_cols', transformation_ctx = 'resolvechoice129')
    dropnullfields130 = DropNullFields.apply(frame = resolvechoice129, transformation_ctx = 'dropnullfields130')
    datasink131 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields130, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_limitation_debug_ctrl_lateral_jerk', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink131')
except:
    print('Failed')

try:
    datasource128 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_can_steering_wheel_angle_csv', transformation_ctx = 'datasource128')
    applymapping129 = ApplyMapping.apply(frame = datasource128, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping129')
    resolvechoice130 = ResolveChoice.apply(frame = applymapping129, choice = 'make_cols', transformation_ctx = 'resolvechoice130')
    dropnullfields131 = DropNullFields.apply(frame = resolvechoice130, transformation_ctx = 'dropnullfields131')
    datasink132 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields131, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_can_steering_wheel_angle', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink132')
except:
    print('Failed')

try:
    datasource129 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_ctrl_cmd_csv', transformation_ctx = 'datasource129')
    applymapping130 = ApplyMapping.apply(frame = datasource129, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('cmd', 'string', 'cmd', 'string'), ('linear_velocity', 'double', 'linear_velocity', 'double'), ('linear_acceleration', 'double', 'linear_acceleration', 'double'), ('steering_angle', 'string', 'steering_angle', 'string')], transformation_ctx = 'applymapping130')
    resolvechoice131 = ResolveChoice.apply(frame = applymapping130, choice = 'make_cols', transformation_ctx = 'resolvechoice131')
    dropnullfields132 = DropNullFields.apply(frame = resolvechoice131, transformation_ctx = 'dropnullfields132')
    datasink133 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields132, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_ctrl_cmd', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink133')
except:
    print('Failed')

try:
    datasource130 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_plugin_discovery_csv', transformation_ctx = 'datasource130')
    applymapping131 = ApplyMapping.apply(frame = datasource130, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('name', 'string', 'name', 'string'), ('versionid', 'string', 'versionid', 'string'), ('type', 'long', 'type', 'long'), ('available', 'boolean', 'available', 'boolean'), ('activated', 'boolean', 'activated', 'boolean'), ('capability', 'string', 'capability', 'string')], transformation_ctx = 'applymapping131')
    resolvechoice132 = ResolveChoice.apply(frame = applymapping131, choice = 'make_cols', transformation_ctx = 'resolvechoice132')
    dropnullfields133 = DropNullFields.apply(frame = resolvechoice132, transformation_ctx = 'dropnullfields133')
    datasink134 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields133, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_plugin_discovery', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink134')
except:
    print('Failed')

try:
    datasource131 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_bestvel_csv', transformation_ctx = 'datasource131')
    applymapping132 = ApplyMapping.apply(frame = datasource131, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('solution_status', 'string', 'solution_status', 'string'), ('velocity_type', 'string', 'velocity_type', 'string'), ('latency', 'double', 'latency', 'double'), ('age', 'double', 'age', 'double'), ('horizontal_speed', 'string', 'horizontal_speed', 'string'), ('track_ground', 'double', 'track_ground', 'double'), ('vertical_speed', 'string', 'vertical_speed', 'string')], transformation_ctx = 'applymapping132')
    resolvechoice133 = ResolveChoice.apply(frame = applymapping132, choice = 'make_cols', transformation_ctx = 'resolvechoice133')
    dropnullfields134 = DropNullFields.apply(frame = resolvechoice133, transformation_ctx = 'dropnullfields134')
    datasink135 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields134, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_bestvel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink135')
except:
    print('Failed')

try:
    datasource132 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_limitation_debug_ctrl_lateral_accel_csv', transformation_ctx = 'datasource132')
    applymapping133 = ApplyMapping.apply(frame = datasource132, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping133')
    resolvechoice134 = ResolveChoice.apply(frame = applymapping133, choice = 'make_cols', transformation_ctx = 'resolvechoice134')
    dropnullfields135 = DropNullFields.apply(frame = resolvechoice134, transformation_ctx = 'dropnullfields135')
    datasink136 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields135, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_limitation_debug_ctrl_lateral_accel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink136')
except:
    print('Failed')

try:
    datasource133 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_comms_outbound_binary_msg_csv', transformation_ctx = 'datasource133')
    applymapping134 = ApplyMapping.apply(frame = datasource133, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('messagetype', 'string', 'messagetype', 'string'), ('content', 'array', 'content', 'string')], transformation_ctx = 'applymapping134')
    resolvechoice135 = ResolveChoice.apply(frame = applymapping134, choice = 'make_cols', transformation_ctx = 'resolvechoice135')
    dropnullfields136 = DropNullFields.apply(frame = resolvechoice135, transformation_ctx = 'dropnullfields136')
    datasink137 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields136, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_comms_outbound_binary_msg', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink137')
except:
    print('Failed')

try:
    datasource134 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_brake_feedback_csv', transformation_ctx = 'datasource134')
    applymapping135 = ApplyMapping.apply(frame = datasource134, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('brake_pedal', 'double', 'brake_pedal', 'double')], transformation_ctx = 'applymapping135')
    resolvechoice136 = ResolveChoice.apply(frame = applymapping135, choice = 'make_cols', transformation_ctx = 'resolvechoice136')
    dropnullfields137 = DropNullFields.apply(frame = resolvechoice136, transformation_ctx = 'dropnullfields137')
    datasink138 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields137, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_brake_feedback', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink138')
except:
    print('Failed')

try:
    datasource135 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_result_twist_lateral_accel_csv', transformation_ctx = 'datasource135')
    applymapping136 = ApplyMapping.apply(frame = datasource135, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping136')
    resolvechoice137 = ResolveChoice.apply(frame = applymapping136, choice = 'make_cols', transformation_ctx = 'resolvechoice137')
    dropnullfields138 = DropNullFields.apply(frame = resolvechoice137, transformation_ctx = 'dropnullfields138')
    datasink139 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields138, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_result_twist_lateral_accel', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink139')
except:
    print('Failed')

try:
    datasource136 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_state_csv', transformation_ctx = 'datasource136')
    applymapping137 = ApplyMapping.apply(frame = datasource136, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('state', 'long', 'state', 'long')], transformation_ctx = 'applymapping137')
    resolvechoice138 = ResolveChoice.apply(frame = applymapping137, choice = 'make_cols', transformation_ctx = 'resolvechoice138')
    dropnullfields139 = DropNullFields.apply(frame = resolvechoice138, transformation_ctx = 'dropnullfields139')
    datasink140 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields139, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_state', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink140')
except:
    print('Failed')

try:
    datasource137 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_driver_discovery_csv', transformation_ctx = 'datasource137')
    applymapping138 = ApplyMapping.apply(frame = datasource137, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('name', 'string', 'name', 'string'), ('status', 'long', 'status', 'long'), ('can', 'boolean', 'can', 'boolean'), ('radar', 'boolean', 'radar', 'boolean'), ('gnss', 'boolean', 'gnss', 'boolean'), ('lidar', 'boolean', 'lidar', 'boolean'), ('roadway_sensor', 'boolean', 'roadway_sensor', 'boolean'), ('comms', 'boolean', 'comms', 'boolean'), ('controller', 'boolean', 'controller', 'boolean'), ('camera', 'boolean', 'camera', 'boolean'), ('imu', 'boolean', 'imu', 'boolean'), ('trailer_angle_sensor', 'boolean', 'trailer_angle_sensor', 'boolean'), ('lightbar', 'boolean', 'lightbar', 'boolean')], transformation_ctx = 'applymapping138')
    resolvechoice139 = ResolveChoice.apply(frame = applymapping138, choice = 'make_cols', transformation_ctx = 'resolvechoice139')
    dropnullfields140 = DropNullFields.apply(frame = resolvechoice139, transformation_ctx = 'dropnullfields140')
    datasink141 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields140, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_driver_discovery', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink141')
except:
    print('Failed')

try:
    datasource138 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_deviation_of_current_position_csv', transformation_ctx = 'datasource138')
    applymapping139 = ApplyMapping.apply(frame = datasource138, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'string', 'data', 'string')], transformation_ctx = 'applymapping139')
    resolvechoice140 = ResolveChoice.apply(frame = applymapping139, choice = 'make_cols', transformation_ctx = 'resolvechoice140')
    dropnullfields141 = DropNullFields.apply(frame = resolvechoice140, transformation_ctx = 'dropnullfields141')
    datasink142 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields141, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_deviation_of_current_position', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink142')
except:
    print('Failed')

try:
    datasource139 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_next_target_mark_csv', transformation_ctx = 'datasource139')
    applymapping140 = ApplyMapping.apply(frame = datasource139, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('ns', 'string', 'ns', 'string'), ('id', 'long', 'id', 'long'), ('type', 'long', 'type', 'long'), ('action', 'long', 'action', 'long'), ('pose', 'string', 'pose', 'string'), ('position', 'string', 'position', 'string'), ('x', 'double', 'x', 'double'), ('y', 'double', 'y', 'double'), ('z', 'double', 'z', 'double'), ('orientation', 'string', 'orientation', 'string'), ('w', 'double', 'w', 'double'), ('scale', 'string', 'scale', 'string'), ('color', 'string', 'color', 'string'), ('r', 'double', 'r', 'double'), ('g', 'double', 'g', 'double'), ('b', 'double', 'b', 'double'), ('a', 'double', 'a', 'double'), ('lifetime', 'string', 'lifetime', 'string'), ('frame_locked', 'boolean', 'frame_locked', 'boolean'), ('points', 'array', 'points', 'string'), ('colors', 'array', 'colors', 'string'), ('text', 'string', 'text', 'string'), ('mesh_resource', 'string', 'mesh_resource', 'string'), ('mesh_use_embedded_materials', 'boolean', 'mesh_use_embedded_materials', 'boolean')], transformation_ctx = 'applymapping140')
    resolvechoice141 = ResolveChoice.apply(frame = applymapping140, choice = 'make_cols', transformation_ctx = 'resolvechoice141')
    dropnullfields142 = DropNullFields.apply(frame = resolvechoice141, transformation_ctx = 'dropnullfields142')
    datasink143 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields142, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_next_target_mark', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink143')
except:
    print('Failed')

try:
    datasource140 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_ctrl_raw_csv', transformation_ctx = 'datasource140')
    applymapping141 = ApplyMapping.apply(frame = datasource140, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('cmd', 'string', 'cmd', 'string'), ('linear_velocity', 'double', 'linear_velocity', 'double'), ('linear_acceleration', 'double', 'linear_acceleration', 'double'), ('steering_angle', 'string', 'steering_angle', 'string')], transformation_ctx = 'applymapping141')
    resolvechoice142 = ResolveChoice.apply(frame = applymapping141, choice = 'make_cols', transformation_ctx = 'resolvechoice142')
    dropnullfields143 = DropNullFields.apply(frame = resolvechoice142, transformation_ctx = 'dropnullfields143')
    datasink144 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields143, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_ctrl_raw', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink144')
except:
    print('Failed')

try:
    datasource141 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_gear_command_echo_csv', transformation_ctx = 'datasource141')
    applymapping142 = ApplyMapping.apply(frame = datasource141, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('command', 'string', 'command', 'string'), ('gear', 'long', 'gear', 'long')], transformation_ctx = 'applymapping142')
    resolvechoice143 = ResolveChoice.apply(frame = applymapping142, choice = 'make_cols', transformation_ctx = 'resolvechoice143')
    dropnullfields144 = DropNullFields.apply(frame = resolvechoice143, transformation_ctx = 'dropnullfields144')
    datasink145 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields144, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_gear_command_echo', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink145')
except:
    print('Failed')

try:
    datasource142 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_can_brake_position_csv', transformation_ctx = 'datasource142')
    applymapping143 = ApplyMapping.apply(frame = datasource142, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping143')
    resolvechoice144 = ResolveChoice.apply(frame = applymapping143, choice = 'make_cols', transformation_ctx = 'resolvechoice144')
    dropnullfields145 = DropNullFields.apply(frame = resolvechoice144, transformation_ctx = 'dropnullfields145')
    datasink146 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields145, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_can_brake_position', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink146')
except:
    print('Failed')

try:
    datasource143 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='hardware_interface_dual_antenna_heading_csv', transformation_ctx = 'datasource143')
    applymapping144 = ApplyMapping.apply(frame = datasource143, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('header', 'string', 'header', 'string'), ('seq', 'long', 'seq', 'long'), ('stamp', 'string', 'stamp', 'string'), ('secs', 'long', 'secs', 'long'), ('nsecs', 'long', 'nsecs', 'long'), ('frame_id', 'string', 'frame_id', 'string'), ('novatel_msg_header', 'string', 'novatel_msg_header', 'string'), ('message_name', 'string', 'message_name', 'string'), ('port', 'string', 'port', 'string'), ('sequence_num', 'long', 'sequence_num', 'long'), ('percent_idle_time', 'double', 'percent_idle_time', 'double'), ('gps_time_status', 'string', 'gps_time_status', 'string'), ('gps_week_num', 'long', 'gps_week_num', 'long'), ('gps_seconds', 'double', 'gps_seconds', 'double'), ('receiver_status', 'string', 'receiver_status', 'string'), ('original_status_code', 'long', 'original_status_code', 'long'), ('error_flag', 'boolean', 'error_flag', 'boolean'), ('temperature_flag', 'boolean', 'temperature_flag', 'boolean'), ('voltage_supply_flag', 'boolean', 'voltage_supply_flag', 'boolean'), ('antenna_powered', 'boolean', 'antenna_powered', 'boolean'), ('antenna_is_open', 'boolean', 'antenna_is_open', 'boolean'), ('antenna_is_shorted', 'boolean', 'antenna_is_shorted', 'boolean'), ('cpu_overload_flag', 'boolean', 'cpu_overload_flag', 'boolean'), ('com1_buffer_overrun', 'boolean', 'com1_buffer_overrun', 'boolean'), ('com2_buffer_overrun', 'boolean', 'com2_buffer_overrun', 'boolean'), ('com3_buffer_overrun', 'boolean', 'com3_buffer_overrun', 'boolean'), ('usb_buffer_overrun', 'boolean', 'usb_buffer_overrun', 'boolean'), ('rf1_agc_flag', 'boolean', 'rf1_agc_flag', 'boolean'), ('rf2_agc_flag', 'boolean', 'rf2_agc_flag', 'boolean'), ('almanac_flag', 'boolean', 'almanac_flag', 'boolean'), ('position_solution_flag', 'boolean', 'position_solution_flag', 'boolean'), ('position_fixed_flag', 'boolean', 'position_fixed_flag', 'boolean'), ('clock_steering_status_enabled', 'boolean', 'clock_steering_status_enabled', 'boolean'), ('clock_model_flag', 'boolean', 'clock_model_flag', 'boolean'), ('oemv_external_oscillator_flag', 'boolean', 'oemv_external_oscillator_flag', 'boolean'), ('software_resource_flag', 'boolean', 'software_resource_flag', 'boolean'), ('aux1_status_event_flag', 'boolean', 'aux1_status_event_flag', 'boolean'), ('aux2_status_event_flag', 'boolean', 'aux2_status_event_flag', 'boolean'), ('aux3_status_event_flag', 'boolean', 'aux3_status_event_flag', 'boolean'), ('receiver_software_version', 'long', 'receiver_software_version', 'long'), ('solution_status', 'string', 'solution_status', 'string'), ('position_type', 'string', 'position_type', 'string'), ('baseline_length', 'double', 'baseline_length', 'double'), ('heading', 'double', 'heading', 'double'), ('pitch', 'double', 'pitch', 'double'), ('heading_sigma', 'double', 'heading_sigma', 'double'), ('pitch_sigma', 'double', 'pitch_sigma', 'double'), ('station_id', 'string', 'station_id', 'string'), ('num_satellites_tracked', 'long', 'num_satellites_tracked', 'long'), ('num_satellites_used_in_solution', 'long', 'num_satellites_used_in_solution', 'long'), ('num_satellites_above_elevation_mask_angle', 'long', 'num_satellites_above_elevation_mask_angle', 'long'), ('num_satellites_above_elevation_mask_angle_l2', 'long', 'num_satellites_above_elevation_mask_angle_l2', 'long'), ('solution_source', 'long', 'solution_source', 'long'), ('extended_solution_status', 'string', 'extended_solution_status', 'string'), ('original_mask', 'long', 'original_mask', 'long'), ('advance_rtk_verified', 'boolean', 'advance_rtk_verified', 'boolean'), ('psuedorange_iono_correction', 'string', 'psuedorange_iono_correction', 'string'), ('signal_mask', 'string', 'signal_mask', 'string'), ('gps_l1_used_in_solution', 'boolean', 'gps_l1_used_in_solution', 'boolean'), ('gps_l2_used_in_solution', 'boolean', 'gps_l2_used_in_solution', 'boolean'), ('gps_l3_used_in_solution', 'boolean', 'gps_l3_used_in_solution', 'boolean'), ('glonass_l1_used_in_solution', 'boolean', 'glonass_l1_used_in_solution', 'boolean'), ('glonass_l2_used_in_solution', 'boolean', 'glonass_l2_used_in_solution', 'boolean')], transformation_ctx = 'applymapping144')
    resolvechoice145 = ResolveChoice.apply(frame = applymapping144, choice = 'make_cols', transformation_ctx = 'resolvechoice145')
    dropnullfields146 = DropNullFields.apply(frame = resolvechoice145, transformation_ctx = 'dropnullfields146')
    datasink147 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields146, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.hardware_interface_dual_antenna_heading', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink147')
except:
    print('Failed')

try:
    datasource144 = glueContext.create_dynamic_frame.from_catalog(database = glue_db, table_name ='guidance_twist_filter_limitation_debug_twist_lateral_jerk_csv', transformation_ctx = 'datasource144')
    applymapping145 = ApplyMapping.apply(frame = datasource144, mappings = [('rosbagtimestamp', 'long', 'rosbagtimestamp', 'long'), ('data', 'double', 'data', 'double')], transformation_ctx = 'applymapping145')
    resolvechoice146 = ResolveChoice.apply(frame = applymapping145, choice = 'make_cols', transformation_ctx = 'resolvechoice146')
    dropnullfields147 = DropNullFields.apply(frame = resolvechoice146, transformation_ctx = 'dropnullfields147')
    datasink148 = glueContext.write_dynamic_frame.from_jdbc_conf(frame = dropnullfields147, catalog_connection = 'core-validation-redshift-conn', connection_options = {'dbtable': f'{redshift_db_name}.guidance_twist_filter_limitation_debug_twist_lateral_jerk', 'database': 'carma-core'}, redshift_tmp_dir = args['TempDir'], transformation_ctx = 'datasink148')
except:
    print('Failed')

job.commit()