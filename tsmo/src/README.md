This directory contains several scripts that can be used to extract relevant data from the various kafka topics and generate
the required plots to validate Carma-Streets UC3 functionality. The user should clone this repo and switch to the UC3 Analysis 
branch. Inside of the "tsmo" directory, create a "data" directory with a set of subdirectories that are called out in the "constants.py"
script (Intersection_Model, Parsed_Log_Output, etc.). After all of the necessary directories have been created, place the raw
test data (Kafka logs/csv files) in the "Raw_Log_Files" directory. Then place the intersection model json file, corresponding to the
intersection that was tested on, in the "Intersection_Model" directory.

At this point, you are ready to begin using the scripts. The first script that should be run is the "intersection_model_parser.py" script,
which will generate a csv containing lanelet ids and their corresponding length. This data is required for several of the plots to be generated. A description of the functionality of the scripts and how to run them can be found at the top of all the scripts in this repo. 
The user can also simply run the desired script with no parameters (ex: python3 frequency_plotter.py) and the script will prompt the user
for the appropriate input. All of the plotting scripts require that the user has previously extracted required data from the kafka logs using the "parser" scripts, as well as some test metadata (vehicle id, signal group, etc.)

Available plots to be generated:
1. Vehicle Acceleration vs time (source: status and intent topic, script: vehicle_acceleration_plotter.py)
2. Spat broadcast frequency vs time (source: modified spat topic, script: frequency_plotter.py)
3. MOM broadcast frequency vs time (source: scheduling plan topic, script: frequency_plotter.py)
4. Vehicle trajectory vs time, w/ spat signal group data (source: status and intent, modified spat topics, script: vehicle_dist_time_plotter.py)
5. All signal groups status vs time (source: modified spat topic, script: all_signal_groups_plotter.py)
6. Vehicle entering time vs time (state data available as well), w/ spat signal group data (source: schedule logs, scheduling plan kafka topic, script: one_veh_et_spat_plotter.py)