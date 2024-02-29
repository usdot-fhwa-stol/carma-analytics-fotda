# VRU use-case scripts

This directory contains scripts to help with data analysis for VRU use-cases.

## `extract_rtf_data`

This script takes in a Docker container log output and extracts realtime factor (RTF) data, outputting the results as
comma-separated values (CSV) with the following format:

```text
<sim_time_nanoseconds>,<rtf_value>
<sim_time_nanoseconds>,<rtf_value>
...
<sim_time_nanoseconds>,<rtf_value>
```

### Usage examples

Output the CSV data directly to the terminal:

```console
docker logs carma-simulation 2>&1 | ./extract_rtf_data
```

Output the CSV data to a file:

```console
docker logs carma-simulation 2>&1 | ./extract_rtf_data > rtf_data.csv
```

Pipe the CSV data into the `plot_rtf_data` script (see below):

```console
docker logs carma-simulation 2>&1 | ./extract_rtf_data | ./plot_rtf_data
```

## `plot_rtf_data`

This script takes in CSV-formatted RTF data from standard input or a file and plots the results. This script
has an optional `--min-required` argument that will plot a horizontal line at the level specified.

### Usage examples

Plot the RTF data from a file:

```console
./plot_rtf_data rtf_data.csv
```

```console
./plot_rtf_data --min-required 0.50 rtf_data.csv
```

Plot the RTF data from standard input (_e.g._, through a pipe):

```console
docker logs carma-simulation 2>&1 | ./extract_rtf_data | ./plot_rtf_data
```

```console
docker logs carma-simulation 2>&1 | ./extract_rtf_data | ./plot_rtf_data --min-required 0.35
```

## `plot_sdsm_position_error`

This script takes in two V2XHub log messages (one for detected objects and another for SDSMs) and plots the position
difference between entries in the two logs. For each detected object and for each timestamp, the script searches in
the SDSM log file for a corresponding entry. It then calculates the distance between the two entires. Finally, the
script plots the results for each object for the duration of the simulation.

### Usage examples

Plot the data:

```console
./plot_sdsm_position_error \
  --sdsm-log <path_to_logs_dir>/v2xhub_sdsm_sub.log \
  --detection-log <path_to_logs_dir>/v2xhub_sim_sensor_detected_object.log
```
