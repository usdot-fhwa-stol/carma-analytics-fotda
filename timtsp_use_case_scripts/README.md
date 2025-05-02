# VRU use-case scripts

This directory contains scripts to help with data analysis for TIM/TSP Usecase.

## `extract_srm_timestamp`

This script takes as input the Priority Request Server and Priority Request Generator log files from the MMITSS MRP and VSP respectively. It ouputs a single column of simulation timestamps for when the SRM was produced at the VSP and received at the MRP in seconds.
comma-separated values (CSV) with the following format:

```text
<sim_time_seconds>
<sim_time_seconds>
...
<sim_time_seconds>
```

### Usage examples

Output the CSV data directly to a file:

```console
cat vehicle_prgLog.log  | ./extract_srm_timestamp > vsp_srm_data.csv
```

or

```console
cat cda_town5_prsLog.log  | ./extract_srm_timestamp > mrp_srm_data.csv
```

## `plot_srm_latency`

This script takes in CSV-formatted vsp and mrp SRM timestamps and plots simulation time latency

```
usage: plot_rtf [-h] [--max-acceptable MAX_ACCEPTABLE] [--plots-dir PLOTS_DIR] [--mrp-data-file MRP_DATA_FILE] [--vsp-data-file VSP_DATA_FILE]

Plot SRM end-to-end simulation latency from a comma-separate values (CSV) file

options:
  -h, --help            show this help message and exit
  --max-acceptable MAX_ACCEPTABLE
                        Maximum acceptable latency. This will add a line in the generated plot
  --plots-dir PLOTS_DIR
                        Directory to store the generated plot
  --mrp-data-file MRP_DATA_FILE
                        csv file containing MRP data
  --vsp-data-file VSP_DATA_FILE
                        csv file containing vsp data
```

## `plot_service_time_data`

This script takes in CSV-formatted data for cdasim, vsp and mrp with simulation time and corresponding system time. It plots each simulation time update against the difference in system between between cdasim and the specific service. cdasim is taken as the base to compare against.

```
usage: plot_service_time_data <path to directory with service logs>
