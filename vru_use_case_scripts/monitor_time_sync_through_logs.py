import time
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import shutil
import threading
import docker
from pathlib import Path

# global values for plotting and other tracking
IS_SYNCED_PLOT_NAME = 'Is Synced?'
# stores how long (in system wall time) it took for the tool receive the next simulation time step
# since MOSAIC commanded its first co-simulation tool to the next time step
tool_activation_delay_since_last_cdasim_step = {}

PATTERN_TO_MATCH = 'Simulation Time: (\d+) where current system time is: (\d+)'

# Main class that tracks and processes log lines retrieved from multiple files and observers
class LogMonitor:
    def __init__(self, files):
        # Pattern to match "Simulation Time: X where current system time is: Y"
        self.log_pattern = re.compile(PATTERN_TO_MATCH)
        # Store system times by simulation time, across all files {simulation_time, list({system_time, file_name})}
        self.simulation_times = {}
        # Latest line from each file that was processed {file_name, latest_line_num}
        self.latest_line_num = {}
        self.file_size = len(files)
        self.files = files
        self.docker_client = docker.from_env()

    def capture_docker_container_log(self, file_path, container_name):
        # Try to get the container by name
        try:
            container = self.docker_client.containers.get(container_name)
        except docker.errors.NotFound:
            print(f"Container {container_name} not found.")
            return

        # Stream the logs
        with open(file_path, 'w') as f:
            for line in container.logs(stream=True, follow=True):
                decoded_line = line.decode('utf-8')
                self.process_line(decoded_line, file_path)
                # Write to the file and flush
                f.write(decoded_line)
                f.flush()

    def process_line(self, line, file_path):
        global tool_activation_delay_since_last_cdasim_step
        global IS_SYNCED_PLOT_NAME

        match = self.log_pattern.search(line)
        if match:
            simulation_time = int(match.group(1))
            system_time = int(match.group(2))

            if simulation_time in self.simulation_times:
                # Already have an entry for this simulation time
                # guards against repeated operation if for some reason, simulation_time is printed repeatedly again in the log

                for _, prev_file in self.simulation_times[simulation_time]:
                    if prev_file in file_path:
                        return

                self.simulation_times[simulation_time].append((system_time, file_path))

            else:
                # New simulation time, store along with its file path
                self.simulation_times[simulation_time] = [(system_time, file_path)]

            # Using try here because guaranteed to encounter keyerror when T2 = 0 T1 = -100
            try:
                # if gathered all T2 of simulation time compare with their T1:

                if (len(self.simulation_times[simulation_time]) == self.file_size and len(self.simulation_times[simulation_time - 100]) == self.file_size):

                    T1 = simulation_time - 100
                    T2 = simulation_time

                    # get max T1 and min T2
                    min_T2_system_time, _ = min(self.simulation_times[T2], key=lambda x: x[0]) # This is guaranteed to be the first time MOSAIC calls one of its tool for T2
                    max_T1_system_time, _ = max(self.simulation_times[T1], key=lambda x: x[0]) # The tool that is called the last
                    min_T1_system_time, _ = min(self.simulation_times[T1], key=lambda x: x[0]) # This is guaranteed to be the first time MOSAIC calls one of its tool for T1

                    # if minimum T2 is bigger than max T1, then it means *highly likely* that all components progressed
                    # only after every component finished progressing (or told to) which means synced simulation time
                    is_synced = min_T2_system_time > max_T1_system_time - 10 # tolerate 10ms error

                    print(f"Did T2 progress healthy? T2: {T2} where system_time difference (t2 - t1) is dt: {(min_T2_system_time - max_T1_system_time):.0f} === is synced?: {is_synced}")

                    # For plotting
                    for (tool_system_time_when_T1_received, cached_file_path) in self.simulation_times[T1]:

                        if ('MOSAIC' in cached_file_path):
                            #plots how long did MOSAIC take between two sim steps from T1 to T2
                            tool_activation_delay_since_last_cdasim_step[cached_file_path].append(min_T2_system_time - tool_system_time_when_T1_received)
                        else:
                            #plots how long did MOSAIC take to give T1 to this tool since calling the first tool
                            tool_activation_delay_since_last_cdasim_step[cached_file_path].append(tool_system_time_when_T1_received - min_T1_system_time)

                    tool_activation_delay_since_last_cdasim_step[IS_SYNCED_PLOT_NAME].append(is_synced) #

                    #already checked this simulation time, so del to save space
                    del self.simulation_times[T1]

            except KeyError:
                # Do nothing
                return


# Handles each event to a file the Watchdog observer is monitoring
class FileEventHandler(FileSystemEventHandler):
    def __init__(self, monitor, file_name):
        self.monitor = monitor
        self.specific_file_name = file_name

    def on_modified(self, event):
        if not event.is_directory:
            if event.src_path in self.specific_file_name:
                self.process_file(event.src_path)

    def process_file(self, file_path):
        with open(file_path, 'r') as file:
            try:
                line_num = self.monitor.latest_line_num[file_path]
            except KeyError:
                line_num = 0

            for i, line in enumerate(file, start=1):
                if i >= line_num:
                    self.monitor.process_line(line, file_path)
                    line_num += 1
            self.monitor.latest_line_num[file_path] = line_num

# Retrieves the most recent file with a pattern by recursively checking the search_path
def find_most_recent_file(search_path, pattern):
    list_of_files = glob.glob(os.path.join(search_path, '**', pattern), recursive=True)
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# Create Watchdog observer that monitors any event in the directory of the give file
# Watchdog is only available to monitor changes in directory not a specific file
def monitor_file(file_path, monitor):
    directory = os.path.dirname(file_path)
    event_handler = FileEventHandler(monitor, file_path)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)  # Monitoring the directory of the file
    observer.start()
    return observer

def get_full_file_path(file_name, rosout_base_path, sim_base_path, carma_streets_base_path):

    file_path = ''
    if 'rosout.log' in file_name:
        file_path = find_most_recent_file(rosout_base_path, 'rosout.log')
        if not file_path:
            print("No recent rosout.log file found.")
            return None
    elif ('Infrastructure.log' in file_name) or ('Traffic.log' in file_name) or ('MOSAIC.log' in file_name):
        file_path = find_most_recent_file(sim_base_path, file_name)
        if not file_path:
            print(f"No recent {file_name} file found.")
            return None
    elif 'v2xhub.log' in file_name:
        file_path = carma_streets_base_path + 'v2xhub.log'
        streets_dir = Path(carma_streets_base_path)
        streets_dir.mkdir(parents=True, exist_ok=True)
    else:
        file_path = file_name

    return file_path

def start_monitoring(rosout_base_path, sim_base_path, carma_streets_base_path, files_to_monitor):
    global tool_activation_delay_since_last_cdasim_step
    tool_activation_delay_since_last_cdasim_step[IS_SYNCED_PLOT_NAME] = []
    log_monitor = LogMonitor(files_to_monitor)

    observers = []

    for file_name in files_to_monitor:
        file_path = get_full_file_path(file_name, rosout_base_path, sim_base_path, carma_streets_base_path)
        print(f"Detected file to monitor: {file_path}")
        tool_activation_delay_since_last_cdasim_step[file_path] = []

        if ('v2xhub' in file_path):
            # Create a Thread to capture docker logging in the background and dont use watchdog library
            log_thread = threading.Thread(target=log_monitor.capture_docker_container_log, args=(file_path, 'v2xhub'))
            log_thread.start()
        else:
            observer = monitor_file(file_path, log_monitor)
            observers.append(observer)

    # Function to update plots for each file
    def update_plots(fig, axs, tool_activation_delay_since_last_cdasim_step):
        # The function treats MOSAIC log special because it overlays it on top of all other tool's subplots
        mosaic_data = None  # Placeholder for MOSAIC.log data
        # Check if MOSAIC.log data is present and remove it from the regular plotting loop
        for file_name in tool_activation_delay_since_last_cdasim_step.keys():
            if 'MOSAIC.log' in file_name:
                mosaic_data = tool_activation_delay_since_last_cdasim_step[file_name]
                break

        for ax, (file_name, delay_list) in zip(axs, tool_activation_delay_since_last_cdasim_step.items()):
            print(f"Size: {len(tool_activation_delay_since_last_cdasim_step[file_name])}")

            if 'MOSAIC.log' in file_name:
                continue  # Skip plotting MOSAIC.log here

            ax.clear()  # Clear the current subplot
            iterations = list(range(1, len(delay_list) + 1))
            ax.plot(iterations, delay_list, label=file_name)
            ax.set_title(file_name)
            ax.set_xlabel('Iteration Times')
            ax.set_ylabel('Max processing time between sim steps, ms')
            if (delay_list):
                ax.set_ylim(max(-500, min(delay_list)), min(500, max(delay_list))) #sometimes the delay is spiking due to other windows starting etc
            ax.legend()
            ax.grid(True)

        # After plotting all tools, overlay MOSAIC.log data on each subplot
        if mosaic_data is not None:
            iterations = list(range(1, len(mosaic_data) + 1))
            for ax in axs:
                # Differentiate red with line
                ax.plot(iterations, mosaic_data, label='MOSAIC.log', linestyle='--', color='red')
                ax.legend()  # Update the legend to include MOSAIC.log

            plt.tight_layout()
            fig.canvas.draw_idle()
            plt.pause(0.1)  # Pause to update the chart

    # Setup figure and subplots for dynamic updating
    plt.ion()  # Turn on interactive mode
    num_plots = len(files_to_monitor)
    fig, axs = plt.subplots(num_plots, 1, figsize=(20, 10))  # Ensure space for MOSAIC.log

    if num_plots == 1:  # Ensure axs is iterable for a single subplot
        axs = [axs]

    try:
        while True:
            time.sleep(2) #only plot every 2 sec
            update_plots(fig, axs, tool_activation_delay_since_last_cdasim_step)

    except KeyboardInterrupt:
        for observer in observers:
            observer.stop()
        for observer in observers:
            observer.join()

def backup_and_clear_sdsm_log(file_path):
    # Split the file path into directory, basename, and extension
    dir_name, base_name = os.path.split(file_path)
    name, ext = os.path.splitext(base_name)

    # Create a new file name by appending "_copy" before the extension
    new_file_name = f"_copy-{name}-{ext}"
    new_file_path = os.path.join(dir_name, new_file_name)

    # Use an incremental number if the copy already exists
    counter = 1
    while os.path.exists(new_file_path):
        new_file_name = f"_copy-{name}-{counter}{ext}"
        new_file_path = os.path.join(dir_name, new_file_name)
        counter += 1

    # Copy the file
    shutil.copyfile(file_path, new_file_path)

    with open(file_path, 'w'):
        pass


if __name__ == "__main__":

    sensor_data_sharing_service_file_base_path = '/home/carma/cdasim_config/sensor_data_sharing_service/logs/'

    # sensor_data_sharing_service creates one file each day and appends new logs to it
    # NOTE: by default when this log is created, it is under user root. It must be chown/chrp to carma to be able to be modified
    current_date = datetime.today().strftime('%Y-%m-%d')
    sensor_data_sharing_service_file_path = sensor_data_sharing_service_file_base_path + 'sensor_data_sharing_service_' + current_date

    # Save previous SDSM service log and start a new one
    #backup_and_clear_sdsm_log(sensor_data_sharing_service_file_path)

    # Base path to start searching for rosout.log
    rosout_base_path = '/opt/carma/logs/carma_1/'

    # Base path to start searching for simulation logs
    sim_base_path = '/opt/carma-simulation/logs/'

    # Base path to save v2xhub logs
    carma_streets_base_path = '/home/carma/vru-logs/v2xhub/'

    # Sleep so that the logs have enough time to start getting populated/created
    time.sleep(2)

    # List of other specific files to monitor, add full paths or just filenames if in the same directory
    # Graph only plots if all files get the anticipated pattern of string
    files_to_monitor = [
        #sensor_data_sharing_service_file_path,
        'rosout.log',  # just file name because its parent folder is dynamically created
        'v2xhub.log',
        'Traffic.log',
        ### ... Save MOSAIC for last
        'MOSAIC.log',
    ]

    start_monitoring(rosout_base_path, sim_base_path, carma_streets_base_path, files_to_monitor)