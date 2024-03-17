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
import subprocess
import threading
import docker
import psutil

#  global values for plotting and other tracking
IS_SYNCED_PLOT_NAME = 'Is Synced?'
delays = {}


class LogMonitor:
    def __init__(self, files):
        # Pattern to match "Simulation Time: X where current system time is: Y"
        self.log_pattern = re.compile(r'Simulation Time: (\d+) where current system time is: (\d+)')
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
        global delays
        global IS_SYNCED_PLOT_NAME

        #print(f"Processing file: {os.path.basename(file_path)}")
        match = self.log_pattern.search(line)
        if match:
            simulation_time = int(match.group(1))
            system_time = int(match.group(2))

            if simulation_time in self.simulation_times:
                # Already have an entry for this simulation time
                # guards against repeated operation if for some reason, simulation_time and file_name is printed in the log
                cached_list_time_and_file = self.simulation_times[simulation_time]

                for _, prev_file in cached_list_time_and_file:
                    if prev_file in file_path:
                        return

                cached_list_time_and_file.append((system_time, file_path))
                self.simulation_times[simulation_time] = cached_list_time_and_file

            else:
                # New simulation time, store along with its file path
                self.simulation_times[simulation_time] = [(system_time, file_path)]

            try:
                # if gathered all T2 of simulation time compare with their T1:
                if (len(self.simulation_times[simulation_time]) == self.file_size and len(self.simulation_times[simulation_time - 100]) == self.file_size):
                    #print(f"simulation_time: {simulation_time} and system time: {system_time}")

                    # try here because guaranteed to encounter keyerror if there is no T1 (a.k.a T2 = 0)
                    T1 = simulation_time - 100
                    T2 = simulation_time

                    # get max T1 and min T2
                    min_T2_system_time, _ = min(self.simulation_times[T2], key=lambda x: x[0])
                    max_T1_system_time, _ = max(self.simulation_times[T1], key=lambda x: x[0])
                    min_T1_system_time, _ = min(self.simulation_times[T1], key=lambda x: x[0])

                    # if minimum T2 is bigger than max T1, then it means *highly likely* that all components only progressed
                    # after every component progressed
                    is_synced = min_T2_system_time > max_T1_system_time - 10 #error

                    print(f"Did T2 progress healthy? T2: {T2} where system_time difference (t2 - t1) is dt: {(min_T2_system_time - max_T1_system_time):.0f} ======================: is synced?: {is_synced}")
                    print(f"is T1 sent around same time? T1: {T1} where system_time difference (t1_max - t1_min) is dt: {(max_T1_system_time - min_T1_system_time):.0f} ")

                    # For plotting
                    for (T1_tool_sys_time, cached_file_path) in self.simulation_times[T1]:
                        if ('MOSAIC' in cached_file_path):
                            delays[cached_file_path].append(min_T2_system_time - T1_tool_sys_time)
                        else:
                            delays[cached_file_path].append(T1_tool_sys_time - min_T1_system_time)

                    delays[IS_SYNCED_PLOT_NAME].append(is_synced)

                    #already checked this simulation time, so del to save space
                    del self.simulation_times[T1]

            except KeyError:
                #print(f"Key error at: {simulation_time} and file_path: {file_path}")
                return

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

def find_most_recent_file(search_path, pattern):
    list_of_files = glob.glob(os.path.join(search_path, '**', pattern), recursive=True)
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def monitor_file(file_path, monitor):
    directory = os.path.dirname(file_path)
    event_handler = FileEventHandler(monitor, file_path)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)  # Monitoring the directory of the file
    observer.start()
    return observer


def start_monitoring(rosout_base_path, sim_base_path, files_to_monitor):
    global delays
    delays[IS_SYNCED_PLOT_NAME] = []
    log_monitor = LogMonitor(files_to_monitor)

    observers = []

    for file_name in files_to_monitor:
        if 'rosout.log' in file_name:
            file_path = find_most_recent_file(rosout_base_path, 'rosout.log')
            if not file_path:
                print("No recent rosout.log file found.")
                continue
        elif 'Infrastructure.log' in file_name or 'Traffic.log' in file_name or 'MOSAIC.log' in file_name :
            file_path = find_most_recent_file(sim_base_path, file_name)
            if not file_path:
                print(f"No recent {file_name} file found.")
                continue
        elif 'v2xhub.log' in file_name:
            # Create a Thread to run the logging in the background
            log_thread = threading.Thread(target=log_monitor.capture_docker_container_log, args=('/home/carma/v2xhub.log', 'v2xhub'))
            print(f"Detected file to monitor: {file_name}")
            delays[file_name] = []
            # Start the thread
            log_thread.start()
            continue
        else:
            file_path = file_name

        print(f"Detected file to monitor: {file_path}")
        delays[file_path] = []
        observer = monitor_file(file_path, log_monitor)
        observers.append(observer)

    # Function to update plots for each file
    def update_plots(fig, axs, delays):
        mosaic_data = None  # Placeholder for MOSAIC.log data

        # Check if MOSAIC.log data is present and remove it from the regular plotting loop
        for file_name in delays.keys():
            if 'MOSAIC.log' in file_name:
                mosaic_data = delays[file_name]
                break

        for ax, (file_name, delay_list) in zip(axs, delays.items()):
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

        # Overlay MOSAIC.log data on each subplot
        if mosaic_data is not None:
            iterations = list(range(1, len(mosaic_data) + 1))
            for ax in axs:
                # You might want to differentiate the MOSAIC.log plot, e.g., with a different color or linestyle
                ax.plot(iterations, mosaic_data, label='MOSAIC.log', linestyle='--', color='red')
                ax.legend()  # Update the legend to include MOSAIC.log

            plt.tight_layout()  # Adjust layout to not overlap
            fig.canvas.draw_idle()  # Redraw the canvas
            plt.pause(0.1)  # Pause to update the chart

        print(f"CPU utilization: {psutil.cpu_percent()}%")

    # Setup figure and subplots for dynamic updating
    plt.ion()  # Turn on interactive mode
    num_plots = len(files_to_monitor)
    fig, axs = plt.subplots(num_plots, 1, figsize=(20, 10))  # Ensure space for MOSAIC.log

    if num_plots == 1:  # Ensure axs is iterable for a single subplot
        axs = [axs]

    try:

        while True:
            time.sleep(2)
            update_plots(fig, axs, delays)

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

def capture_docker_container_log_OLD(file_path, container_name):

    with open(file_path, 'w') as f:
    # Start the process
        with subprocess.Popen(["docker", "logs", "-f", container_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True) as proc:
            # Check if the process has a valid stdout
            if proc.stdout:
                # Iterate through each line of the output
                for line in proc.stdout:
                    # Write to the file and flush
                    f.write(line)
                    f.flush()

if __name__ == "__main__":

    current_date = datetime.today().strftime('%Y-%m-%d')
    sensor_data_sharing_service_file_path = '/home/carma/cdasim_config/sensor_data_sharing_service/logs/sensor_data_sharing_service_' + current_date

    # So that we don't forget
    backup_and_clear_sdsm_log(sensor_data_sharing_service_file_path)

    # Base path to start searching for rosout.log
    rosout_base_path = '/opt/carma/logs/carma_1/'

    # Base path to start searching for simulation logs
    sim_base_path = '/opt/carma-simulation/logs/'

    time.sleep(2)

    # List of other specific files to monitor, add full paths or just filenames if in the same directory
    files_to_monitor = [
        #sensor_data_sharing_service_file_path,
        'rosout.log',  # This will be replaced by the path to the most recent rosout.log found
        '/home/carma/v2xhub.log',
        'MOSAIC.log',
        'Traffic.log',
    ]

    start_monitoring(rosout_base_path, sim_base_path, files_to_monitor)