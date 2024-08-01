# Generate a box plot that benchmarks the route tracking performance of the red and blue C1T trucks at varying speeds


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import argparse, argcomplete
import os
import yaml
from plot_crosstrack_error import plot_crosstrack_error


def plot_multiple_crosstrack_errors(bags, show_plots=True):
    bag_metadata = []
    for bag in bags:
        bag_path = os.path.normpath(os.path.abspath(bag))
        if os.path.isfile(os.path.join(bag_path, "red_truck_params.yaml")):
            truck = "red"
            params_file = os.path.join(bag_path, "red_truck_params.yaml")
        elif os.path.isfile(os.path.join(bag_path, "blue_truck_params.yaml")):
            truck = "blue"
            params_file = os.path.join(bag_path, "blue_truck_params.yaml")
        else:
            raise ValueError("%s does not have a parameters file. Exiting..." % (bag_path,))
        with open(params_file, "r") as f:
            speed = yaml.load(f, Loader=yaml.SafeLoader)["controller_server"]["ros__parameters"]["FollowPath"]["desired_linear_vel"]
        bag_metadata.append((bag_path, truck, speed))
    red_truck_data, blue_truck_data = [], []
    red_truck_speeds, blue_truck_speeds = [], []
    for run in bag_metadata:
        _, crosstrack_errors = plot_crosstrack_error(run[0], show_plots=False)
        if run[1] == "red":
            red_truck_data.append(np.abs(crosstrack_errors))
            red_truck_speeds.append(run[2])
        else:
            blue_truck_data.append(np.abs(crosstrack_errors))
            blue_truck_speeds.append(run[2])

    red_truck_sort = np.argsort(red_truck_speeds)
    red_truck_sorted_data = []
    red_truck_sorted_speeds = []
    for i in red_truck_sort:
        red_truck_sorted_data.append(red_truck_data[i])
        red_truck_sorted_speeds.append(red_truck_speeds[i])

    blue_truck_sort = np.argsort(blue_truck_speeds)
    blue_truck_sorted_data = []
    blue_truck_sorted_speeds = []
    for i in blue_truck_sort:
        blue_truck_sorted_data.append(blue_truck_data[i])
        blue_truck_sorted_speeds.append(blue_truck_speeds[i])

    data = red_truck_sorted_data + blue_truck_sorted_data

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, medianprops = dict(color = "black", linewidth = 1.5))
    for i in range(len(data)):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        if i < len(red_truck_data):
            ax.add_patch(Polygon(box_coords, facecolor="darksalmon"))
        else:
            ax.add_patch(Polygon(box_coords, facecolor="cornflowerblue"))
    ax.set_xticklabels([str(speed) for speed in red_truck_sorted_speeds + blue_truck_sorted_speeds])
    fig.text(0.925, 0.85, 'Red Truck',
        backgroundcolor="darksalmon", color='black', weight='roman',
        size='x-small')
    fig.text(0.925, 0.815, 'Blue Truck',
        backgroundcolor="cornflowerblue",
        color='white', weight='roman', size='x-small')
    plt.xlabel("Vehicle Speed (m/s)")
    plt.ylabel("Crosstrack Error (m)")
    plt.title("Crosstrack Error at Varying Speeds")
    if show_plots:
        plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate box plots for multiple runs of C1T vehicles")
    parser.add_argument("bags", type=str, help="Directories of bags to load", nargs='*')
    parser.add_argument("--png_out", type=str, help="File path to save the plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_multiple_crosstrack_errors(argdict["bags"])
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
