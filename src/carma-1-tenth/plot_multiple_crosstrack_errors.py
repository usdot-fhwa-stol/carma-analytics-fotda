# Plot the crosstrack error as a function of downtrack (distance traveled along the route)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import argparse, argcomplete
import os
from plot_crosstrack_error import plot_crosstrack_error


def plot_multiple_crosstrack_errors(red_point_five_bag, red_point_seven_bag, red_one_bag, blue_point_five_bag, blue_point_seven_bag, blue_one_bag, show_plots=True):
    _, red_point_five_crosstrack_errors = plot_crosstrack_error(red_point_five_bag, show_plots=False)
    _, red_point_seven_crosstrack_errors = plot_crosstrack_error(red_point_seven_bag, show_plots=False)
    _, red_one_crosstrack_errors = plot_crosstrack_error(red_one_bag, show_plots=False)
    _, blue_point_five_crosstrack_errors = plot_crosstrack_error(blue_point_five_bag, show_plots=False)
    _, blue_point_seven_crosstrack_errors = plot_crosstrack_error(blue_point_seven_bag, show_plots=False)
    _, blue_one_crosstrack_errors = plot_crosstrack_error(blue_one_bag, show_plots=False)

    data = [np.sort(np.abs(red_point_five_crosstrack_errors)),
            np.sort(np.abs(red_point_seven_crosstrack_errors)),
            np.sort(np.abs(red_one_crosstrack_errors)),
            np.sort(np.abs(blue_point_five_crosstrack_errors)),
            np.sort(np.abs(blue_point_seven_crosstrack_errors)),
            np.sort(np.abs(blue_one_crosstrack_errors))]

    box_colors = ["darksalmon", "cornflowerblue"]
    if show_plots:
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
            # Alternate between Dark Khaki and Royal Blue
            ax.add_patch(Polygon(box_coords, facecolor=box_colors[i // 3]))
        ax.set_xticklabels(["0.5", "0.7", "1.0", "0.5", "0.7", "1.0"])
        fig.text(0.925, 0.85, 'Red Truck',
         backgroundcolor=box_colors[0], color='black', weight='roman',
         size='x-small')
        fig.text(0.925, 0.815, 'Blue Truck',
         backgroundcolor=box_colors[1],
         color='white', weight='roman', size='x-small')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate box plots for multiple runs of C1T vehicles")
    parser.add_argument("red_0.5_bag", type=str, help="Directory of bag to load for red truck traveling at 0.5 m/s")
    parser.add_argument("red_0.7_bag", type=str, help="Directory of bag to load for red truck traveling at 0.7 m/s")
    parser.add_argument("red_1.0_bag", type=str, help="Directory of bag to load for red truck traveling at 1.0 m/s")
    parser.add_argument("blue_0.5_bag", type=str, help="Directory of bag to load for blue truck traveling at 0.5 m/s")
    parser.add_argument("blue_0.7_bag", type=str, help="Directory of bag to load for blue truck traveling at 0.7 m/s")
    parser.add_argument("blue_1.0_bag", type=str, help="Directory of bag to load for blue truck traveling at 1.0 m/s")
    parser.add_argument("--png_out", type=str, help="File path to save the plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_multiple_crosstrack_errors(os.path.normpath(os.path.abspath(argdict["red_0.5_bag"])),
                                    os.path.normpath(os.path.abspath(argdict["red_0.7_bag"])),
                                    os.path.normpath(os.path.abspath(argdict["red_1.0_bag"])),
                                    os.path.normpath(os.path.abspath(argdict["blue_0.5_bag"])),
                                    os.path.normpath(os.path.abspath(argdict["blue_0.7_bag"])),
                                    os.path.normpath(os.path.abspath(argdict["blue_1.0_bag"])))
    plt.xlabel("Vehicle Speed (m/s)")
    plt.ylabel("Crosstrack Error (m)")
    plt.title("Crosstrack Error at Varying Speeds")
    if argdict["png_out"]:
        plt.savefig(argdict["png_out"])
    plt.show()
