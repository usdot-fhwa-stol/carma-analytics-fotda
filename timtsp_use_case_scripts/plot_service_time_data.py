import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def get_rtfcsv_as_df(csv_source):
    try:
        df = pd.read_csv(
            csv_source,
            parse_dates=['system_time_dt'],
            dtype={ "sim_time_ns": int, "rtf": float},
        )
    except ValueError:
        print("plot_rtf_data: malformed csv data")
        sys.exit(1)
    return df

def get_service_time_csv_as_df(csv_source):
    try:
        df = pd.read_csv(
            csv_source,
            dtype={ "Real Time (ms) ": int, " Carma Time (ms)": int},
        )
        df.rename(columns={"Real Time (ms)": "real_time_ms", " Carma Time (ms)": "carma_time_ms"}, inplace=True)
    except ValueError:
        print("plot_rtf_data: malformed csv data")
        sys.exit(1)
    return df


def get_csv_files(directory):
    """
    Read through a directory and return a list of all CSV files.

    Args:
        directory (str): Path to the directory

    Returns:
        list: List of CSV filenames
    """
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return []

    # Use glob to find all .csv files in the directory
    csv_path = os.path.join(directory, "*.csv")
    csv_files = glob.glob(csv_path)

    # Extract just the filenames from the full paths
    csv_filenames = [os.path.basename(file) for file in csv_files]

    # Create dictionary to store the CSV files as df
    csv_dict = {}
    for file in csv_filenames:
        print("Adding", file, "to the list of CSV sources")
        # Add the file to the dictionary
        if "cdasim" in file.split("_"):
            csv_dict[file] = get_rtfcsv_as_df(os.path.join(directory, file))
        else:
            csv_dict[file] = get_service_time_csv_as_df(os.path.join(directory, file))

    return csv_dict


def synchronize_dfs(csv_dfs):
    # Convert all system_time and real_time columns to datetime
    max_reference_time = 0
    for key, df in csv_dfs.items():
        if "cdasim" in key.split("_"):
            df['system_time_ms'] = pd.to_datetime(df['system_time_dt'], unit='ns')
            reference_time_values_in_ms = df["sim_time_ns"].unique() /1e6
            base_df = df
        else:
            df['real_time_ms'] = pd.to_datetime(df['real_time_ms'], unit='ms')
            max_reference_time = max(df["carma_time_ms"].max(), max_reference_time)

    base_df['sim_time_ms'] = base_df['sim_time_ns'] / 1e6
    # Remove 0 sim time from the reference time values
    reference_time_values_in_ms = reference_time_values_in_ms[reference_time_values_in_ms <= max_reference_time]
    reference_time_values_in_ms = reference_time_values_in_ms[1:]
    base_df = base_df[base_df["sim_time_ms"] >= reference_time_values_in_ms.min()]
    base_df = base_df[base_df["sim_time_ms"] <= reference_time_values_in_ms.max()]
    columns_to_keep = ['system_time_ms', 'sim_time_ms']
    base_df_subset = base_df[columns_to_keep]


    merged_df_dict = {}
    # Merge base_df_subset with all other dataframes based on sim_time_ms
    for key, df in csv_dfs.items():
        if "cdasim" not in key.split("_"):
            df = df[df["carma_time_ms"].isin(reference_time_values_in_ms)]
            # Merge the dataframes
            # merged_df = pd.merge(base_df_subset, df, left_on='sim_time_ms', right_on='carma_time_ms', how='inner')
            merged_df = pd.merge(base_df_subset, df, left_on='sim_time_ms', right_on='carma_time_ms', how='right').dropna()
            # Find difference between system_time_ms and real_time_ms in milliseconds
            merged_df['real_time_diff'] = (merged_df['real_time_ms'].astype('int64') - merged_df['system_time_ms'].astype('int64')) / 1e6
            columns_to_keep = [ 'sim_time_ms', 'real_time_diff']
            merged_df = merged_df[columns_to_keep]
            csv_dfs[key] = merged_df
            print("Merged DataFrame for", key)
            print(merged_df)
            merged_df_dict[key] = merged_df
    return merged_df_dict

def plot_merged_data(merged_df_dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, merged_df in merged_df_dict.items():
        ax.plot(merged_df["sim_time_ms"], merged_df["real_time_diff"], label=key.split(".csv")[0].split("_")[1])
    ax.set_title("Real Time Difference vs Simulation Time")
    ax.set_ylabel("Real Time Difference [ms]")
    ax.set_xlabel("Simulation Time [s]")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()




def main():
    # Check if directory path is provided as argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    csv_dfs = get_csv_files(directory)

    merged_dfs = synchronize_dfs(csv_dfs)
    plot_merged_data(merged_dfs)

if __name__ == "__main__":
    main()