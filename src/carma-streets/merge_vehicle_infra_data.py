import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Merge infrastructure and vehcile detected objects into one csv file
def merge_objects_from_logs(infrastructure_data, vehicle_data, merged_data):
    # Read the two CSV files
    df1 = pd.read_csv(infrastructure_data)
    df2 = pd.read_csv(vehicle_data)

    # Merge the two dataframes on 'timestamp'
    merged_df = pd.merge(df1, df2, on='Timestamp (ms)', how='left')

    # Fill missing values in 'x_y' (from file2) with 0
    merged_df['Positionx_y'] = merged_df['Positionx_y'].fillna(pd.NA) #TODO: check with the team what use for the missing data for a timestamp
    merged_df['Positiony_y'] = merged_df['Positiony_y'].fillna(pd.NA)


    # Rename the columns 
    merged_df.columns = ['Timestamp (ms)', 'Type_i', 'ObjID_i', 'Positionx_i', 'Positiony_i', 'ObjID_v', 'Positionx_v',  'Positiony_v']

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(merged_data, index=False)

# Plot x, y and x error and y error and save the images
def plot_merged_detected_objects(merged_detected_objects):
    # Read the merged CSV file
    merged_df = pd.read_csv(merged_detected_objects)

    # Group by 'Object ID' and plot 'x_file1' and 'x_file2' over 'timestamp'
    grouped = merged_df.groupby('ObjID_i')
    for name, group in grouped:
        plt.figure(figsize=(12, 6))
        plt.subplot(4, 1, 1)
        plt.plot(group['Timestamp (ms)'].to_numpy(), group['Positionx_i'].to_numpy(), label='x_infra')
        plt.plot(group['Timestamp (ms)'].to_numpy(), group['Positionx_v'].to_numpy(), label='x_vehicle')
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('X Value')
        plt.title(f'Object {name} - X Values Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(group['Timestamp (ms)'].to_numpy(), group['Positiony_i'].to_numpy(), label='y_infra')
        plt.plot(group['Timestamp (ms)'].to_numpy(), group['Positiony_v'].to_numpy(), label='y_vehicle')
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('Y Value')
        plt.title(f'Object {name} - Y Values Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(group['Timestamp (ms)'].to_numpy(), group['Positionx_i'].to_numpy() - group['Positionx_v'].to_numpy(), label='x_error')
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('X Error')
        plt.title(f'Object {name} - X Error Values Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(group['Timestamp (ms)'].to_numpy(), group['Positiony_i'].to_numpy() - group['Positiony_v'].to_numpy(), label='y_error')
        plt.xlabel('Timestamp (ms)')
        plt.ylabel('Y Error')
        plt.title(f'Object {name} - Y Error Values Over Time')
        plt.legend()
        plt.grid(True)

        # Save the plot in the 'plots' directory
        Path("plots").mkdir(exist_ok=True)
        plt.savefig(f'plots/object_{name}_plot.png')
        plt.close()

   

def main():
    parser = argparse.ArgumentParser(description='Script to Merge Detected Object Data frm Infrastructure and Vehicle')
    parser.add_argument('--infra-data', help='CSV file containing intfrastructure data.', type=str, required=True) 
    parser.add_argument('--vehicle-data', help='CSV file containing vehicle data.', type=str, required=True)
    parser.add_argument('--merged-data', help='CSV file containing merged vehicle and infrastructure data.', type=str, required=True) 
    args = parser.parse_args()

    merge_objects_from_logs(args.infra_data, args.vehicle_data, args.merged_data)
    plot_merged_detected_objects(args.merged_data)


if __name__ == '__main__':
    main()
