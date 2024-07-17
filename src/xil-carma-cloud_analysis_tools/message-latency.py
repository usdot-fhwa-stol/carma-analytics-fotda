import pandas as pd
from datetime import datetime


def tcr_tcm_list(input_file, output_file):
    # Initialize an empty list to store rows containing TCR's
    found_rows = []  
    count = 0

    # Read into a pandas DataFrame
    df = pd.read_csv(input_file, sep='delimiter')
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Check if "TrafficControlRequest" is in any cell of the current row
        if 'TrafficControlRequest' in str(row.values):
            tcr_id_value = str(row.values)[str(row.values).find('<reqid>')+7:str(row.values).find('<reqseq>')-8]    #Grabs the tcr id for this row

            if 'Sent TCR to cloud' in str(row.values):  #checks if file is cloud
                tcr_time_value = str(row.values)[14:26] #grabs time stanp of cloud message
            else:    
                tcr_time_value = str(row.values)[str(row.values).find('DEBUG')+6:str(row.values).find('[TcmReqServlet]')-1] #grabs timestamp of v2xhub message
            
            found_rows.append([tcr_id_value, tcr_time_value])
            count = 0
    
    found_df = pd.DataFrame(found_rows)
    found_df.columns = ["tcr_id_value", "tcr_time_stamp"]

    # Write the filtered DataFrame to a new Excel file
    found_df.to_csv(output_file, index=False)
    print("Analysis File Created")

def latency_calculation(input_cloud_file, input_infrastructure_file, output_combined):

    # Read in the two csv files using pandas 
    df_tx = pd.read_csv(input_infrastructure_file, header=0)
    df_rx = pd.read_csv(input_cloud_file, header=0)
    
    # Merge both dataframes on the "tcr_id_value" column
    combined_df = df_tx.merge(df_rx, how='left', on='tcr_id_value', suffixes=('_tx', '_rx'))
    
    # Calculate latency and write to column
    combined_df['latency (s)'] = (pd.to_datetime(combined_df['tcr_time_stamp_rx']) - pd.to_datetime(combined_df['tcr_time_stamp_tx'])).dt.total_seconds()
     
    # Write the filtered DataFrame to a new Excel file
    combined_df.to_csv(output_combined, index=False)
    print(combined_df)
    print("File Created")

def overall_metrics(input_latency_file):
    df = pd.read_csv(input_latency_file, header=0)
    print("**************************************************************************")
    print("The total number of packets transfered was : " + str(len(df.index)))
    print("The average latency across all messages was : " + str(df['latency (s)'].mean(skipna=True)))
    
def main():
    # Create csv with only TCR info from cloud logs
    input_cloud_file = '/home/test/Downloads/xil-testing/data/carmacloud.log'  # Replace with your input Excel file path
    output_cloud_file = 'tcr_filtered_cloud.csv'  # Replace with your desired output Excel file path
    tcr_tcm_list(input_cloud_file, output_cloud_file)

    # Create csv with only TCR info from v2xhub logs
    input_infrastructure_file = '/home/test/Downloads/xil-testing/data/v2xhub.log'  # Replace with your input Excel file path
    output_infrastructure_file = 'tcr_filtered_v2xhub.csv'  # Replace with your desired output Excel file path
    tcr_tcm_list(input_infrastructure_file, output_infrastructure_file)

    # Create csv for the matched/combined tcr from cloud and v2xhub
    output_combined_file = 'latency-combined.csv'  # Replace with your desired output Excel file path
    latency_calculation(output_cloud_file, output_infrastructure_file, output_combined_file)

    # Print metrics
    overall_metrics(output_combined_file)

if __name__ == '__main__':
    main()

