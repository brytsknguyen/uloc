import pandas as pd
import numpy as np
import os

def replace_timestamps(data1, data2):
    matched_data = data2.copy()
    
    # Replace timestamps in data2 with timestamps from data1
    for idx, row in data1.iterrows():
        time1 = row[0]  # Get the timestamp from the first file
        
        # Find the index with the closest timestamp in data2
        closest_idx = (np.abs(data2[0] - time1)).idxmin()
        
        # Replace the timestamp in data2 with the one from data1
        matched_data.iloc[closest_idx, 0] = time1
    
    return matched_data

def process_two_tum_files(file1_path, file2_path, output_folder, output_file_name):
    # Read data from the two TUM files
    data1 = pd.read_csv(file1_path, delim_whitespace=True, header=None)
    data2 = pd.read_csv(file2_path, delim_whitespace=True, header=None)
    
    # Replace timestamps in data2 with those from data1
    updated_data2 = replace_timestamps(data1, data2)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the output file path
    output_file_path = os.path.join(output_folder, output_file_name)
    
    # Save the updated data2 to the output file
    updated_data2.to_csv(output_file_path, sep=' ', header=False, index=False)
    print(f"Processed data saved to: {output_file_path}")

def process_folders(folder1, folder2, output_folder):
    # Iterate through the files in folder1
    for file1_name in os.listdir(folder1):
        if file1_name.endswith('.tum'):
            # Find the corresponding file in folder2 (with '_packed' format)
            file_base_name = file1_name.replace('.tum', '')  # Get the base name of file1
            file2_name = file_base_name + '_packed.tum'
            
            file1_path = os.path.join(folder1, file1_name)
            file2_path = os.path.join(folder2, file2_name)
            
            if os.path.exists(file2_path):
                print(f"Processing: {file1_name} and {file2_name}")
                
                # Set the output file name based on file1
                output_file_name = f'processed_{file1_name}'
                
                # Process the two files and save the result
                process_two_tum_files(file1_path, file2_path, output_folder, output_file_name)
            else:
                print(f"File {file2_name} does not exist in {folder2}")

# Example usage:
folder1 = '/home/dat/slict_ws/src/uwbreloc/evo/output_evo'  # Replace with your path to folder 1
folder2 = '/home/dat/slict_ws/src/uwbreloc/evo/groundtruth_evo'  # Replace with your path to folder 2
output_folder = '/home/dat/slict_ws/src/uwbreloc/evo/groundtruth_evo_mod'  # Replace with your desired output folder path

# Process all files in the two folders
process_folders(folder1, folder2, output_folder)
