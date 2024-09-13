import pandas as pd
import os

def convert_csv_to_tum(csv_folder, output_folder):
    # List all CSV files in the specified folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # Process each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder, csv_file)
        
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Convert the CSV to TUM format for EVO processing
        tum_df = df[['Time', 'Position_X', 'Position_Y', 'Position_Z', 
                     'Orientation_X', 'Orientation_Y', 'Orientation_Z', 'Orientation_W']]

        # Create output filename by replacing .csv with .tum
        tum_file_name = os.path.splitext(csv_file)[0] + '.tum'
        tum_file_path = os.path.join(output_folder, tum_file_name)

        # Save the new TUM formatted data to a file
        tum_df.to_csv(tum_file_path, sep=' ', header=False, index=False)

        print(f"Converted {csv_file} to TUM format and saved to: {tum_file_path}")

# Example usage:
csv_folder = '/home/dat/slict_ws/src/uwbreloc/evo/output'  # Replace with the path to your CSV files folder
output_folder = '/home/dat/slict_ws/src/uwbreloc/evo/output_evo'  # Replace with the path where you want to save the TUM files

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert all CSV files in the folder to TUM format
convert_csv_to_tum(csv_folder, output_folder)
