import pandas as pd
import numpy as np

# Load the bias comparison results from the CSV file
bias_results_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/BiasDetection.csv"  
bias_results_df = pd.read_csv(bias_results_path)

# Load the dataset for the bias correction (same dataset as before)
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/New.xlsx"  
xls = pd.ExcelFile(file_path)

# List of stations and satellite products
stations = ['Mehrabad', 'Geophysics', 'Chitgar', 'Shemiran']
satellite_products = ["TRMM", "CHIRPS", "PERSIANN", "GPM"]

# Create a writer object to save the corrected data to multiple sheets in an Excel file
with pd.ExcelWriter("/Users/nikitakakani/Documents/Project/Cleaned_Data/BiasCorrectedData.xlsx") as writer:

    # Loop through each station to apply bias correction
    for station in stations:

        df = xls.parse(station)

        # Loop through each satellite product to apply the bias correction
        for satellite in satellite_products:
            satellite_column = satellite
        
            # Check if the satellite column exists
            if satellite_column in df.columns:
                # Get the MBE value for the current station and satellite
                MBE_value = round(bias_results_df[(bias_results_df['Station'] == station) & 
                                  (bias_results_df['Satellite'] == satellite)]['MBE'].values[0], 2)

            
                # Apply bias correction: subtract the MBE from the satellite data
                df[satellite_column] = df[satellite_column] - MBE_value
                
                # Round the corrected satellite values to 2 decimal places
                df[satellite_column] = df[satellite_column].round(2)
            
                # Print the correction details for this satellite and station
                print(f"Applying Bias Correction for {satellite} at {station} using MBE: {MBE_value:.2f}")
                print(f"First 5 corrected values for {satellite}:")
                print(df[[station, satellite_column]].head())
                print("-" * 50)
        
        # Save the corrected data for the current station to the Excel file in separate sheet
        df.to_excel(writer, sheet_name=station, index=False)

# Message confirming that the corrected data is saved
print("Bias Corrected Data saved to 'BiasCorrectedData.xlsx' with separate sheets for each station.")