import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the Excel file with bias-corrected data
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/BiasCorrectedData.xlsx"  
xls = pd.ExcelFile(file_path)

# Satellite products to check (columns are named exactly the same as these)
satellite_products = ["TRMM", "CHIRPS", "PERSIANN", "GPM"]
stations = ["Mehrabad", "Geophysics", "Chitgar", "Shemiran"]  

# Create a writer object to save the corrected data to multiple sheets in an Excel file
with pd.ExcelWriter("/Users/nikitakakani/Documents/Project/Cleaned_Data/UpdatedDataset.xlsx") as writer:

    # Iterate through each sheet (station)
    for station in stations:
        # Load data for the current station
        df = xls.parse(station)

        # Define station column (same as the sheet name)
        station_column = station
        
        # Loop through each satellite product
        for satellite in satellite_products:
            # Construct the satellite column name (same as satellite name)
            satellite_column = satellite

            # Ensure the satellite column exists
            if satellite_column in df.columns:
                # Recalculate MBE, MAE, RMSE for each row and update the respective columns
                for i in range(len(df)):
                    # Drop NaNs from the current row (only station and satellite values)
                    if not np.isnan(df.loc[i, station_column]) and not np.isnan(df.loc[i, satellite_column]):
                        # Calculate MBE, MAE, and RMSE for each row
                        MBE = df.loc[i, satellite_column] - df.loc[i, station_column]  # MBE for this row
                        MAE = np.abs(df.loc[i, satellite_column] - df.loc[i, station_column])  # MAE for this row
                        RMSE = np.sqrt((df.loc[i, satellite_column] - df.loc[i, station_column])**2)  # RMSE for this row

                        # Round the calculated values to 2 decimal places
                        MBE = round(MBE, 2)
                        MAE = round(MAE, 2)
                        RMSE = round(RMSE, 2)

                        # Update the respective error columns with the new values
                        if satellite == "TRMM":
                            df.at[i, "MBE"] = MBE
                            df.at[i, "MAE"] = MAE
                            df.at[i, "RMSE"] = RMSE
                        elif satellite == "CHIRPS":
                            df.at[i, "MBE.1"] = MBE
                            df.at[i, "MAE.1"] = MAE
                            df.at[i, "RMSE.1"] = RMSE
                        elif satellite == "PERSIANN":
                            df.at[i, "MBE.2"] = MBE
                            df.at[i, "MAE.2"] = MAE
                            df.at[i, "RMSE.2"] = RMSE
                        elif satellite == "GPM":
                            df.at[i, "MBE.3"] = MBE
                            df.at[i, "MAE.3"] = MAE
                            df.at[i, "RMSE.3"] = RMSE

        # Save the updated data for each station in the new file
        df.to_excel(writer, sheet_name=station, index=False)

# Confirmation message
print("Bias corrected data with recalculated errors (MBE, MAE, RMSE) saved to 'Bias_Corrected_Recalculated.xlsx'.")
