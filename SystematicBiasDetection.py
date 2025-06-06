import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the Excel file (assumes all sheets have same structure)
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/New.xlsx" 
xls = pd.ExcelFile(file_path)

# Satellite products to check (columns are named exactly the same as these)
satellite_products = ["GPM", "TRMM", "PERSIANN", "CHIRPS"]
stations = ["Mehrabad", "Geophysics", "Chitgar", "Shemiran"] 

# Create an empty list to store the results
results_list = []

# Iterate through each sheet (station) and process
for sheet_name in stations:
    df = xls.parse(sheet_name)  # Load data for each station
    print(f"Processing data for sheet: {sheet_name}")

    station_column = sheet_name  # Define the station column name

    # Loop through each satellite product and check MBE, MAE, RMSE
    for satellite in satellite_products:
        satellite_column = satellite  

        if satellite_column in df.columns:
            df_cleaned = df.dropna(subset=[station_column, satellite_column]) # Drop rows where any NaN values exist

            # Check if there are any valid rows remaining after dropping NaNs
            if len(df_cleaned) > 0:
                MBE = df_cleaned[satellite_column].mean() - df_cleaned[station_column].mean()  # Mean Bias Error
                MAE = mean_absolute_error(df_cleaned[station_column], df_cleaned[satellite_column])  # Mean Absolute Error
                RMSE = np.sqrt(mean_squared_error(df_cleaned[station_column], df_cleaned[satellite_column]))  # Root Mean Squared Error

                # Print results for each satellite product
                print(f"Results for {satellite} in {sheet_name}:")
                print(f"  MBE: {MBE:.2f}")
                print(f"  MAE: {MAE:.2f}")
                print(f"  RMSE: {RMSE:.2f}")
                print("-" * 50)

                # Append the results to the list
                results_list.append({
                    "Station": sheet_name,
                    "Satellite": satellite,
                    "MBE": MBE,
                    "MAE": MAE,
                    "RMSE": RMSE
                })

                # Plot Bias Detection
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=df_cleaned["Date"], y=df_cleaned[station_column], label=f"{sheet_name} Ground Truth", color='blue')
                sns.lineplot(x=df_cleaned["Date"], y=df_cleaned[satellite_column], label=f"{satellite} Satellite", color='orange')
                plt.title(f"Bias Detection for {satellite} vs {sheet_name} ({MBE:.2f} MBE)")
                plt.xlabel("Date")
                plt.ylabel("Precipitation (mm)")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.show()
            else:
                print(f"Warning: No valid data available for {satellite} vs {sheet_name} after removing NaNs.")
        else:
            print(f"Warning: Column {satellite_column} not found for {satellite} in {sheet_name}")

# Convert the list of results into a DataFrame
results_df = pd.DataFrame(results_list)

# 1. Printing the results
print("Results of Bias Comparison:")
print(results_df)

# 2. Exporting the results to a CSV file for further analysis
results_df = results_df.round(2)

results_df.to_csv("/Users/nikitakakani/Documents/Project/Cleaned_Data/BiasDetection.csv", index=False)
print("Results exported to 'BiasDetection.csv'.")