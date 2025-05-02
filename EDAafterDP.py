import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/New.xlsx"  
xls = pd.ExcelFile(file_path, engine="openpyxl")  # Load Excel file

# List of stations (sheet names)
stations = ["Mehrabad", "Geophysics", "Chitgar", "Shemiran"]

# Iterate through each sheet and perform EDA separately
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)
    df["Sheet"] = sheet_name  # Track sheet name
    
    print(f"\nPerforming EDA after data processing for the: {sheet_name} sheet")
    
    # Convert 'Date' column to datetime if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
    # Select numeric columns for analysis
    numeric_df = df.select_dtypes(include=['number'])
    
    # Set visualization style
    sns.set_style("whitegrid")
    
    ### Handling Missing Data (Bar Chart with Proper Axis Sizing)**
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_data = missing_data[missing_data > 0]
    missing_percentage = missing_percentage[missing_percentage > 0]

    # Missing Data Visualization
    if not missing_data.empty:
        missing_df = pd.DataFrame({"Column": missing_percentage.index, "Missing %": missing_percentage.values})
        sns.barplot(data=missing_df, x="Column", y="Missing %", hue="Column", dodge=False)
        
        plt.xticks(rotation=45, fontsize=9, ha="right")
        plt.ylabel("Percentage of Missing Data (%)", fontsize=12)
        plt.xlabel("Columns with Missing Values", fontsize=12)
        plt.title(f"Missing Data Percentage ({sheet_name})", fontsize=14, fontweight="bold")
        plt.ylim(0, 5)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Add percentage labels
        for p in plt.gca().patches:
            plt.text(p.get_x() + p.get_width() / 2., p.get_height(), f"{p.get_height():.1f}%", 
                     ha='center', va='bottom', fontsize=9, color='black', fontweight="bold")
        plt.show()  # Show plot separately
    else:
        print(f"No missing data in {sheet_name}")
        
        
    ### Data Distribution Visualization (Histogram + KDE)**
    if not numeric_df.empty:
        plt.figure(figsize=(8, 5))
        for col in numeric_df.columns:
            sns.histplot(numeric_df[col], bins=50, kde=True, label=col, alpha=0.6)
        plt.xscale("log")  # Log scale for better visibility
        plt.xlabel("Values (Log Scale)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Distribution of Key Variables ({sheet_name})", fontsize=14, fontweight="bold")
        plt.legend(fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
    else:
        print(f"⚠️ No numeric columns available in {sheet_name} for distribution analysis.")
        

    ### 🔹 Time-Series Trend Analysis ###
    if "Date" in df.columns and sheet_name in stations:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="Date", y=sheet_name, data=df, label=sheet_name)
        plt.title(f"Time Series of Precipitation for {sheet_name} (After Data Processing)", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Precipitation (mm)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
    else:
        print(f"⚠️ No valid 'Date' column or station column found in {sheet_name}. Skipping time-series analysis.")


print("\nEDA Completed: All four plots generated in the same figure for each sheet (Including Bias Detection).")





# Calculate correlation matrix
#correlation_matrix = df.corr()

# Plot the correlation matrix as a heatmap
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
#plt.title("Correlation Matrix After Data Processing", fontsize=16)
#plt.show()

# Assuming 'Date' and precipitation values are the same, but from the cleaned dataset
#plt.figure(figsize=(12, 6))


# Loop through each station and plot separately
#for station, label in zip(stations, station_labels):
 #   plt.figure(figsize=(12, 6))
    
  #  plt.plot(df['Date'], df[station], marker="o", linestyle="-", label=label)
    
    # Adding titles and labels
   # plt.title(f"Time Series of Precipitation for {label} (Processed Data)", fontsize=16)
    #plt.xlabel("Date", fontsize=12)
  #  plt.ylabel("Precipitation (mm)", fontsize=12)
   # plt.legend()
   # plt.xticks(rotation=45)
   # plt.grid(True, linestyle="--", alpha=0.7)
   # plt.show()

# Correlation Matrix
#correlation_matrix = df[['Mehrabad_M', 'Geophysics_G', 'Chitgar_C', 'Shemiran_S']].corr()

# Plotting the correlation matrix as a heatmap
#plt.figure(figsize=(12, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
#plt.title("Correlation Matrix After Data Processing")
#plt.show()


# List of stations to plot
#stations = ['Mehrabad_M', 'Geophysics_G', 'Chitgar_C', 'Shemiran_S']
#station_labels = ['Mehrabad', 'Geophysics', 'Chitgar', 'Shemiran']
#satellite_products = ["GPM", "TRMM", "PERSIANN", "CHIRPS"]

### Re-visualizing trends after data processing (line plot for each station)
#for station, label in zip(stations, station_labels):
 #   plt.figure(figsize=(12, 6))
  #  sns.lineplot(x='Date', y=station, data=df, label=label)
   # plt.title(f"Time Series of Precipitation for {label} (After Data Processing)")
#    plt.xlabel("Date")
 #   plt.ylabel("Precipitation (mm)")
  #  plt.legend()
   # plt.xticks(rotation=45)
#    plt.grid(True)
 #   plt.show()


#numeric_df = df.drop(columns=["Date"])

### Boxplot to detect outliers
#plt.figure(figsize=(12, 6))
#sns.boxplot(data=numeric_df)
#plt.xticks(rotation=45)
#plt.title("Boxplot of Precipitation Data (Outlier Detection)")
#plt.show()

# Heatmap to visualize correlation
# Create a grid of scatter plots
#fig, axes = plt.subplots(nrows=len(stations), ncols=len(satellite_products), figsize=(14, 12))
#fig.suptitle("Satellite vs. Gauge Precipitation Comparison", fontsize=16)

# Loop through each station and satellite product
#for i, station in enumerate(stations):
 #   for j, satellite in enumerate(satellite_products):
  #      satellite_column = f"{satellite}_{station[0]}"  # Construct column name dynamically
   #     if satellite_column in df.columns and station in df.columns:
    #        ax = axes[i, j]
     #     ax.set_ylabel(f"{station} (Gauge)")
      #      ax.set_title(f"{satellite} vs. {station}")
       # else:
        #    axes[i, j].set_visible(False)  # Hide plot if column is missing

#plt.tight_layout(rect=[0, 0.90, 1, 0.96])  # Adjust layout to fit the main title
#plt.show()
