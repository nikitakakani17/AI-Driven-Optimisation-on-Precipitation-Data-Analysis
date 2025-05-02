# Exploratory Data Analysis (EDA) for Each Sheet in an Excel File with Bias Detection
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = "/Users/nikitakakani/Documents/Project/Dataset copy.xlsx"  
xls = pd.ExcelFile(file_path, engine="openpyxl")  # Load Excel file

# List of stations (sheet names)
stations = ["Mehrabad", "Geophysics", "Chitgar", "Shemiran"]

# Iterate through each sheet and perform EDA separately
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)
    df["Sheet"] = sheet_name  # Track sheet name

    print(f"\nPerforming EDA for Sheet: {sheet_name}")

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
        plt.title(f"Time Series of Precipitation for {sheet_name} (Before Data Processing)", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Precipitation (mm)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
    else:
        print(f"⚠️ No valid 'Date' column or station column found in {sheet_name}. Skipping time-series analysis.")

print("\nEDA Completed: All three plots generated.")
