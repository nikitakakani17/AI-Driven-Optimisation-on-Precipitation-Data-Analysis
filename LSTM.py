import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load dataset
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/UpdatedDataset.xlsx"
sheet_names = ["Mehrabad", "Geophysics", "Shemiran", "Chitgar"]

# Dictionary to store the data
correlation_data = {}
performance_data = []

# Process each sheet (station) for LSTM
for sheet in sheet_names:
    # Load the data
    df = pd.read_excel(file_path, sheet_name=sheet)
    
    # Remove "Date" and Error Metrics (MBE, MAE, RMSE) – Keep only Gauge & Satellite Data
    filtered_columns = [col for col in df.columns if col not in ["Date", "MBE", "MAE", "RMSE"]]
    df_filtered = df[filtered_columns]

    # Select features and target variable
    X = df_filtered.drop(columns=[sheet])  # All columns except the target (precipitation)
    y = df_filtered[sheet]  # The target variable (precipitation)

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape X to be 3D for LSTM [samples, time steps, features]
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build LSTM Model
    model = Sequential()

    # Add LSTM layer (use more layers if needed)
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

    # Add Dense layer (output layer)
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE) for {sheet}: {loss}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    performance_data.append([sheet, loss, rmse, mae, r2])
    print(f"📈 Test MSE  : {loss:.2f}"
          f"\n📈 RMSE : {rmse:.2f}"
          f"\n📉 MAE  : {mae:.2f}"
          f"\n📊 R²   : {r2:.2f}"
          f"\n\n")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Precipitation")
    plt.plot(y_pred, label="Predicted Precipitation")
    plt.title(f"LSTM Model: Precipitation Prediction - {sheet}")
    plt.xlabel("Time")
    plt.ylabel("Precipitation")
    plt.legend()
    plt.show()

# Create DataFrame for heatmap
performance_df = pd.DataFrame(performance_data, columns=["Station", "Test MSE", "RMSE", "MAE", "R2"])
performance_df.set_index("Station", inplace=True)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(performance_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label': 'Metric Value'})
plt.title("LSTM Model Evaluation Metrics by Station")
plt.xlabel("Performance Metrics (mm)")
plt.ylabel("Gauge Stations")
plt.yticks(rotation=0)
plt.show()