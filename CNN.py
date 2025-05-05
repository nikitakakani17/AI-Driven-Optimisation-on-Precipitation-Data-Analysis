import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# Load dataset
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/UpdatedDataset.xlsx"
sheet_names = ["Mehrabad", "Geophysics", "Shemiran", "Chitgar"]
performance_data = []

# Process each sheet (station) for CNN
for sheet in sheet_names:
    # Load the data
    df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')

    # Remove "Date" and Error Metrics (MBE, MAE, RMSE) – Keep only Gauge & Satellite Data
    filtered_columns = [col for col in df.columns if col not in ["Date", "MBE", "MAE", "RMSE"]]
    df_filtered = df[filtered_columns]

    # Select features and target variable
    X = df_filtered.drop(columns=[sheet])  # All columns except the target (precipitation)
    y = df_filtered[sheet]  # The target variable (precipitation)

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape X to be 3D for CNN [samples, time steps, features]
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build CNN Model
    model = Sequential()

    # Add 1D Convolutional Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))

    # Add another 1D Convolutional Layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Flatten the output for Dense layers
    model.add(Flatten())

    # Add Dense layer (output layer)
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Loss for {sheet}: {loss}")

    # Make predictions
    y_pred = model.predict(X_test)

    performance_data.append({
        "Station": sheet,
        "Test MSE": loss,
        "MAE": mean_absolute_error(y_test, model.predict(X_test)),
        "RMSE": np.sqrt(mean_squared_error(y_test, model.predict(X_test))),
        "R²": r2_score(y_test, model.predict(X_test))
    })
    # Print performance metrics
    print(f"MAE for {sheet}: {performance_data[-1]['MAE']}")
    print(f"RMSE for {sheet}: {performance_data[-1]['RMSE']}")
    print(f"R² for {sheet}: {performance_data[-1]['R²']}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Precipitation")
    plt.plot(y_pred, label="Predicted Precipitation")
    plt.title(f"CNN Model: Precipitation Prediction - {sheet}")
    plt.xlabel("Time")
    plt.ylabel("Precipitation")
    plt.legend()
    plt.show()

# DataFrame and Heatmap
performance_df = pd.DataFrame(performance_data, columns=["Station", "Test MSE", "RMSE", "MAE", "R²"])
performance_df.set_index("Station", inplace=True)

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(performance_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label': 'Metric Value'})
plt.title("CNN Model Evaluation Metrics by Station")
plt.xlabel("Performance Metrics (mm)")
plt.ylabel("Gauge Stations")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

