import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform
import seaborn as sns

# Load dataset
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/UpdatedDataset.xlsx"
sheet_names = ["Mehrabad", "Geophysics", "Shemiran", "Chitgar"]

# Hybrid Model - CNN + LSTM
def create_hybrid_model(input_shape):
    model = Sequential()
    
    # CNN layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    
    # LSTM (no Flatten!)
    model.add(LSTM(units=50, return_sequences=False))
    
    # Output
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Store performance data
performance_data = []

# Loop through each sheet (station)
for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)
    
    # Data preprocessing
    filtered_columns = [col for col in df.columns if col not in ["Date", "MBE", "MAE", "RMSE"]]
    df_filtered = df[filtered_columns]
    
    X = df_filtered.drop(columns=[sheet])
    y = df_filtered[sheet]
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Create the model
    model = create_hybrid_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE) for {sheet}: {loss}")

    # Make predictions
    y_pred = model.predict(X_test)

    y_true = y_test.to_numpy()

    # Error metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)


    print(f"📈 RMSE : {rmse:.2f}")
    print(f"📈 MAE  : {mae:.2f}")
    print(f"📊 R²   : {r2:.2f}")

    # Store results
    performance_data.append([sheet, loss, rmse, mae, r2])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Precipitation")
    plt.plot(y_pred, label="Predicted Precipitation")
    plt.title(f"Hybrid Model (CNN + LSTM): Precipitation Prediction - {sheet}")
    plt.xlabel("Time")
    plt.ylabel("Precipitation")
    plt.legend()
    plt.show()

# Create DataFrame
performance_df = pd.DataFrame(performance_data, columns=["Station", "Test MSE", "RMSE", "MAE", "R²"])
performance_df.set_index("Station", inplace=True)

metrics_df = performance_df[["Test MSE", "RMSE", "MAE", "R²"]].astype(float)

plt.figure(figsize=(10, 6))
sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label': 'Metric Value'})
plt.title("Hybrid (CNN+LSTM) Model Evaluation Metrics by Station ")
plt.xlabel("Performance Metrics (mm)")
plt.ylabel("Gauge Station")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()