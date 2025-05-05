import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Define LSTM model ====
def create_lstm_model(units=50, learning_rate=0.001, input_shape=None):
    model = Sequential()
    model.add(Input(shape=input_shape))  
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# ==== Load data ====
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/UpdatedDataset.xlsx"
sheet_names = ["Mehrabad", "Geophysics", "Shemiran", "Chitgar"]
performance_data = []
results = {}


for sheet in sheet_names:
    print(f"\n🔧 Tuning LSTM for: {sheet}")
    df = pd.read_excel(file_path, sheet_name=sheet)

    # Preprocessing
    filtered_columns = [col for col in df.columns if col not in ["Date", "MBE", "MAE", "RMSE"]]
    df_filtered = df[filtered_columns]
    X = df_filtered.drop(columns=[sheet])
    y = df_filtered[sheet]

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model wrapper
    model = KerasRegressor(
        model=create_lstm_model,
        verbose=0,
        epochs=50,
        batch_size=32,
        model__input_shape=(X_train.shape[1], X_train.shape[2])  
    )

    # Parameter grid with proper prefixes
    param_dist = {
        "model__units": [50, 100, 200],
        "model__learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64]
    }

    # Random search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    
    random_search.fit(X_train, y_train)

    # Evaluate
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_loss = mean_squared_error(y_test, y_pred)
    print(f"✅ New Test MSE for {sheet} after hyperparameter tuning: {test_loss:.2f}")
    
    results[sheet] = test_loss

    # Error metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    print(f"\n📈 RMSE : {rmse:.2f}"
          f"📈 MAE  : {mae:.2f}"
          f"\n📊 R²   : {r2:.2f}" )
    # Store performance data
    performance_data.append([sheet, test_loss, rmse, mae, r2])    

    # Plotting
   # plt.figure(figsize=(10, 6))
  #  plt.plot(y_test, label='Actual Precipitation')
 #   plt.plot(y_pred, label='Predicted Precipitation')
    #plt.title(f"LSTM Model: Precipitation Prediction for {sheet}")
   # plt.xlabel('Time')
  #  plt.ylabel('Precipitation')
 #   plt.legend()
#    plt.show()

# Create DataFrame for heatmap
performance_df = pd.DataFrame(performance_data, columns=["Station", "New Test MSE", "RMSE", "MAE", "R2"])
performance_df.set_index("Station", inplace=True)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(performance_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label': 'Metric Value'})
plt.title("LSTM Evaluation Metrics by station (After Hyperparameter Tuning)")
plt.xlabel("Performance Metrics (mm)")
plt.ylabel("Gauge Stations")
plt.yticks(rotation=0)
plt.show()