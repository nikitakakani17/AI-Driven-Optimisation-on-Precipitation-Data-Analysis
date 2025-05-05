import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import keras_tuner as kt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber   

# Load Data
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/UpdatedDataset.xlsx"
sheet_names = ["Mehrabad", "Geophysics", "Shemiran", "Chitgar"]
performance_data = []

# Model builder
def model_builder(hp, input_shape):
    inputs = Input(shape=input_shape)

    # CNN layers
    hp_filters1 = hp.Int("filters1", min_value=32, max_value=128, step=32)
    hp_kernel1 = hp.Choice("kernel1", values=[3, 5])
    x = Conv1D(filters=hp_filters1, kernel_size=hp_kernel1, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)

    hp_filters2 = hp.Int("filters2", min_value=32, max_value=128, step=32)
    hp_kernel2 = hp.Choice("kernel2", values=[3, 5])
    x = Conv1D(filters=hp_filters2, kernel_size=hp_kernel2, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # LSTM
    hp_lstm_units = hp.Int("lstm_units", min_value=32, max_value=128, step=32)
    x = LSTM(units=hp_lstm_units, return_sequences=False)(x)

    # Dropout
    hp_dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
    x = Dropout(rate=hp_dropout)(x)

    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss=Huber(delta=1.0))

    return model

# Tune model for each station
for sheet in sheet_names:
    print(f"\n📊 Processing Station: {sheet}...")

    df = pd.read_excel(file_path, sheet_name=sheet)
    filtered_columns = [col for col in df.columns if col not in ["Date", "MBE", "MAE", "RMSE"]]
    df_filtered = df[filtered_columns].dropna()
    X = df_filtered.drop(columns=[sheet])
    y = df_filtered[sheet]

    # Normalize and reshape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Tuner setup
    tuner = kt.RandomSearch(
        lambda hp: model_builder(hp, input_shape=X_train.shape[1:]),
        objective="val_loss",
        max_trials=25,
        executions_per_trial=1,
        directory="kt_cnn_lstm_tuning",
        project_name=f"cnn_lstm_tuning_{sheet}"
    )

    # Run tuner (suppress logs)
    tuner.search(X_train, y_train, epochs=80, validation_split=0.2,
                 callbacks=[EarlyStopping(monitor="val_loss", patience=15)],
                 verbose=0)

    # Train best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.fit(X_train, y_train, epochs=80, batch_size=32,
                   validation_data=(X_test, y_test),
                   verbose=0)  

    # Evaluate
    loss = best_model.evaluate(X_test, y_test, verbose=0)
    y_pred = best_model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ {sheet} | New Test Loss (MSE): {loss:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}")

    performance_data.append([sheet, loss, rmse, mae, r2])

    # Optional: Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual Precipitation')
    plt.plot(y_pred, label='Predicted Precipitation')
    plt.title(f"Hybrid Model (CNN + LSTM): Precipitation Prediction - {sheet}")
    plt.xlabel("Time")
    plt.ylabel("Precipitation")
    plt.legend()
    plt.tight_layout()
    plt.show()

# DataFrame and Heatmap
performance_df = pd.DataFrame(performance_data, columns=["Station", "New Test MSE", "RMSE", "MAE", "R²"])
performance_df.set_index("Station", inplace=True)

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(performance_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label': 'Metric Value'})
plt.title("Hybrid (CNN+LSTM) Model Evaluation Metrics by Station (After Hyperparameter Tuning)")
plt.xlabel("Performance Metrics (mm)")
plt.ylabel("Gauge Station")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
