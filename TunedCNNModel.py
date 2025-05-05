import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from scipy.stats import uniform
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# Load dataset
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/UpdatedDataset.xlsx"
sheet_names = ["Mehrabad", "Geophysics", "Shemiran", "Chitgar"]

# Function to build CNN model
def build_cnn(filters=64, kernel_size=3, pool_size=2, learning_rate=0.001, input_shape=None):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error", metrics=["mse"])
    return model

# Store all results
results = {}

# Hyperparameter tuning loop
for sheet in sheet_names:
    print(f"\n🔧 Tuning CNN for {sheet}...")
    df = pd.read_excel(file_path, sheet_name=sheet)
    filtered_columns = [col for col in df.columns if col not in ["Date", "MBE", "MAE", "RMSE"]]
    df_filtered = df[filtered_columns].dropna()

    X = df_filtered.drop(columns=[sheet])
    y = df_filtered[sheet]

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # KerasRegressor model wrapper
    model = KerasRegressor(
        model=build_cnn,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        verbose=0,
        metrics=["mse"]
    )

    # Search space
    param_dist = {
        "model__filters": [32, 64],
        "model__kernel_size": [2, 3],
        "model__pool_size": [1, 2],
        "optimizer__learning_rate": uniform(0.001, 0.01),
        "batch_size": [32],
        "epochs": [50, 80]
    }

    # Random search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_cv_mse = -random_search.best_score_

    print(f"✅ Best Parameters for {sheet}: {best_params}")
    print(f"📉 Best CV MSE: {best_cv_mse:.2f}")

    # Train final model
    final_model = build_cnn(
        filters=best_params['model__filters'],
        kernel_size=best_params['model__kernel_size'],
        pool_size=best_params['model__pool_size'],
        learning_rate=best_params['optimizer__learning_rate'],
        input_shape=(X_train.shape[1], 1)
    )

    final_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)
    test_loss = final_model.evaluate(X_test, y_test, verbose=0)

    # If metrics=["mse"], test_loss = [loss, mse]
    test_loss_value = test_loss[0] if isinstance(test_loss, list) else test_loss

    # Make predictions
    y_pred = final_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"🧪 New Test MSE: {test_loss_value:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}")

    # Store results
    results[sheet] = {
        "Best Parameters": best_params,
        "Best CV MSE": best_cv_mse,
        "New Test MSE": test_loss_value,
        "RMSE": rmse,
        "MAE": mae,
        "R² Score": r2
    }

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual Precipitation")
    plt.plot(y_pred, label="Predicted Precipitation")
    plt.title(f"CNN Prediction After Tuning - {sheet}")
    plt.xlabel("Time")
    plt.ylabel("Precipitation")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Summary table
result_df = pd.DataFrame(results).T
print("\n📊 Summary of CNN Results Across Stations:")
print(result_df)

# Only keep numeric metrics for heatmap
metrics_df = result_df[["New Test MSE", "RMSE", "MAE", "R² Score"]].astype(float)

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label': 'Metric Value'})
plt.title("CNN Model Evaluation Metrics by Station (After Hyperparameter Tuning)")
plt.xlabel("Performance Metrics (mm)")
plt.ylabel("Gauge Stations")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

