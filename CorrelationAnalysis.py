import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# Load dataset (processed data after bias correction)
file_path = "/Users/nikitakakani/Documents/Project/Cleaned_Data/UpdatedDataset.xlsx"
sheet_names = ["Mehrabad", "Geophysics", "Shemiran", "Chitgar"]

# Dictionary to store data
correlation_data = {}

### Feature Selection:

# 1. Correlation Analysis
for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)

    # Remove "Date" and Error Metrics (MBE, MAE, RMSE) – Keep only Gauge & Satellite Data
    filtered_columns = [col for col in df.columns if col not in ["Date"] and "MBE" not in col and "MAE" not in col and "RMSE" not in col]
    df_filtered = df[filtered_columns]

    # Compute correlation matrix
    corr_matrix = df_filtered.corr()
    
    # Extract correlation values of satellite products with the gauge station
    if sheet in corr_matrix.index:
        correlation_data[sheet] = corr_matrix.loc[sheet, ["TRMM", "CHIRPS", "PERSIANN", "GPM"]].values

# Convert dictionary to DataFrame for visualization
correlation_df = pd.DataFrame.from_dict(correlation_data, orient="index", columns=["TRMM", "CHIRPS", "PERSIANN", "GPM"])

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xlabel("Satellite Products")
plt.ylabel("Gauge Stations")
plt.title("Correlation Matrix (Gauge Stations vs. Satellite Products)")
plt.show()

# 2. Mutual Information feature selection
def feature_selection_mutual_info(df_sheet, target_column):
    X = df_sheet.drop(columns=[target_column])
    y = df_sheet[target_column]
    
    # Compute mutual information for each feature
    mi = mutual_info_regression(X, y)
    
    # Create a DataFrame to display mutual information for each feature
    mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Information": mi})
    mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)
    
    print(f"Mutual Information Selected Features for {target_column}:")
    print(mi_df)
    print("\n")
    
    return mi_df

# 3. Recursive Feature Elimination (RFE)
def feature_selection_rfe(df_sheet, target_column):
    X = df_sheet.drop(columns=[target_column])
    y = df_sheet[target_column]
    
    # Initialize a model (Linear Regression)
    model = LinearRegression()
    
    # Apply RFE with the model to select all features (since we have only 4 features)
    rfe = RFE(model, n_features_to_select=X.shape[1]) 
    rfe.fit(X, y)
    
    # Get the selected features
    selected_features_rfe = X.columns[rfe.support_]
    
    print(f"RFE Selected Features for {target_column}:")
    print(selected_features_rfe)
    print("\n")
    
    return selected_features_rfe

# 3. Lasso Regression (L1 Regularization)
def feature_selection_lasso(df_sheet, target_column):
    X = df_sheet.drop(columns=[target_column])
    y = df_sheet[target_column]
    
    # Apply Lasso Regression with a regularization strength (alpha)
    lasso = Lasso(alpha=0.01)
    lasso.fit(X, y)
    
    # Get non-zero coefficients (selected features)
    selected_features_lasso = X.columns[lasso.coef_ != 0]
    
    print(f"Lasso Regression Selected Features for {target_column}:")
    print(selected_features_lasso)
    print("\n")
    
    return selected_features_lasso

# 4. Combine the Results
def combine_feature_selection_results(mi_results, rfe_results, lasso_results):
    return set(mi_results['Feature'].head(5)) & set(rfe_results) & set(lasso_results)

# Apply Feature Selection for each sheet
for sheet in sheet_names:
    # Load each sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet)
    
    # Remove "Date" and Error Metrics (MBE, MAE, RMSE) – Keep only Gauge & Satellite Data
    filtered_columns = [col for col in df.columns if col not in ["Date"] and "MBE" not in col and "MAE" not in col and "RMSE" not in col]
    df_filtered = df[filtered_columns]

    # Apply Feature Selection Techniques
    mi_results = feature_selection_mutual_info(df_filtered, sheet)
    rfe_results = feature_selection_rfe(df_filtered, sheet)
    lasso_results = feature_selection_lasso(df_filtered, sheet)

    # Combine the results from Mutual Information, RFE, and Lasso Regression
    final_selected_features = combine_feature_selection_results(mi_results, rfe_results, lasso_results)

    # Print the final selected features for the current sheet
    print(f"Final Selected Features for {sheet}: {final_selected_features}")
    
    # 5. Model Training

    # Prepare the data for model training using the final selected features
    X = df_filtered[list(final_selected_features)]
    y = df_filtered[sheet]  # The target variable (precipitation)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        "LR": LinearRegression(),
        "DTR": DecisionTreeRegressor(),
        "RFR ": RandomForestRegressor(),
        "GBR": GradientBoostingRegressor()
    }

    # Store model performance results
    performance = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        performance[model_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        }
        
    # Print performance for each model
    print(f"Model performance for {sheet}:")
    for model_name, metrics in performance.items():
        print(f"{model_name} - MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}, R²: {metrics['R²']}")
    print("\n")
    
    # Plot the performance comparison
    models = list(performance.keys())
    mae_scores = [performance[model]["MAE"] for model in models]
    rmse_scores = [performance[model]["RMSE"] for model in models]
    r2_scores = [performance[model]["R²"] for model in models]

    # Create subplots for the comparison
    plt.figure(figsize=(12, 5))

    # MAE Plot
    plt.subplot(1, 3, 1)
    bars = plt.bar(models, mae_scores, color=['#2980b9', '#f39c12', '#27ae60', '#8e44ad'], width=0.6, align='edge') 
    plt.title(f"Mean Absolute Error (MAE) - {sheet}")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')  # Display value at the top of the bar

    # RMSE Plot
    plt.subplot(1, 3, 2)
    bars = plt.bar(models, rmse_scores, color=['#2980b9', '#f39c12', '#27ae60', '#8e44ad'], width=0.6, align='edge') 
    plt.title(f"Root Mean Squared Error (RMSE) - {sheet}")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')  # Display value at the top of the bar

    # R² Plot
    plt.subplot(1, 3, 3)
    bars = plt.bar(models, r2_scores, color=['#2980b9', '#f39c12', '#27ae60', '#8e44ad'], width=0.6, align='edge') 
    plt.title(f"R² Score - {sheet}")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')  # Display value at the top of the bar

    plt.tight_layout()
    plt.show()
    
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting stages
    'learning_rate': [0.01, 0.05, 0.1],  # Shrinks contribution of each tree
    'max_depth': [3, 5, 7],  # Maximum depth of trees
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf
    'subsample': [0.8, 0.9, 1.0]  # Fraction of samples used for fitting trees
}

performance_data = []

# Loop over each sheet
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

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize the Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit GridSearchCV to find the best model
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters for {sheet} using Gradient Boosting Regressor: {best_params}")

    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Make predictions using the best model
    y_pred = best_model.predict(X_test)

    # Compute performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Print the model performance for each sheet
    print(f"Test performance for {sheet} using Gradient Boosting Regressor with tuned parameters:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

    performance_data.append([sheet, mae, rmse, r2])
    # Store the performance data
    performance_df = pd.DataFrame(performance_data, columns=["Station", "MAE", "RMSE", "R²"])
    performance_df.set_index("Station", inplace=True)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual Precipitation")
    plt.plot(y_pred, label="Predicted Precipitation")
    plt.title(f"Gradient Boosting Regressor: Precipitation Prediction - {sheet}")
    plt.xlabel("Time")
    plt.ylabel("Precipitation")
    plt.legend()
    plt.show()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(performance_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label': 'Metric Value'})
plt.title("GBR Model Evaluation Metrics After Hyperparameter Tuning")
plt.xlabel("Gauge Station")
plt.ylabel("Performance Metrics (mm)")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()





