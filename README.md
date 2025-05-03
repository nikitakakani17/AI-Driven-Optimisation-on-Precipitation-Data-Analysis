# AI-Driven-Optimisation-on-Precipitation-Data-Analysis

# Overview:
This repository provides steps to create an algorithum for forcasting rainfall estimation on four different gauge station in Terhan and four satellite-based product. 

This algorithum creates using multiple .python scripts (Including step-by-step process to improve accuracy of precipitation data.

# System Specification:
-	System: MacBook Air
-	M2 Chip / 8GB
-	OS: Ventura 13.4
- Python: 3.10

# Setup Instructions:
1. Clone Repository of this project
2. pip install -r requirements.txt
3. Verify Dataset Ensure that the Cleaned_Data/ folder contains:
   - Dataset.xlsx
  
# How to run the code step-by-step phases:
1. Exploratory Data Analysis (Before Data Processing)
   - Script: **EDAbeforeDP.py**
   - The purpose is to identify missing data patterns, precipitation distribution trends, and time-series variations in satellite estimations when compared to in situ ground measurements. (Gives you the plots)

2. Data Processing
   - Script: **DataProcessing.py**
   - The purpose is to standardise the dataset by addressing missing values, fixing inconsistencies, and ensuring that all data sources are synchronised daily.

3. Exploratory Data Analysis (Post Processing)
   - Script: **EDAafterDP.py**
   - The purpose is to re-verify the dataset to ensure that missing values are handled, timestamps are aligned, and numerical data is organised correctly.
  
4. Systematic Bias Detection
   - Script: **SystematicBiasDetection.py**
   - The goal of systematic bias detection is to quantify these differences using error metrics (MAE, MBE, RSME). (This will gives you the BiasDetection.cvs with the detected bias table for all the stations)

5. Systematic Bias Correction
   - Script: **SystematicBiasCorrection.py / RevisulizeBiasToEnsureDataQuality.py**
   - The goal is to correct the bias in the dataset using the formula (Corrected Satellite Data = Raw Satellite Data - MBE) for all the stations. (It will gives you the Bias Corrected datasets)
   - Then the second script is to revisualize the bias to make that it corrects now in the dataset. 
  
6. Recalculate the error metrics
   - Script: **RecalculateErrorMetrics.py**
   - Based on the corrected bias in gauge station and satellite data, this script correct the correspond errors in the dataset and gives you the plots for mean of errors.
  
7. Feature Selection
   - Script: **CorrelationAnalysis.py**
   - This section explores various techniques (Correlation Analysis, Mutual Information Method, Recursive Feature Elimination Method and Lasso Regression Method) to select most relevant features from the datasets.
  
8. Implementation of Baseline Models
   - Script: **CorrelationAnalysis.py**
   - This script applies the four machine learning models (Gradient Boosting Regressor, Random Forest Regressor, Decision Tree Regressor, and Linear Regression) and evaluate them using error metrics.

9. Implementation of Advanced AI Models
    - Scrips:
        - Neural Networks Implementation:
            - LSTM (Long Sort-Term Memory) Model: **LSTM.py**
            - CNN (Convolutional Neural Network) Model: **CNN.py**
        - Hybrid Model
            - CNN+LSTM Model: **Hybrid Model(CNN+LSTM).py**
    - This script implemented CNN, LSTM and Hybrid models to improve rainfall estimation and evaluate them using error metrics.

10. Hyperparameter Tuning
    - Scripts:
        - **TunedCNNModel.py**
        - **TunedLSTMModel.py**
        - **TunedHybridModel.py**
        - **CorrelationAnalysis.py** (After baseline models implementation, there will be continuous code for GBR Model with applied hyperparameter tuning)
    - This script applied the hyperparameter tuning for all four models (CNN, LSTM, Hybrid Model and Gradient Boosting Regressor) and evaluated them using error metrics. Along with the scrip explaination, report will descrbe the comparative result before and after tuning the each model.
    - Hyperparameter tuning helps to optimze the accuracy by setting the parameters and find the optimal balance that gives the high accuracy with lowest errors (MAE and RMSE).

Outputs:
- Plots will be generated after each script running except those who applied to correct the biases.


 
