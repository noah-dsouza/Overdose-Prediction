"""
Drug Overdose Predictor:
Trains the Random Forest Regressor machine learning model 
to predict future drug-related deaths based on trends in the dataset
"""

import pandas as pd                    
from sklearn.model_selection import train_test_split   
from sklearn.ensemble import RandomForestRegressor     
from sklearn.metrics import mean_absolute_error, r2_score  
import matplotlib.pyplot as plt          
import numpy as np                      

# Load dataset 
df = pd.read_csv("/Users/noahdsouza/Desktop/overdoses/deaths_drugs.csv")

# Verify file, print rows and columns
print("Data loaded successfully")
print("Shape of dataset:", df.shape)    
print("Columns found:", df.columns.tolist())  

# Col renaming
df.columns = [
    "Country",           
    "Code",              
    "Year",              
    "Drug_Deaths",       
    "All_Cause_Deaths"   
]

# Remove dots from numbers
for col in ["Drug_Deaths", "All_Cause_Deaths"]:
    df[col] = (
        df[col]               
        .astype(str)           
        .str.replace(".", "", regex=False)  
        .astype(float)          
    )

# Drop rows with missing values 
df = df.dropna(subset=["Drug_Deaths", "All_Cause_Deaths", "Year", "Country"])

# Trend feature for Years since 1990 for better accuracy
df["Years_Since_1990"] = df["Year"] - 1990

# Make each country its own column so understood as seperate places not numbers
df = pd.get_dummies(df, columns=["Country"], drop_first=True)

# Log scale so model handles large death ranges better
df["Drug_Deaths"] = np.log1p(df["Drug_Deaths"])  
df["All_Cause_Deaths"] = np.log1p(df["All_Cause_Deaths"])

# Split data into x = inputs & y = prediction
X = df.drop(columns=["Drug_Deaths", "Code"]) 
y = df["Drug_Deaths"]

# Split data into training and testing sets, 20% testing, 80% training to beef the model lol
# random_state=42 ensures the split is the same every time 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model w/ 300 trees for more accuracy
# Keep random_state=42 for consistency
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

# Train model on training data
# Model is *supposed to* learn patterns between the features and drug deaths
model.fit(X_train, y_train)

# Predict values for the test data 
y_pred_log = model.predict(X_test)

# Convert log nums to real death counts
y_pred = np.expm1(y_pred_log)  
y_true = np.expm1(y_test)

# Evaluate model performance
# MAE: Average difference between predicted and actual results
# R²: Model variance (1.0 = perfect)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Performance metrics
print("\nModel Performance:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R-squared (R²):", round(r2, 3))

# Predictions for future years
future_years = pd.DataFrame({
    "Year": [2023, 2024, 2025],
    "All_Cause_Deaths": [df["All_Cause_Deaths"].mean()] * 3,
    "Years_Since_1990": [2023-1990, 2024-1990, 2025-1990]
})

# Add missing country cols that = 0
for col in X.columns:
    if col not in future_years.columns:
        future_years[col] = 0

# Predict future log values, convert back
future_predictions_log = model.predict(future_years[X.columns])
future_predictions = np.expm1(future_predictions_log)
future_years["Predicted_Drug_Deaths"] = future_predictions

# Input prediction table
print("\nPredicted Drug Deaths for Future Years:")
print(future_years[["Year", "Predicted_Drug_Deaths"]])

# Actual vs Predicted deaths from test data
# Scatter plot of real data points vs what model predicted
plt.figure(figsize=(8, 5))
# Blue = actual values, Orange = predicted values
plt.scatter(X_test["Year"], y_true, label="Actual", alpha=0.6)    
plt.scatter(X_test["Year"], y_pred, label="Predicted", alpha=0.6, color="orange")  
#trend line:
plt.plot(sorted(X_test["Year"]), 
         np.poly1d(np.polyfit(X_test["Year"], y_true, 1))(sorted(X_test["Year"])), 
         color="blue", linewidth=2, linestyle="--", label="Actual Trend")
plt.plot(sorted(X_test["Year"]), 
         np.poly1d(np.polyfit(X_test["Year"], y_pred, 1))(sorted(X_test["Year"])), 
         color="orange", linewidth=2, linestyle="--", label="Predicted Trend")
plt.title("Actual vs Predicted Drug Deaths by Year")              
plt.xlabel("Year")                                                
plt.ylabel("Drug-related Deaths")                                 
plt.legend()                                                      
plt.tight_layout()
plt.show()   

# Most important features influencing prediction
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTop Features Driving Predictions:")
print(importances.head(10))

# Plot feature importances
plt.figure(figsize=(9,5))
importances.head(10).plot(kind="barh", color="teal")
plt.title("Top 10 Most Important Features")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
