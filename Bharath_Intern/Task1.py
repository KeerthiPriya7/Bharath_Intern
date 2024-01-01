import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

file_path = 'C:\\Users\\keert\\OneDrive\\Desktop\\housing.csv'
data = pd.read_csv(file_path)  # or specify the correct encoding for your file
print(data.head())
print(data.columns)
print(data.dtypes)

# Select features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Separate categorical and numerical features
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(exclude=['object']).columns

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = pd.concat([X[numerical_columns], pd.get_dummies(X[categorical_columns], drop_first=True)], axis=1)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42, max_depth=10)  # Adjust max_depth as needed
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Random Forest Mean Squared Error: {mse_rf}')

new = [[8575000,8800,3,2,2,1,0,0,0,1,2,0,1]]
predicted_price=rf_model.predict(new)
print(f'Predicted price for new house={predicted_price}')
