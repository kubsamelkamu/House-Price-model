# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from data_Processing import load_data,preprocess_data

data = load_data('../data/train.csv')
processed_data = preprocess_data(data)

X = processed_data.drop('SalePrice', axis=1)
y = processed_data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"{model_name} MAE:", mean_absolute_error(y_test, y_pred))
    print(f"{model_name} MSE:", mean_squared_error(y_test, y_pred))
    print(f"{model_name} R2:", r2_score(y_test, y_pred))
    
    joblib.dump(model, f'../models/{model_name}.pkl')

lr_model = LinearRegression()
train_and_evaluate(lr_model, 'linear_regression')

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
train_and_evaluate(rf_model, 'random_forest')

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
train_and_evaluate(gb_model, 'gradient_boosting')