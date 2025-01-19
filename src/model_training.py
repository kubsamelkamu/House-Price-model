import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train and evaluate the model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"{model_name} Performance:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R²: {r2_score(y_test, y_pred)}")
    print("-" * 30)
    
    joblib.dump(model, f'./models/{model_name}.pkl')

if __name__ == "__main__":
    # Load the preprocessed data
    data = load_data('./data/processed_train.csv')

    # Split the data into features and target variable
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train models
    lr_model = LinearRegression()
    train_and_evaluate(lr_model, X_train, y_train, X_test, y_test, 'linear_regression')

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_evaluate(rf_model, X_train, y_train, X_test, y_test, 'random_forest')

    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    train_and_evaluate(gb_model, X_train, y_train, X_test, y_test, 'gradient_boosting')