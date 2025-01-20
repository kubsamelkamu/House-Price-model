import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import joblib

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

if __name__ == "__main__":
    # Load the preprocessed data
    data = load_data('./data/processed_train.csv')

    # Split the data into features and target variable
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    
    X.columns = X.columns.astype(str)


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for Random Forest (simplified)
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Define the parameter grid for Gradient Boosting (simplified)
    gb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 4]
    }

    # Define the metric for evaluation
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Perform Randomized Search for Random Forest
    rf_random_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42),
                                          param_distributions=rf_param_grid,
                                          scoring=mse_scorer,
                                          n_iter=10,
                                          cv=5,
                                          n_jobs=-1,
                                          verbose=2)
    rf_random_search.fit(X_train, y_train)
    print("Best parameters for Random Forest:", rf_random_search.best_params_)
    print("Best score for Random Forest:", rf_random_search.best_score_)

    # Perform Randomized Search for Gradient Boosting
    gb_random_search = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                                          param_distributions=gb_param_grid,
                                          scoring=mse_scorer,
                                          n_iter=10,
                                          cv=5,
                                          n_jobs=-1,
                                          verbose=2)
    gb_random_search.fit(X_train, y_train)
    print("Best parameters for Gradient Boosting:", gb_random_search.best_params_)
    print("Best score for Gradient Boosting:", gb_random_search.best_score_)

    # Evaluate the best models using cross-validation
    best_rf_model = rf_random_search.best_estimator_
    rf_cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring=mse_scorer)
    print("Random Forest CV scores:", rf_cv_scores)
    print("Random Forest CV mean score:", rf_cv_scores.mean())

    best_gb_model = gb_random_search.best_estimator_
    gb_cv_scores = cross_val_score(best_gb_model, X_train, y_train, cv=5, scoring=mse_scorer)
    print("Gradient Boosting CV scores:", gb_cv_scores)
    print("Gradient Boosting CV mean score:", gb_cv_scores.mean())

    # Save the best models
    joblib.dump(best_rf_model, './models/best_random_forest.pkl')
    joblib.dump(best_gb_model, './models/best_gradient_boosting.pkl')
