from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

from utils import logger

def get_models():
    """
    Returns a dictionary of base regression models to compare.
    """
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0, max_iter=10000),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    }

def evaluate_predictions(y_true, y_pred):
    """
    Calculates MAE, MSE, RMSE, and R2 score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def perform_cross_validation(model, X, y, cv=5):
    """
    Performs cross-validation and returns mean and standard deviation of R2 scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    return scores.mean(), scores.std()

def tune_hyperparameters(X_train, y_train):
    """
    Uses GridSearchCV to tune a Random Forest Regressor and returns the best model and parameters.
    """
    logger.info("Initializing GridSearchCV for Random Forest Regressor...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [8, 12, 16],
        'min_samples_split': [2, 5],
        'max_features': [1.0, 'sqrt'] # 1.0 represents 'auto' (which is deprecated in modern sklearn)
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"GridSearchCV complete. Best Params: {grid_search.best_params_}")
    logger.info(f"Best R2 CV Score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
