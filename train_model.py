import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import logger
from preprocessing import (
    load_raw_data, clean_data, enrich_dataset,
    remove_outliers, fit_transform_pipeline
)
from feature_engineering import engineer_features
from model import get_models, evaluate_predictions, perform_cross_validation, tune_hyperparameters

def save_plots(model, X_test, y_test, feature_names, charts_dir="charts"):
    """
    Generates and saves model evaluation plots:
    - Actual vs Predicted Scatter Plot
    - Residual Distribution
    - Feature Importance (Top 15 features)
    """
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
        logger.info(f"Created charts directory at {charts_dir}")
        
    y_pred = model.predict(X_test)
    
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="#4F46E5")
    # Draw line representing perfect prediction
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--r', lw=2)
    plt.title("Actual vs. Predicted House Prices", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Actual Price (INR)", fontsize=12)
    plt.ylabel("Predicted Price (INR)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    # 2. Residuals Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color="#10B981", bins=30)
    plt.axvline(0, color="red", linestyle="--", lw=1.5)
    plt.title("Distribution of Residuals (Errors)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Residual Value (Actual - Predicted)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "residuals_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Feature Importance (Random Forest Regressor)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take top 15 features
        top_n = min(15, len(feature_names))
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_importances, y=top_features, palette="viridis", hue=top_features, legend=False)
        plt.title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Relative Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.grid(True, axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "feature_importance.png"), dpi=300)
        plt.close()
        logger.info("Saved all evaluation charts to charts/")
    elif hasattr(model, 'coef_'):
        # For linear models, plot coefficients
        coefs = np.abs(model.coef_)
        indices = np.argsort(coefs)[::-1]
        top_n = min(15, len(feature_names))
        top_indices = indices[:top_n]
        top_coefs = coefs[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_coefs, y=top_features, palette="viridis", hue=top_features, legend=False)
        plt.title(f"Top {top_n} Feature Coefficients (Linear Regression)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Absolute Coefficient Value", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.grid(True, axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "feature_importance.png"), dpi=300)
        plt.close()
        logger.info("Saved all evaluation charts to charts/")

def main():
    logger.info("Starting House Price Prediction Training Pipeline...")
    
    # 1. Load Data
    raw_df = load_raw_data()
    
    # Save a copy of raw data description
    os.makedirs("dataset", exist_ok=True)
    
    # 2. Clean Data
    cleaned_df = clean_data(raw_df)
    
    # 3. Enrich Dataset
    enriched_df = enrich_dataset(cleaned_df)
    
    # 4. Feature Engineering
    engineered_df = engineer_features(enriched_df)
    
    # Save final cleaned/engineered dataset for reference
    processed_csv_path = "dataset/house_prices_processed.csv"
    engineered_df.to_csv(processed_csv_path, index=False)
    logger.info(f"Saved processed dataset to {processed_csv_path}")
    
    # 5. Outlier Removal
    final_df = remove_outliers(engineered_df)
    
    # 6. Fit & Transform Pipelines (scaling & encoding)
    X_train_proc, X_test_proc, y_train, y_test, scaler, encoder, feature_names = fit_transform_pipeline(final_df)
    
    # 7. Model Training & Comparison
    models = get_models()
    comparison_results = []
    
    for name, model in models.items():
        logger.info(f"Training model: {name}...")
        # Cross validation score
        cv_r2_mean, cv_r2_std = perform_cross_validation(model, X_train_proc, y_train)
        
        # Fit on training data
        model.fit(X_train_proc, y_train)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test_proc)
        metrics = evaluate_predictions(y_test, y_test_pred)
        
        comparison_results.append({
            'Model': name,
            'Train CV R2': f"{cv_r2_mean:.4f} ± {cv_r2_std:.4f}",
            'Test R2': metrics['R2'],
            'Test MAE': metrics['MAE'],
            'Test MSE': metrics['MSE'],
            'Test RMSE': metrics['RMSE']
        })
        
    df_comparison = pd.DataFrame(comparison_results)
    logger.info("\n--- Model Comparison Table ---\n" + df_comparison.to_string(index=False))
    
    # 8. Hyperparameter Tuning of Random Forest
    tuned_rf, best_params, best_cv_score = tune_hyperparameters(X_train_proc, y_train)
    
    # Evaluate Tuned RF
    y_test_pred_tuned = tuned_rf.predict(X_test_proc)
    tuned_metrics = evaluate_predictions(y_test, y_test_pred_tuned)
    
    logger.info(f"Tuned Random Forest Performance: Test R2 = {tuned_metrics['R2']:.4f}, Test MAE = {tuned_metrics['MAE']:.2f}")
    
    # Add Tuned RF to comparison dataframe
    tuned_row = {
        'Model': 'Tuned Random Forest Regressor',
        'Train CV R2': f"{best_cv_score:.4f}",
        'Test R2': tuned_metrics['R2'],
        'Test MAE': tuned_metrics['MAE'],
        'Test MSE': tuned_metrics['MSE'],
        'Test RMSE': tuned_metrics['RMSE']
    }
    df_comparison = pd.concat([df_comparison, pd.DataFrame([tuned_row])], ignore_index=True)
    
    # Save comparison metrics as JSON for frontend
    metrics_path = "dataset/model_metrics.json"
    df_comparison.to_json(metrics_path, orient='records', indent=4)
    logger.info(f"Saved comparison metrics to {metrics_path}")
    
    # 9. Model Selection
    # Select Best Model based on Test R2
    best_model_idx = df_comparison['Test R2'].astype(float).idxmax()
    best_model_name = df_comparison.loc[best_model_idx, 'Model']
    logger.info(f"Selected Best Model for Production: {best_model_name}")
    
    if best_model_name == 'Tuned Random Forest Regressor':
        best_model = tuned_rf
    else:
        # Fallback to model trained in the loop or refit
        best_model = models.get(best_model_name)
        if best_model is None:
            # If not in dict, use tuned RF as default best
            best_model = tuned_rf
            
    # Save the best model
    model_save_path = "house_price_model.pkl"
    joblib.dump(best_model, model_save_path)
    logger.info(f"Saved the best trained model to {model_save_path}")
    
    # 10. Generate and Save Visualizations
    save_plots(best_model, X_test_proc, y_test, feature_names)
    
    logger.info("Model training pipeline completed successfully!")

if __name__ == "__main__":
    main()
