# run_pipeline.py
# Complete ML pipeline runner for insurance enrollment prediction

import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import argparse
import time

# Import project modules
from main import load_data, explore_data, preprocess_data, train_and_evaluate_models, select_best_model, save_model
from eda import create_eda_visualizations
from feature_importance import analyze_feature_importance
from hyperparameter_tuning import tune_all_models

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(data_file, output_dir, run_tuning=False):
    """
    Run the complete ML pipeline
    
    Args:
        data_file (str): Path to the CSV data file
        output_dir (str): Directory to save outputs
        run_tuning (bool): Whether to run hyperparameter tuning
        
    Returns:
        dict: Pipeline results
    """
    start_time = time.time()
    logger.info(f"Starting ML pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    models_dir = os.path.join(output_dir, "models")
    results_dir = os.path.join(output_dir, "results")
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    df = load_data(data_file)
    
    # Step 2: Exploratory data analysis
    logger.info("Step 2: Performing exploratory data analysis")
    eda_results = explore_data(df)
    
    # Save EDA results
    pd.DataFrame(eda_results["target_distribution"], index=[0]).to_csv(
        f"{results_dir}/target_distribution.csv", index=False
    )
    
    # Step 3: Create visualizations
    logger.info("Step 3: Creating visualizations")
    df_with_bins = create_eda_visualizations(df, figures_dir)
    
    # Step 4: Preprocess data
    logger.info("Step 4: Preprocessing data")
    X, y, preprocessor = preprocess_data(df)
    
    # Step 5: Split data
    logger.info("Step 5: Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Step 6: Train and evaluate models
    logger.info("Step 6: Training and evaluating models")
    model_results = train_and_evaluate_models(X, y, preprocessor)
    
    # Save model evaluation results
    for model_name, result in model_results.items():
        pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            "Value": [
                result["accuracy"],
                result["precision"],
                result["recall"],
                result["f1"],
                result["roc_auc"]
            ]
        }).to_csv(f"{results_dir}/{model_name}_metrics.csv", index=False)
    
    # Step 7: Feature importance analysis
    logger.info("Step 7: Analyzing feature importance")
    feature_importance_results = analyze_feature_importance(model_results, X, y, figures_dir)
    
    # Step 8: Hyperparameter tuning (optional)
    tuned_models = None
    if run_tuning:
        logger.info("Step 8: Running hyperparameter tuning")
        tuning_dir = os.path.join(output_dir, "tuning")
        os.makedirs(tuning_dir, exist_ok=True)
        
        tuned_models = tune_all_models(X_train, y_train, preprocessor, cv=5, n_jobs=-1, output_dir=tuning_dir)
        
        # Save tuned models
        for model_name, result in tuned_models.items():
            save_model(result["model"], f"{models_dir}/{model_name}_tuned_pipeline.pkl")
    
    # Step 9: Select best model
    logger.info("Step 9: Selecting best model")
    best_model_name, best_model = select_best_model(model_results)
    
    # Step 10: Save best model
    logger.info(f"Step 10: Saving best model ({best_model_name})")
    save_model(best_model["pipeline"], f"{models_dir}/{best_model_name}_pipeline.pkl")
    
    elapsed_time = time.time() - start_time
    logger.info(f"ML pipeline completed in {elapsed_time:.2f} seconds")
    
    return {
        "eda_results": eda_results,
        "model_results": model_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "tuned_models": tuned_models
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the complete ML pipeline for insurance enrollment prediction')
    parser.add_argument('--data_file', type=str, default='employee_data.csv', help='Path to the CSV data file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--run_tuning', action='store_true', help='Run hyperparameter tuning (time-consuming)')
    
    args = parser.parse_args()
    
    run_pipeline(args.data_file, args.output_dir, args.run_tuning)
