# feature_importance.py
# Calculates and visualizes feature importance using various methods.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance 
from sklearn.linear_model import LogisticRegression 
import shap
import os
import logging
import yaml 
import joblib 
import datetime 
import argparse 
from typing import Dict, Any, Tuple, List, Optional
from sklearn.pipeline import Pipeline # Import Pipeline for type hinting
from sklearn.compose import ColumnTransformer

# --- Logging Setup ---
def setup_logging(log_dir: str = "logs/analysis", level: int = logging.INFO) -> logging.Logger: 
    """Configures logging to console and a timestamped file for the analysis script."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"feature_importance_{timestamp}.log")

    # Configure root logger specifically for this script run
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger("feature_importance")
    logger.setLevel(level)
    logger.propagate = False 
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    logger.info(f"Feature importance logging configured. Log file: {log_file}")
    return logger

logger = setup_logging()

# --- Configuration Loading ---
def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Loads configuration from a YAML file and constructs absolute paths."""
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure output paths are correctly constructed relative to output_dir
        output_dir = config.get('output_dir', 'output')
        for key in ['figures_dir', 'models_dir', 'tuning_dir', 'processed_data_dir']:
             if key in config and isinstance(config[key], str) and config[key].startswith('output/'):
                 config[key] = os.path.join(output_dir, config[key].split('/', 1)[1])
             elif key not in config: 
                 config[key] = os.path.join(output_dir, key.replace('_dir',''))
        logger.info("Configuration loaded and paths processed.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

# --- Utility Functions ---
def get_feature_names(column_transformer: ColumnTransformer) -> List[str]:
    """
    Extracts feature names after transformation by a ColumnTransformer.
    Handles OneHotEncoder and passthrough columns.
    
    Args:
        column_transformer: Fitted ColumnTransformer object.
        
    Returns:
        List of feature names in the order they appear after transformation.
    """
    output_features = []
    try:
        for name, transformer, original_cols in column_transformer.transformers_:
            if name == 'remainder':
                # Get names of columns passed through 'remainder'
                remainder_indices = column_transformer.transformers_[-1][-1] # Assumes remainder is last
                if isinstance(remainder_indices, slice): # Handle slice object
                     remainder_cols_names = column_transformer.feature_names_in_[remainder_indices]
                else: # Handle list of indices/booleans
                     remainder_cols_names = [column_transformer.feature_names_in_[i] for i in remainder_indices if i < len(column_transformer.feature_names_in_)]
                output_features.extend(remainder_cols_names)
            elif transformer == 'drop' or transformer == 'passthrough': 
                # 'passthrough' case handled by remainder logic if it's the remainder transformer
                # If explicitly set as 'passthrough' for specific columns, add them
                if transformer == 'passthrough':
                     output_features.extend(original_cols)
            elif hasattr(transformer, 'get_feature_names_out'):
                # Handle transformers like OneHotEncoder, StandardScaler (if named steps)
                if isinstance(transformer, Pipeline): # If transformer is a Pipeline
                     last_step = transformer.steps[-1][1]
                     if hasattr(last_step, 'get_feature_names_out'):
                          # Pass original column names to the last step's get_feature_names_out
                          output_features.extend(last_step.get_feature_names_out(original_cols))
                     else: # Fallback for pipeline steps without get_feature_names_out
                          output_features.extend(original_cols)
                else: # If transformer is not a Pipeline
                     output_features.extend(transformer.get_feature_names_out(original_cols))
            else:
                # Fallback for transformers without get_feature_names_out
                output_features.extend(original_cols)
    except Exception as e:
        logger.error(f"Error extracting feature names from ColumnTransformer: {e}. Falling back to original column names.")
        # Fallback to original names if extraction fails
        if hasattr(column_transformer, 'feature_names_in_'):
             return column_transformer.feature_names_in_.tolist()
        else: # Cannot determine names
             return []
             
    return output_features

# --- Analysis Function ---
def analyze_feature_importance(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, 
                               model_name: str, config: Dict[str, Any]) -> Optional[Dict[str, Optional[pd.DataFrame]]]:
    """
    Analyzes and visualizes feature importance for a single fitted model pipeline.
    Calculates model-specific importance (coefficients or feature_importances_),
    permutation importance, and SHAP values. Saves plots and importance data.
    
    Args:
        pipeline: Fitted scikit-learn pipeline containing 'preprocessor' and 'model' steps.
        X_test: Test features DataFrame (original, before preprocessing).
        y_test: Test target variable.
        model_name: Name of the model (e.g., "GradientBoosting", "LogisticRegression").
        config: Configuration dictionary.
        
    Returns:
        Dictionary containing DataFrames for model-specific and permutation importance, 
        or None if a critical error occurs.
    """
    logger.info(f"Analyzing feature importance for {model_name}")
    
    output_dir = config.get('figures_dir', 'output/figures') 
    shap_sample_size = config.get('shap_sample_size', 500) 
    os.makedirs(output_dir, exist_ok=True)
    
    feature_importance_results: Dict[str, Optional[pd.DataFrame]] = {
        "model_specific_importance": None,
        "permutation_importance": None
    }
    
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
    except KeyError:
        logger.error("Pipeline structure mismatch: Expected 'preprocessor' and 'model' steps.")
        return None 

    # Get feature names after preprocessing
    try:
        # Ensure preprocessor is fitted if it wasn't already (should be by main.py)
        if not hasattr(preprocessor, 'transformers_') or not preprocessor.transformers_:
             logger.warning("Preprocessor might not be fitted. Attempting to fit on X_test for feature name extraction.")
             preprocessor.fit(X_test, y_test) # Fit on test data only for name extraction - not ideal
             
        feature_names = get_feature_names(preprocessor)
        if not feature_names: # If get_feature_names failed
             raise ValueError("Feature name extraction resulted in an empty list.")
        logger.debug(f"Successfully extracted {len(feature_names)} feature names.")
    except Exception as e:
        logger.error(f"Could not get feature names from preprocessor: {e}. Analysis might be incomplete.")
        feature_names = X_test.columns.tolist() # Fallback

    # --- Model-Specific Importance ---
    feature_imp_df = None
    if isinstance(model, LogisticRegression) and hasattr(model, 'coef_'):
        try:
            coefficients = model.coef_[0]
            if len(coefficients) == len(feature_names):
                importance = np.abs(coefficients)
                feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
                feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
                
                # Visualize coefficients
                plt.figure(figsize=(12, 10))
                top_features = feature_imp_df.head(20) 
                coef_map = dict(zip(feature_names, coefficients))
                top_coefs = [coef_map.get(f, 0) for f in top_features['Feature']]
                colors = ['#FF9999' if c < 0 else '#99FF99' for c in top_coefs] 
                
                sns.barplot(x='Importance', y='Feature', data=top_features, palette=colors)
                plt.title(f"Feature Importance - {model_name} (Coefficient Magnitude)", fontsize=16)
                plt.xlabel("Coefficient Magnitude (|coef_|)", fontsize=14)
                plt.ylabel("Feature", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model_name}_coefficients.png"))
                plt.close()
                logger.info(f"Coefficient importance plot saved for {model_name}.")
            else:
                 logger.warning(f"Length mismatch: Coefficients ({len(coefficients)}) vs Feature Names ({len(feature_names)}). Skipping coefficient plot.")
        except Exception as e:
            logger.warning(f"Could not plot coefficients for {model_name}: {e}")
            
    elif hasattr(model, 'feature_importances_'): # For tree-based models
        try:
            importance = model.feature_importances_
            if len(importance) == len(feature_names):
                feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
                feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
                
                # Visualize
                plt.figure(figsize=(12, 10))
                top_features = feature_imp_df.head(20) 
                sns.barplot(x='Importance', y='Feature', data=top_features)
                plt.title(f"Feature Importance - {model_name} (Gini/Gain)", fontsize=16)
                plt.xlabel("Importance", fontsize=14)
                plt.ylabel("Feature", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"))
                plt.close()
                logger.info(f"Model feature importance plot saved for {model_name}.")
            else:
                 logger.warning(f"Length mismatch: Importances ({len(importance)}) vs Feature Names ({len(feature_names)}). Skipping importance plot.")
        except Exception as e:
            logger.warning(f"Could not plot feature importances for {model_name}: {e}")
    else:
        logger.info(f"Model type {model_name} does not have standard feature_importances_ or coef_ attribute.")
    
    feature_importance_results["model_specific_importance"] = feature_imp_df

    # --- Permutation Importance ---
    perm_imp_df = None
    try:
        logger.info(f"Calculating permutation importance for {model_name}...")
        perm_sample_size = min(1000, len(X_test)) 
        X_perm_sample = X_test.sample(perm_sample_size, random_state=config.get('random_state', 42))
        y_perm_sample = y_test.loc[X_perm_sample.index]

        # Calculate permutation importance using the full pipeline
        perm_importance_result = permutation_importance(
            pipeline, X_perm_sample, y_perm_sample, 
            n_repeats=10, 
            random_state=config.get('random_state', 42),
            n_jobs=config.get('tuning', {}).get('n_jobs', -1) 
        )
        
        # Use original feature names (X_test.columns) for permutation importance results
        perm_imp_df = pd.DataFrame({
            'Feature': X_test.columns, 
            'Importance': perm_importance_result.importances_mean,
            'Std Dev': perm_importance_result.importances_std
        })
        perm_imp_df = perm_imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Visualize permutation importance
        plt.figure(figsize=(12, 10))
        top_features_perm = perm_imp_df.head(min(20, len(perm_imp_df)))
        plt.barh(
            top_features_perm['Feature'], 
            top_features_perm['Importance'], 
            align='center', 
            alpha=0.8
        )
        plt.gca().invert_yaxis() 
        plt.title(f"Permutation Importance - {model_name}", fontsize=16)
        plt.xlabel("Mean Importance (Decrease in Score)", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_permutation_importance.png"))
        plt.close()
        logger.info(f"Permutation importance calculated and plot saved for {model_name}.")
    except Exception as e:
        logger.warning(f"Could not calculate or plot permutation importance for {model_name}: {e}", exc_info=True)
        
    feature_importance_results["permutation_importance"] = perm_imp_df

    # --- SHAP Analysis ---
    X_sample = X_test.sample(min(shap_sample_size, len(X_test)), random_state=config.get('random_state', 42))
    
    try:
        logger.info(f"Calculating SHAP values for {model_name} (sample size: {len(X_sample)})...")
        X_sample_transformed = preprocessor.transform(X_sample)
        
        # Try creating DataFrame with feature names for better SHAP plots
        try:
            X_sample_transformed_df = pd.DataFrame(X_sample_transformed, columns=feature_names, index=X_sample.index)
            plot_data = X_sample_transformed_df
            plot_feature_names = feature_names
        except Exception as df_err: 
            logger.warning(f"Could not create DataFrame for transformed SHAP data: {df_err}. Using numpy array and generic names.")
            X_sample_transformed_df = X_sample_transformed # Use numpy array
            plot_data = X_sample_transformed
            # Generate generic names if feature_names list is incorrect length
            num_transformed_features = X_sample_transformed.shape[1]
            if len(feature_names) != num_transformed_features:
                 plot_feature_names = [f"Feature_{i}" for i in range(num_transformed_features)]
            else:
                 plot_feature_names = feature_names


        # Create SHAP explainer
        explainer = None
        shap_values = None
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, plot_data) # Pass data directly
            shap_values = explainer.shap_values(plot_data) # Explain the data
        elif hasattr(model, 'predict_proba'): 
             if type(model).__name__ in ['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'LGBMClassifier']:
                 # Initialize TreeExplainer with model and background data
                 explainer = shap.TreeExplainer(model, plot_data, feature_perturbation="interventional", model_output="raw")
                 # Calculate SHAP values for the sample data
                 shap_values = explainer.shap_values(plot_data) 
             else: 
                 logger.warning(f"Using KernelExplainer for {model_name}, this might be slow.")
                 # KernelExplainer needs a function that takes transformed data
                 predict_proba_func = lambda x: pipeline.named_steps['model'].predict_proba(x)[:,1] 
                 explainer = shap.KernelExplainer(predict_proba_func, plot_data) 
                 shap_values = explainer.shap_values(plot_data, nsamples=100) 
        else: 
             logger.warning(f"SHAP analysis might not be fully supported for model type {type(model).__name__}. Skipping.")

        if shap_values is not None:
            # Ensure shap_values is a 2D numpy array
            if isinstance(shap_values, list): 
                 shap_values = np.array(shap_values)
            # Handle potential multi-class output from KernelExplainer or others
            if len(shap_values.shape) == 3: # e.g., (n_classes, n_samples, n_features)
                 shap_values = shap_values[1] # Assume index 1 corresponds to the positive class
            elif len(shap_values.shape) == 1: # Reshape if it's 1D
                 shap_values = shap_values.reshape(-1, 1)
                 
            # Final dimension check
            if shap_values.shape[1] != len(plot_feature_names):
                 logger.error(f"Final SHAP values dimension ({shap_values.shape[1]}) still does not match feature names ({len(plot_feature_names)}). Skipping SHAP plots.")
            else:
                # Plot SHAP summary
                plt.figure() 
                shap.summary_plot(shap_values, plot_data, feature_names=plot_feature_names, show=False)
                plt.title(f"SHAP Feature Importance - {model_name}", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"))
                plt.close()
                logger.info(f"SHAP summary plot saved for {model_name}.")
                    
                # Plot SHAP dependence plots for top features
                mean_abs_shap = np.abs(shap_values).mean(0)
                top_features_idx = np.argsort(-mean_abs_shap)[:3] 
                    
                for idx in top_features_idx:
                    if idx < len(plot_feature_names): 
                        feature_name_safe = plot_feature_names[idx].replace('/', '_').replace('\\', '_').replace(':', '_') 
                        plt.figure() 
                        shap.dependence_plot(idx, shap_values, plot_data, feature_names=plot_feature_names, show=False)
                        plt.title(f"SHAP Dependence - {plot_feature_names[idx]} ({model_name})", fontsize=14) 
                        plt.tight_layout() 
                        plt.savefig(os.path.join(output_dir, f"{model_name}_shap_dependence_{feature_name_safe}.png"))
                        plt.close()
                    else:
                         logger.warning(f"Index {idx} out of bounds for feature names list during SHAP dependence plot.")
                logger.info(f"SHAP dependence plots saved for {model_name}.")
                
    except Exception as e:
        logger.warning(f"Could not generate SHAP visualizations for {model_name}: {e}", exc_info=True)
    
    logger.info(f"Feature importance analysis completed for {model_name}")
    return feature_importance_results

# --- Standalone Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze feature importance for a trained model.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to analyze (e.g., GradientBoosting).')
    
    args = parser.parse_args()
    
    logger.info(f"--- Starting Feature Importance Analysis for {args.model_name} ---")
    
    config = load_config(args.config)
    
    models_dir = config.get('models_dir')
    processed_data_dir = config.get('processed_data_dir')
    figures_dir = config.get('figures_dir') 
    
    # Construct model path (prefer tuned model)
    tuned_model_path = os.path.join(models_dir, f"{args.model_name}_tuned_pipeline.joblib")
    initial_model_path = os.path.join(models_dir, f"{args.model_name}_pipeline.joblib")

    model_path_to_load = None
    if os.path.exists(tuned_model_path):
        model_path_to_load = tuned_model_path
    elif os.path.exists(initial_model_path):
        model_path_to_load = initial_model_path
    
    if model_path_to_load:
        logger.info(f"Loading model from: {model_path_to_load}")
        try:
            pipeline = joblib.load(model_path_to_load)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path_to_load}: {e}")
            exit(1)
    else:
        logger.error(f"No model file found for {args.model_name} in {models_dir}")
        exit(1)
        
    # Load processed TEST data
    try:
        X_test = pd.read_pickle(os.path.join(processed_data_dir, 'X_test.pkl'))
        y_test = pd.read_pickle(os.path.join(processed_data_dir, 'y_test.pkl'))
    except Exception as e:
        logger.error(f"Failed to load processed test data from {processed_data_dir}: {e}")
        exit(1)

    # Run analysis
    analysis_results = analyze_feature_importance(pipeline, X_test, y_test, args.model_name, config)

    # Save results to CSV
    if analysis_results:
         if analysis_results.get("permutation_importance") is not None:
              perm_imp_path = os.path.join(figures_dir, f"{args.model_name}_permutation_importance.csv")
              try:
                   analysis_results["permutation_importance"].to_csv(perm_imp_path, index=False)
                   logger.info(f"Permutation importance saved to {perm_imp_path}")
              except Exception as e:
                   logger.error(f"Failed to save permutation importance CSV: {e}")
                   
         if analysis_results.get("model_specific_importance") is not None:
              model_imp_path = os.path.join(figures_dir, f"{args.model_name}_model_importance.csv")
              try:
                   analysis_results["model_specific_importance"].to_csv(model_imp_path, index=False)
                   logger.info(f"Model-specific importance saved to {model_imp_path}")
              except Exception as e:
                   logger.error(f"Failed to save model-specific importance CSV: {e}")
              
    logger.info(f"--- Feature Importance Analysis Finished for {args.model_name} ---")
