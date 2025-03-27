# main.py
# Core machine learning pipeline for insurance enrollment prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, make_scorer)
import pickle
import joblib # Using joblib for potentially large numpy arrays in preprocessor
import os
import logging
import yaml 
import datetime 
from typing import Dict, Any, Tuple, List, Optional 

# --- Logging Setup ---
def setup_logging(log_dir="logs", level=logging.INFO):
    """Configures logging to console and a timestamped file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_run_{timestamp}.log")

    # Remove existing handlers if any to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure root logger
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format,
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler() # To console
                        ])
    logger = logging.getLogger(__name__) 
    logger.info(f"Logging configured. Log file: {log_file}")
    return logger

logger = setup_logging() 

# --- Configuration Loading ---
def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        # Basic validation for essential keys
        essential_keys = ['random_state', 'test_size', 'data_file', 'output_dir', 'models']
        if not all(k in config for k in essential_keys):
            logger.warning(f"Config file might be missing essential keys: {essential_keys}")
        return config 
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

# --- Data Handling ---
def load_data(file_path):
    """
    Load the employee data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def explore_data(df):
    """
    Perform exploratory data analysis
    
    Args:
        df (pandas.DataFrame): Employee data
        
    Returns:
        dict: Dictionary with EDA results
    """
    logger.info("Performing exploratory data analysis")
    
    # Basic information
    eda_results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df["enrolled"].value_counts().to_dict()
    }
    
    # Check for duplicates
    eda_results["duplicate_rows"] = df.duplicated().sum()
    
    # Summary statistics for numerical columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "employee_id" in numerical_cols:
        numerical_cols.remove("employee_id")  # Remove ID from numerical analysis
    if "enrolled" in numerical_cols:
        numerical_cols.remove("enrolled")  # Remove target from numerical analysis
    
    eda_results["numerical_stats"] = df[numerical_cols].describe().to_dict()
    
    # Distribution of categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    eda_results["categorical_distributions"] = {col: df[col].value_counts().to_dict() for col in categorical_cols}
    
    logger.info("Completed exploratory data analysis")
    # Note: EDA results are not used further in this specific script's main flow.
    return eda_results

def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Preprocess the data: feature/target split, define preprocessor.
    
    Args:
        df (pandas.DataFrame): Input data.
        config (dict): Configuration dictionary.
        
    Returns:
        tuple: X (features), y (target), preprocessor object.
    """
    logger.info("Preprocessing data")
    
    # Simple check for potential target leakage via high correlation
    if 'enrolled' in df.columns:
        numeric_df = df.select_dtypes(include=np.number)
        if 'employee_id' in numeric_df.columns:
             numeric_df = numeric_df.drop('employee_id', axis=1)
        # Ensure 'enrolled' is numeric before calculating correlation
        if pd.api.types.is_numeric_dtype(numeric_df.get('enrolled')):
            correlations = numeric_df.corr()['enrolled'].abs().sort_values(ascending=False)
            # Check for features highly correlated with target (excluding self-correlation of 1.0)
            potential_leaks = correlations[(correlations > 0.95) & (correlations < 1.0)]
            if not potential_leaks.empty:
                logger.warning(f"Potential target leakage detected! Features with >0.95 correlation with 'enrolled': {potential_leaks.index.tolist()}")
        else:
            logger.warning("Target 'enrolled' is not numeric, skipping correlation check for leakage.")

    # Drop ID column if present
    if "employee_id" in df.columns:
        df = df.drop("employee_id", axis=1)
        logger.info("Dropped 'employee_id' column.")
        
    # Define target and features
    target_col = "enrolled"
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in dataframe.")
        raise ValueError(f"Target column '{target_col}' not found.")
        
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Identify column types
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    logger.info(f"Identified {len(numerical_cols)} numerical columns: {numerical_cols}")
    logger.info(f"Identified {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    # Define transformers (Imputation strategies could be made configurable)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])
    
    # Create the preprocessor object
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ], 
        remainder='passthrough' # Keep any columns not specified in transformers
    )
    
    logger.info("Preprocessor defined.")
    return X, y, preprocessor

# --- Model Training and Evaluation ---
def train_and_evaluate_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.Series, y_test: pd.Series, 
                              preprocessor: ColumnTransformer, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate multiple models using the provided preprocessor and data splits.
    
    Args:
        X_train, X_test, y_train, y_test: Data splits.
        preprocessor (ColumnTransformer): Fitted preprocessor object.
        config (dict): Configuration dictionary.
        
    Returns:
        dict: Dictionary with model evaluation results for the test set.
    """
    logger.info("Training and evaluating models on test set.")
    
    # Define models using parameters from config, applying global random_state if not specified per model
    models_config = config.get('models', {})
    global_random_state = config.get('random_state', 42) 
    
    models = {}
    # Instantiate models specified in the config
    if 'LogisticRegression' in models_config:
        params = models_config['LogisticRegression'].copy() 
        params['random_state'] = params.get('random_state', global_random_state) 
        models["LogisticRegression"] = LogisticRegression(**params)
        
    if 'RandomForest' in models_config:
        params = models_config['RandomForest'].copy()
        params['random_state'] = params.get('random_state', global_random_state)
        models["RandomForest"] = RandomForestClassifier(**params)
        
    if 'GradientBoosting' in models_config:
        params = models_config['GradientBoosting'].copy()
        params['random_state'] = params.get('random_state', global_random_state)
        models["GradientBoosting"] = GradientBoostingClassifier(**params)

    if not models:
        logger.error("No models found in the configuration under the 'models' key.")
        raise ValueError("No models specified in configuration.")

    results = {}
    
    # The preprocessor passed to this function is assumed to be fitted already.
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        
        # Create a pipeline combining the pre-fitted preprocessor and the current model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor), 
            ('model', model)
        ])
        
        # Train the model step (preprocessor is already fitted and will not be re-fitted)
        logger.info(f"Training {name}...")
        pipeline.fit(X_train, y_train) 
        logger.info(f"{name} trained.")
        
        # Make predictions on the test set
        logger.info(f"Predicting with {name} on test set...")
        y_pred = pipeline.predict(X_test)
        try:
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        except AttributeError:
            logger.warning(f"{name} does not support predict_proba. ROC AUC will be NaN.")
            y_pred_proba = np.nan 

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # Calculate ROC AUC, handling potential NaN from predict_proba issues
        # Use .any() to check if any element is NaN
        roc_auc = roc_auc_score(y_test, y_pred_proba) if not np.isnan(y_pred_proba).any() else np.nan
        conf_matrix = confusion_matrix(y_test, y_pred)
        try:
            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        except ValueError: 
             logger.warning(f"Could not generate classification report for {name}, possibly due to no predicted samples in a class.")
             class_report = {} 

        # Store evaluation results
        results[name] = {
            "pipeline": pipeline, # Keep the fitted pipeline
            "accuracy": accuracy,
            "precision": precision, # Note: precision_score defaults to pos_label=1
            "recall": recall,       # Note: recall_score defaults to pos_label=1
            "f1": f1,               # Note: f1_score defaults to pos_label=1
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix.tolist(), 
            "classification_report": class_report 
        }
        
        logger.info(f"{name} Test Results: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")
    
    return results

# --- Visualization ---
def plot_roc_curves(results: Dict[str, Dict[str, Any]], X_test: pd.DataFrame, 
                    y_test: pd.Series, figures_dir: str):
    """
    Plots ROC curves for evaluated models on the test set.
    
    Args:
        results (dict): Dictionary containing model evaluation results, including fitted pipelines.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        figures_dir (str): Directory to save the ROC curve plot.
    """
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        pipeline = result["pipeline"]
        # Ensure ROC AUC is calculable before plotting
        if 'roc_auc' in result and not np.isnan(result['roc_auc']):
            try:
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.4f})")
            except Exception as e:
                 logger.warning(f"Could not plot ROC curve for {name}: {e}")
        else:
             logger.warning(f"Skipping ROC curve for {name} as ROC AUC is NaN.")

    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models (Test Set)')
    plt.legend()
    
    # Ensure directory exists
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "roc_curves_test.png")
    plt.savefig(save_path)
    logger.info(f"ROC curve plot saved to {save_path}")
    plt.close()

# --- Model Selection and Saving ---
def select_best_model(results: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
    """
    Selects the best model from the results based on a weighted score of metrics.

    Args:
        results (dict): Dictionary containing model evaluation results.
        weights (dict): Dictionary defining the weights for 'roc_auc', 'f1', and 'recall'.

    Returns:
        tuple: Name of the best model and its corresponding results dictionary.
        
    Raises:
        ValueError: If no valid models are found in the results.
    """
    logger.info(f"Selecting best model based on weighted score: {weights}")
    
    best_score = -np.inf 
    best_model_name: Optional[str] = None
    
    # Filter for models where ROC AUC could be calculated
    valid_models = {name: res for name, res in results.items() 
                    if res.get('roc_auc') is not None and not np.isnan(res.get('roc_auc'))}

    if not valid_models:
         logger.error("No valid models found in results to select from (ROC AUC might be NaN for all).")
         raise ValueError("Cannot select best model, no valid results with calculable ROC AUC.")

    for name, result in valid_models.items():
        # Get metrics, defaulting to 0 if missing or not a number
        roc_auc = result.get("roc_auc", 0)
        f1 = result.get("f1", 0) # Assumes f1 is for the positive class
        recall = result.get("recall", 0) # Assumes recall is for the positive class
        
        roc_auc = roc_auc if isinstance(roc_auc, (int, float)) and not np.isnan(roc_auc) else 0
        f1 = f1 if isinstance(f1, (int, float)) and not np.isnan(f1) else 0
        recall = recall if isinstance(recall, (int, float)) and not np.isnan(recall) else 0

        # Calculate weighted score
        score = (weights.get("roc_auc", 0) * roc_auc +
                 weights.get("f1", 0) * f1 +
                 weights.get("recall", 0) * recall)
        
        logger.debug(f"Model: {name}, ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Weighted Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model_name = name
            
    # Fallback to ROC AUC if weighted score somehow fails to select a model
    if best_model_name is None: 
        logger.warning("Could not determine best model based on weighted score. Falling back to highest ROC AUC.")
        best_model_name = max(valid_models, key=lambda name: valid_models[name].get("roc_auc", -np.inf))
        best_score = valid_models[best_model_name].get("roc_auc", -np.inf) 

    best_model_results = results[best_model_name] 
    
    # Log the selected model and its key performance indicators
    log_roc = best_model_results.get('roc_auc', np.nan)
    log_f1 = best_model_results.get('f1', np.nan) # F1 for positive class
    log_recall = best_model_results.get('recall', np.nan) # Recall for positive class
    logger.info(f"Best model selected: {best_model_name} with Weighted Score = {best_score:.4f} "
                f"(ROC AUC: {log_roc:.4f}, F1: {log_f1:.4f}, Recall: {log_recall:.4f})")
    
    return best_model_name, best_model_results

def save_object(obj: Any, file_path: str):
    """Saves a Python object to a file using joblib.

    Args:
        obj: The Python object to save.
        file_path: The path where the object will be saved.
    """
    logger.info(f"Saving object to {file_path}")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {e}")
        raise

# --- Main Pipeline Execution ---
def main(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the main ML pipeline steps: load, preprocess, split, save data, train, evaluate, select, save model.
    
    Args:
        config: Dictionary loaded from the configuration file.
        
    Returns:
        Dictionary containing summary information like best model name and paths to artifacts.
    """
    # Extract configuration values with defaults
    data_file = config.get('data_file', 'employee_data.csv')
    output_dir = config.get('output_dir', 'output') # Base output dir
    # Construct specific output paths from base output_dir and config
    figures_dir = config.get('figures_dir', os.path.join(output_dir, 'figures')) 
    models_dir = config.get('models_dir', os.path.join(output_dir, 'models'))
    processed_data_dir = config.get('processed_data_dir', os.path.join(output_dir, 'processed_data'))
    
    random_state = config.get('random_state', 42)
    test_size = config.get('test_size', 0.2)
    selection_weights = config.get('selection_weights', {"roc_auc": 0.4, "f1": 0.3, "recall": 0.3})

    # Ensure all output directories exist
    for dir_path in [figures_dir, models_dir, processed_data_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # --- Pipeline Steps ---
    logger.info("Step 1: Loading data...")
    df = load_data(data_file)
    
    logger.info("Step 2: Defining preprocessor...")
    X, y, preprocessor = preprocess_data(df, config)
    
    logger.info("Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y 
    )
    logger.info(f"Data split complete: Train={X_train.shape}, Test={X_test.shape}")

    logger.info("Step 4: Fitting preprocessor and saving processed data...")
    preprocessor.fit(X_train, y_train) 
    logger.info("Preprocessor fitted.")
    
    preprocessor_path = os.path.join(processed_data_dir, 'preprocessor.joblib')
    save_object(preprocessor, preprocessor_path)
    X_train.to_pickle(os.path.join(processed_data_dir, 'X_train.pkl'))
    X_test.to_pickle(os.path.join(processed_data_dir, 'X_test.pkl'))
    y_train.to_pickle(os.path.join(processed_data_dir, 'y_train.pkl'))
    y_test.to_pickle(os.path.join(processed_data_dir, 'y_test.pkl'))
    logger.info(f"Preprocessor and data splits saved to {processed_data_dir}")

    logger.info("Step 5: Training and evaluating initial models...")
    model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, config)
    
    logger.info("Step 6: Plotting ROC curves...")
    plot_roc_curves(model_results, X_test, y_test, figures_dir)
    
    logger.info("Step 7: Selecting best initial model...")
    best_model_name, best_model_results = select_best_model(model_results, selection_weights)
    
    logger.info("Step 8: Saving best initial model pipeline...")
    best_model_path = os.path.join(models_dir, f"{best_model_name}_pipeline.joblib")
    save_object(best_model_results["pipeline"], best_model_path)
    
    # Return summary information about the run
    return {
        "best_initial_model_name": best_model_name,
        "best_initial_model_path": best_model_path,
        "initial_test_set_metrics": {name: {k: v for k, v in res.items() if k != 'pipeline'} 
                                     for name, res in model_results.items()},
        "preprocessor_path": preprocessor_path,
        "processed_data_dir": processed_data_dir
    }

if __name__ == "__main__":
    logger.info("--- Starting Main Pipeline Execution ---")
    pipeline_config = load_config() 
    pipeline_summary = main(pipeline_config)
    
    # Log summary results
    logger.info("--- Main Pipeline Execution Summary ---")
    logger.info(f"Best Initial Model: {pipeline_summary['best_initial_model_name']}")
    logger.info(f"Saved To: {pipeline_summary['best_initial_model_path']}")
    logger.info(f"Preprocessor Saved To: {pipeline_summary['preprocessor_path']}")
    logger.info(f"Processed Data Saved To: {pipeline_summary['processed_data_dir']}")
    logger.info("Initial Model Test Set Metrics (Positive Class):")
    for model_name, metrics in pipeline_summary['initial_test_set_metrics'].items():
         log_roc = metrics.get('roc_auc', np.nan)
         log_f1 = metrics.get('f1', np.nan) # F1 for positive class
         log_recall = metrics.get('recall', np.nan) # Recall for positive class
         log_precision = metrics.get('precision', np.nan) # Precision for positive class
         logger.info(f"  {model_name}: ROC AUC={log_roc:.4f}, F1={log_f1:.4f}, Recall={log_recall:.4f}, Precision={log_precision:.4f}")
    logger.info("--- Main Pipeline Execution Finished ---")
