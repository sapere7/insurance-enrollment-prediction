# test_script.py
# Loads a trained model pipeline and tests it with sample data and optionally evaluates on a dataset.

import os
import joblib 
import pandas as pd
import numpy as np
import argparse
import logging 
import datetime 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

# --- Logging Setup ---
def setup_logging(log_dir: str = "logs/test", level: int = logging.INFO) -> logging.Logger: 
    """Configures logging to console and a timestamped file for the test script."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_run_{timestamp}.log")

    # Configure logger for this script
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger("test_script")
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

    logger.info(f"Test script logging configured. Log file: {log_file}")
    return logger

logger = setup_logging()

# --- Model Loading ---
def load_model(model_path: str) -> Any:
    """
    Loads a saved model pipeline using joblib.
    
    Args:
        model_path: Path to the saved model file (.joblib).
        
    Returns:
        The loaded model pipeline object.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: For other loading errors.
    """
    logger.info(f"Loading model from {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# --- Test Data Creation ---
def create_test_employees() -> pd.DataFrame:
    """
    Creates a small DataFrame of sample employee records for prediction testing.
    
    Returns:
        DataFrame with sample employee data.
    """
    logger.info("Creating sample test employee records")
    test_employees = [
        {"age": 35, "gender": "Female", "marital_status": "Married", "salary": 75000, "employment_type": "Full-time", "region": "Northeast", "has_dependents": True, "tenure_years": 5.5},
        {"age": 28, "gender": "Male", "marital_status": "Single", "salary": 55000, "employment_type": "Full-time", "region": "West", "has_dependents": False, "tenure_years": 2.0},
        {"age": 45, "gender": "Male", "marital_status": "Married", "salary": 95000, "employment_type": "Full-time", "region": "Midwest", "has_dependents": True, "tenure_years": 12.0},
        {"age": 33, "gender": "Female", "marital_status": "Single", "salary": 62000, "employment_type": "Part-time", "region": "Southeast", "has_dependents": False, "tenure_years": 4.0},
        {"age": 52, "gender": "Female", "marital_status": "Divorced", "salary": 85000, "employment_type": "Full-time", "region": "Northeast", "has_dependents": True, "tenure_years": 8.0},
        {"age": 24, "gender": "Male", "marital_status": "Single", "salary": 48000, "employment_type": "Contract", "region": "West", "has_dependents": False, "tenure_years": 0.5},
        {"age": 42, "gender": "Male", "marital_status": "Married", "salary": 78000, "employment_type": "Full-time", "region": "Southwest", "has_dependents": True, "tenure_years": 7.0},
        {"age": 56, "gender": "Female", "marital_status": "Widowed", "salary": 92000, "employment_type": "Full-time", "region": "Midwest", "has_dependents": False, "tenure_years": 15.0},
        {"age": 31, "gender": "Female", "marital_status": "Married", "salary": 68000, "employment_type": "Full-time", "region": "Southeast", "has_dependents": True, "tenure_years": 3.5},
        {"age": 38, "gender": "Male", "marital_status": "Divorced", "salary": 72000, "employment_type": "Full-time", "region": "Northeast", "has_dependents": True, "tenure_years": 6.0}
    ]
    return pd.DataFrame(test_employees)

# --- Test Functions ---
def test_with_sample_data(model: Any, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Tests the loaded model pipeline with predefined sample employee data, 
    logs predictions, and optionally saves a visualization.
    
    Args:
        model: The loaded (fitted) model pipeline.
        output_dir: Directory to save the prediction visualization plot. If None, plot is not saved.
        
    Returns:
        DataFrame containing the sample data along with prediction probabilities and outcomes.
        
    Raises:
        Exception: If prediction fails.
    """
    logger.info("Testing model with sample employee data...")
    test_data = create_test_employees()
    
    try:
        probabilities = model.predict_proba(test_data)[:, 1]
        predictions = model.predict(test_data)
        
        test_data["enrollment_probability"] = probabilities
        test_data["predicted_enrollment"] = predictions
        
        logger.info("Prediction Results for Sample Data:")
        for i, row in test_data.iterrows():
            logger.info(f"--- Employee {i+1} ---")
            logger.info(f"  Input: Age={row['age']}, Gender={row['gender']}, Marital={row['marital_status']}, Deps={row['has_dependents']}, Salary={row['salary']}, EmpType={row['employment_type']}, Region={row['region']}, Tenure={row['tenure_years']}")
            logger.info(f"  Output: Probability={row['enrollment_probability']:.4f}, Prediction={'Yes' if row['predicted_enrollment'] else 'No'}")
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(12, 8))
            
            sorted_indices = np.argsort(test_data["enrollment_probability"])
            plt.barh(
                range(len(sorted_indices)), 
                test_data["enrollment_probability"].iloc[sorted_indices],
                color=['#FF9999' if p > 0.5 else '#9999FF' for p in test_data["enrollment_probability"].iloc[sorted_indices]]
            )
            
            employee_labels = [
                f"Age: {row['age']}, {row['gender']}, {row['employment_type']}, ${row['salary']/1000:.0f}K" 
                for i, row in test_data.iloc[sorted_indices].iterrows()
            ]
            
            plt.yticks(range(len(sorted_indices)), employee_labels)
            plt.xlabel("Enrollment Probability", fontsize=14)
            plt.title("Predicted Enrollment Probabilities for Test Employees", fontsize=16)
            plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
            plt.legend()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, "test_sample_predictions.png")
            plt.savefig(save_path)
            logger.info(f"Sample prediction visualization saved to {save_path}")
            plt.close() 
            
        return test_data
        
    except Exception as e:
        logger.error(f"Error making predictions on sample data: {e}", exc_info=True)
        raise

def test_with_actual_data(model: Any, data_file: str, output_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Tests the model with data from a file, calculates performance metrics, 
    and optionally saves evaluation plots.
    
    Args:
        model: The loaded (fitted) model pipeline.
        data_file: Path to the CSV data file for evaluation.
        output_dir: Directory to save evaluation plots (confusion matrix, P-R curve). If None, plots are not saved.
        
    Returns:
        Dictionary containing performance metrics ('classification_report', 'confusion_matrix', 'roc_auc'), 
        or None if testing fails (e.g., file not found).
        
    Raises:
        Exception: For errors during prediction or metric calculation.
    """
    logger.info(f"Testing model with actual data from {data_file}")
    
    try:
        logger.info(f"Loading actual data from {data_file}")
        df = pd.read_csv(data_file)
        
        # Prepare data (assuming same structure as training, dropping ID and target)
        X = df.drop(["employee_id", "enrolled"], axis=1, errors='ignore')
        if "enrolled" not in df.columns:
             logger.error("Target column 'enrolled' not found in data file.")
             return None
        y = df["enrolled"]
        
        logger.info("Making predictions on actual data...")
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        logger.info("Predictions completed.")
        
        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        classification_rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
        conf_mat = confusion_matrix(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)
        
        logger.info("Model Performance on Actual Data:")
        logger.info(f"  Accuracy: {classification_rep.get('accuracy', 'N/A'):.4f}")
        class_1_metrics = classification_rep.get('1', {}) # Metrics for the positive class (1)
        logger.info(f"  Precision (Class 1): {class_1_metrics.get('precision', 'N/A'):.4f}")
        logger.info(f"  Recall (Class 1): {class_1_metrics.get('recall', 'N/A'):.4f}")
        logger.info(f"  F1 Score (Class 1): {class_1_metrics.get('f1-score', 'N/A'):.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        # Create visualizations if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Enrolled', 'Enrolled'], yticklabels=['Not Enrolled', 'Enrolled']
            )
            plt.xlabel('Predicted Label', fontsize=14); plt.ylabel('True Label', fontsize=14)
            plt.title('Confusion Matrix (Actual Data)', fontsize=16)
            plt.tight_layout()
            cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close() 
            
            # Plot precision-recall curve
            from sklearn.metrics import precision_recall_curve 
            precision, recall, _ = precision_recall_curve(y, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, marker='.', label='Model')
            plt.xlabel('Recall', fontsize=14); plt.ylabel('Precision', fontsize=14)
            plt.title('Precision-Recall Curve (Actual Data)', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pr_path = os.path.join(output_dir, "test_precision_recall_curve.png")
            plt.savefig(pr_path)
            plt.close() 
            
            logger.info(f"Result visualizations saved to {output_dir}/")
        
        return {
            "classification_report": classification_rep,
            "confusion_matrix": conf_mat.tolist(), 
            "roc_auc": roc_auc
        }
        
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_file}")
        return None
    except Exception as e:
        logger.error(f"Error testing with actual data: {e}", exc_info=True)
        raise 

# --- Main Execution ---
def main():
    """Parses arguments, loads the model, and runs tests."""
    parser = argparse.ArgumentParser(description='Test insurance enrollment prediction model')
    parser.add_argument('--model_path', type=str, default='output/models/GradientBoosting_pipeline.joblib', 
                        help='Path to the saved model pipeline file (.joblib)')
    parser.add_argument('--data_file', type=str, default='employee_data.csv', 
                        help='Path to the data file for evaluation (optional)')
    parser.add_argument('--output_dir', type=str, default='output/test_results', 
                        help='Directory to save test results and visualizations')
    
    args = parser.parse_args()
    
    logger.info("--- Starting Model Test Script ---")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data file for evaluation: {args.data_file}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        model = load_model(args.model_path)
        test_with_sample_data(model, args.output_dir)
        
        if os.path.exists(args.data_file):
            test_with_actual_data(model, args.data_file, args.output_dir)
        else:
            logger.warning(f"Data file {args.data_file} not found. Skipping evaluation with actual data.")
            
        logger.info("--- Model Test Script Finished ---")

    except Exception as e:
        # Errors during loading or prediction are logged within the functions
        logger.info("--- Model Test Script Finished with Errors ---")


if __name__ == "__main__":
    main()
