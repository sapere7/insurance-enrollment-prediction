# main.py
# Core machine learning pipeline for insurance enrollment prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import pickle
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    return eda_results

def preprocess_data(df):
    """
    Preprocess the data for modeling
    
    Args:
        df (pandas.DataFrame): Employee data
        
    Returns:
        tuple: X (features), y (target), and preprocessing pipeline
    """
    logger.info("Preprocessing data")
    
    # Remove employee_id as it's not useful for prediction
    if "employee_id" in df.columns:
        df = df.drop("employee_id", axis=1)
    
    # Split features and target
    X = df.drop("enrolled", axis=1)
    y = df["enrolled"]
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    logger.info(f"Numerical columns: {numerical_cols}")
    logger.info(f"Categorical columns: {categorical_cols}")
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    logger.info("Completed preprocessing setup")
    return X, y, preprocessor

def train_and_evaluate_models(X, y, preprocessor):
    """
    Train and evaluate multiple models
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        preprocessor (ColumnTransformer): Data preprocessing pipeline
        
    Returns:
        dict: Dictionary with model evaluation results
    """
    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models to evaluate
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training and evaluating {name}")
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results[name] = {
            "pipeline": pipeline,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report
        }
        
        logger.info(f"{name} results: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")
    
    return results

def plot_roc_curves(results, X_test, y_test, output_dir="figures"):
    """
    Plot ROC curves for all models
    
    Args:
        results (dict): Dictionary with model evaluation results
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        pipeline = result["pipeline"]
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend()
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/roc_curves.png")
    plt.close()

def select_best_model(results):
    """
    Select the best model based on ROC AUC score
    
    Args:
        results (dict): Dictionary with model evaluation results
        
    Returns:
        tuple: Best model name and pipeline
    """
    logger.info("Selecting best model based on ROC AUC score")
    
    # Find model with highest ROC AUC score
    best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = results[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with ROC AUC = {best_model['roc_auc']:.4f}")
    
    return best_model_name, best_model

def save_model(model_pipeline, file_path):
    """
    Save the model pipeline to a file
    
    Args:
        model_pipeline (Pipeline): Trained model pipeline
        file_path (str): Path to save the model
    """
    logger.info(f"Saving model to {file_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model_pipeline, f)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main(data_file, output_dir="output"):
    """
    Main function to run the ML pipeline
    
    Args:
        data_file (str): Path to the CSV data file
        output_dir (str): Directory to save outputs
        
    Returns:
        dict: Dictionary with results and best model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    
    # Load and explore data
    df = load_data(data_file)
    eda_results = explore_data(df)
    
    # Preprocess data
    X, y, preprocessor = preprocess_data(df)
    
    # Split data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train and evaluate models
    model_results = train_and_evaluate_models(X, y, preprocessor)
    
    # Plot ROC curves
    plot_roc_curves(model_results, X_test, y_test, output_dir=f"{output_dir}/figures")
    
    # Select best model
    best_model_name, best_model = select_best_model(model_results)
    
    # Save best model
    save_model(best_model["pipeline"], f"{output_dir}/models/{best_model_name}_pipeline.pkl")
    
    return {
        "eda_results": eda_results,
        "model_results": model_results,
        "best_model_name": best_model_name,
        "best_model": best_model
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ML Pipeline for Insurance Enrollment Prediction')
    parser.add_argument('--data_file', type=str, default='employee_data.csv', help='Path to the CSV data file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    args = parser.parse_args()
    
    main(args.data_file, args.output_dir)
