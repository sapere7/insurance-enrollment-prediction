# test_prediction.py
# Test script for insurance enrollment prediction model

import os
import pickle
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """
    Load the trained model pipeline
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        object: Loaded model pipeline
    """
    print(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def create_test_employees():
    """
    Create a set of test employee records
    
    Returns:
        pandas.DataFrame: Test employee data
    """
    print("Creating test employee records")
    
    # Define test data
    test_employees = [
        {
            "age": 35,
            "gender": "Female",
            "marital_status": "Married",
            "salary": 75000,
            "employment_type": "Full-time",
            "region": "Northeast",
            "has_dependents": True,
            "tenure_years": 5.5
        },
        {
            "age": 28,
            "gender": "Male",
            "marital_status": "Single",
            "salary": 55000,
            "employment_type": "Full-time",
            "region": "West",
            "has_dependents": False,
            "tenure_years": 2.0
        },
        {
            "age": 45,
            "gender": "Male",
            "marital_status": "Married",
            "salary": 95000,
            "employment_type": "Full-time",
            "region": "Midwest",
            "has_dependents": True,
            "tenure_years": 12.0
        },
        {
            "age": 33,
            "gender": "Female",
            "marital_status": "Single",
            "salary": 62000,
            "employment_type": "Part-time",
            "region": "Southeast",
            "has_dependents": False,
            "tenure_years": 4.0
        },
        {
            "age": 52,
            "gender": "Female",
            "marital_status": "Divorced",
            "salary": 85000,
            "employment_type": "Full-time",
            "region": "Northeast",
            "has_dependents": True,
            "tenure_years": 8.0
        },
        {
            "age": 24,
            "gender": "Male",
            "marital_status": "Single",
            "salary": 48000,
            "employment_type": "Contract",
            "region": "West",
            "has_dependents": False,
            "tenure_years": 0.5
        },
        {
            "age": 42,
            "gender": "Male",
            "marital_status": "Married",
            "salary": 78000,
            "employment_type": "Full-time",
            "region": "Southwest",
            "has_dependents": True,
            "tenure_years": 7.0
        },
        {
            "age": 56,
            "gender": "Female",
            "marital_status": "Widowed",
            "salary": 92000,
            "employment_type": "Full-time",
            "region": "Midwest",
            "has_dependents": False,
            "tenure_years": 15.0
        },
        {
            "age": 31,
            "gender": "Female",
            "marital_status": "Married",
            "salary": 68000,
            "employment_type": "Full-time",
            "region": "Southeast",
            "has_dependents": True,
            "tenure_years": 3.5
        },
        {
            "age": 38,
            "gender": "Male",
            "marital_status": "Divorced",
            "salary": 72000,
            "employment_type": "Full-time",
            "region": "Northeast",
            "has_dependents": True,
            "tenure_years": 6.0
        }
    ]
    
    return pd.DataFrame(test_employees)

def test_with_sample_data(model, output_dir=None):
    """
    Test the model with sample employee data
    
    Args:
        model: Trained model pipeline
        output_dir (str): Directory to save visualization (optional)
        
    Returns:
        pandas.DataFrame: Test results
    """
    print("Testing model with sample employee data")
    
    # Create test data
    test_data = create_test_employees()
    
    # Make predictions
    try:
        probabilities = model.predict_proba(test_data)[:, 1]
        predictions = model.predict(test_data)
        
        # Add predictions to the test data
        test_data["enrollment_probability"] = probabilities
        test_data["predicted_enrollment"] = predictions
        
        # Print predictions
        print("\nPrediction Results:")
        for i, row in test_data.iterrows():
            print(f"Employee {i+1}:")
            print(f"  Age: {row['age']}, Gender: {row['gender']}")
            print(f"  Marital Status: {row['marital_status']}, Has Dependents: {row['has_dependents']}")
            print(f"  Salary: ${row['salary']}, Employment: {row['employment_type']}")
            print(f"  Region: {row['region']}, Tenure: {row['tenure_years']} years")
            print(f"  Enrollment Probability: {row['enrollment_probability']:.2f}")
            print(f"  Predicted Enrollment: {'Yes' if row['predicted_enrollment'] else 'No'}")
            print("")
        
        # Create visualization if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot the enrollment probabilities
            plt.figure(figsize=(12, 8))
            
            # Sort by probability for better visualization
            sorted_indices = np.argsort(test_data["enrollment_probability"])
            plt.barh(
                range(len(sorted_indices)), 
                test_data["enrollment_probability"].iloc[sorted_indices],
                color=['#FF9999' if p > 0.5 else '#9999FF' for p in test_data["enrollment_probability"].iloc[sorted_indices]]
            )
            
            # Add employee info as labels
            employee_labels = []
            for i in sorted_indices:
                row = test_data.iloc[i]
                label = f"Age: {row['age']}, {row['gender']}, {row['employment_type']}, ${row['salary']/1000:.0f}K"
                employee_labels.append(label)
            
            plt.yticks(range(len(sorted_indices)), employee_labels)
            plt.xlabel("Enrollment Probability", fontsize=14)
            plt.title("Predicted Enrollment Probabilities for Test Employees", fontsize=16)
            plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
            plt.legend()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(f"{output_dir}/test_predictions.png")
            print(f"Visualization saved to {output_dir}/test_predictions.png")
        
        return test_data
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise

def test_with_actual_data(model, data_file, output_dir=None):
    """
    Test the model with actual data file and evaluate performance
    
    Args:
        model: Trained model pipeline
        data_file (str): Path to the data file
        output_dir (str): Directory to save results (optional)
        
    Returns:
        dict: Performance metrics
    """
    print(f"Testing model with actual data from {data_file}")
    
    try:
        # Load the data
        df = pd.read_csv(data_file)
        
        # Split features and target
        X = df.drop(["employee_id", "enrolled"], axis=1, errors='ignore')
        y = df["enrolled"]
        
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Calculate performance metrics
        classification_rep = classification_report(y, y_pred, output_dict=True)
        conf_mat = confusion_matrix(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)
        
        print("\nModel Performance:")
        print(f"Accuracy: {classification_rep['accuracy']:.4f}")
        print(f"Precision: {classification_rep['1']['precision']:.4f}")
        print(f"Recall: {classification_rep['1']['recall']:.4f}")
        print(f"F1 Score: {classification_rep['1']['f1-score']:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Create visualizations if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                conf_mat, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Not Enrolled', 'Enrolled'],
                yticklabels=['Not Enrolled', 'Enrolled']
            )
            plt.xlabel('Predicted Label', fontsize=14)
            plt.ylabel('True Label', fontsize=14)
            plt.title('Confusion Matrix', fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/confusion_matrix.png")
            
            # Plot precision-recall curve
            from sklearn.metrics import precision_recall_curve
            precision, recall, thresholds = precision_recall_curve(y, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, marker='.', label='Model')
            plt.xlabel('Recall', fontsize=14)
            plt.ylabel('Precision', fontsize=14)
            plt.title('Precision-Recall Curve', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/precision_recall_curve.png")
            
            print(f"Result visualizations saved to {output_dir}/")
        
        return {
            "classification_report": classification_rep,
            "confusion_matrix": conf_mat,
            "roc_auc": roc_auc
        }
        
    except Exception as e:
        print(f"Error testing with actual data: {e}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test insurance enrollment prediction model')
    parser.add_argument('--model_path', type=str, default='output/models/GradientBoosting_pipeline.pkl', 
                        help='Path to the saved model file')
    parser.add_argument('--data_file', type=str, default='employee_data.csv', 
                        help='Path to the data file for evaluation (optional)')
    parser.add_argument('--output_dir', type=str, default='output/test_results', 
                        help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    
    # Test with sample data
    test_with_sample_data(model, args.output_dir)
    
    # Test with actual data if file exists
    if os.path.exists(args.data_file):
        test_with_actual_data(model, args.data_file, args.output_dir)
    else:
        print(f"Data file {args.data_file} not found. Skipping evaluation with actual data.")

if __name__ == "__main__":
    main()
