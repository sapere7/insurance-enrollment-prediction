# eda.py
# Generates exploratory data analysis visualizations for the employee dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging 
import datetime 
from typing import Optional

# --- Logging Setup ---
def setup_logging(log_dir: str = "logs/analysis", level: int = logging.INFO) -> logging.Logger: 
    """Configures logging to console and a timestamped file for the EDA script."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"eda_run_{timestamp}.log")

    # Configure root logger specifically for this script run
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger("eda_script") 
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

    logger.info(f"EDA script logging configured. Log file: {log_file}")
    return logger

logger = setup_logging() 

# --- EDA Function ---
def create_eda_visualizations(df: pd.DataFrame, output_dir: str = "figures") -> Optional[pd.DataFrame]:
    """
    Generates and saves various EDA plots for the provided DataFrame.

    Args:
        df: Input DataFrame containing employee data.
        output_dir: Directory where the generated plots will be saved.

    Returns:
        The input DataFrame, potentially with added temporary columns used for plotting (e.g., age_group), 
        or None if an error occurs during plotting.
    """
    try:
        logger.info(f"Starting EDA visualization generation. Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Plotting Setup ---
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8) # Default figure size
        
        # --- Plot 1: Target Distribution ---
        logger.debug("Generating target distribution plot...")
        plt.figure()
        ax = sns.countplot(x="enrolled", data=df)
        plt.title("Distribution of Enrollment Status", fontsize=16)
        plt.xlabel("Enrolled (1 = Yes, 0 = No)", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        
        total = len(df)
        for p in ax.patches:
            percentage = f"{100 * p.get_height() / total:.1f}%"
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "target_distribution.png"))
        plt.close()
        
        # --- Plot 2: Age Analysis ---
        logger.debug("Generating age analysis plots...")
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 2, 1)
        sns.histplot(df["age"], kde=True, bins=20)
        plt.title("Age Distribution", fontsize=16)
        plt.xlabel("Age", fontsize=14); plt.ylabel("Count", fontsize=14)
        
        plt.subplot(2, 2, 2)
        sns.boxplot(x="enrolled", y="age", data=df)
        plt.title("Age by Enrollment Status", fontsize=16)
        plt.xlabel("Enrolled (1 = Yes, 0 = No)", fontsize=14); plt.ylabel("Age", fontsize=14)
        
        plt.subplot(2, 2, 3)
        sns.boxplot(x="gender", y="age", data=df)
        plt.title("Age Distribution by Gender", fontsize=16)
        plt.xlabel("Gender", fontsize=14); plt.ylabel("Age", fontsize=14)
        
        plt.subplot(2, 2, 4)
        # Define age bins and labels (could be moved to config)
        age_bins = [18, 25, 35, 45, 55, 65, 100] 
        age_labels = [f'{age_bins[i]}-{age_bins[i+1]-1}' for i in range(len(age_bins)-1)]
        age_labels[-1] = f'{age_bins[-2]}+' 
        
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False) 
        enrollment_by_age = df.groupby('age_group', observed=False)['enrolled'].mean() * 100 
        enrollment_by_age.plot(kind='bar')
        plt.title("Enrollment Rate by Age Group", fontsize=16)
        plt.xlabel("Age Group", fontsize=14); plt.ylabel("Enrollment Rate (%)", fontsize=14)
        plt.xticks(rotation=45)
        
        for i, v in enumerate(enrollment_by_age):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "age_analysis.png"))
        plt.close()
        
        # --- Plot 3: Salary Analysis ---
        logger.debug("Generating salary analysis plots...")
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 2, 1)
        sns.histplot(df["salary"], kde=True, bins=20)
        plt.title("Salary Distribution", fontsize=16)
        plt.xlabel("Salary ($)", fontsize=14); plt.ylabel("Count", fontsize=14)
        
        plt.subplot(2, 2, 2)
        sns.boxplot(x="enrolled", y="salary", data=df)
        plt.title("Salary by Enrollment Status", fontsize=16)
        plt.xlabel("Enrolled (1 = Yes, 0 = No)", fontsize=14); plt.ylabel("Salary ($)", fontsize=14)
        
        plt.subplot(2, 2, 3)
        sns.boxplot(x="employment_type", y="salary", data=df)
        plt.title("Salary by Employment Type", fontsize=16)
        plt.xlabel("Employment Type", fontsize=14); plt.ylabel("Salary ($)", fontsize=14)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        # Use quantiles for salary bands
        try:
            df['salary_band'] = pd.qcut(df['salary'], q=5, labels=['Lowest', 'Low', 'Medium', 'High', 'Highest'], duplicates='drop')
        except ValueError: 
            logger.warning("Could not create unique quantile bins for salary, using equal-width bins instead.")
            df['salary_band'] = pd.cut(df['salary'], bins=5, labels=['Lowest', 'Low', 'Medium', 'High', 'Highest'])
            
        enrollment_by_salary = df.groupby('salary_band', observed=False)['enrolled'].mean() * 100 
        enrollment_by_salary.plot(kind='bar')
        plt.title("Enrollment Rate by Salary Band (Quintiles)", fontsize=16)
        plt.xlabel("Salary Band", fontsize=14); plt.ylabel("Enrollment Rate (%)", fontsize=14)
        
        for i, v in enumerate(enrollment_by_salary):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "salary_analysis.png"))
        plt.close()
        
        # --- Plot 4: Categorical Variables Analysis ---
        logger.debug("Generating categorical analysis plots...")
        categorical_cols = ["gender", "marital_status", "employment_type", "region", "has_dependents"]
        
        plt.figure(figsize=(15, 15))
        for i, col in enumerate(categorical_cols):
            plt.subplot(3, 2, i+1)
            
            if col in df.columns:
                percentage_enrolled = df.groupby(col, observed=False)["enrolled"].mean() * 100 
            else:
                logger.warning(f"Column '{col}' not found for categorical analysis plot.")
                continue 
                
            percentage_enrolled.plot(kind="bar")
            plt.title(f"Enrollment Rate by {col.replace('_', ' ').title()}", fontsize=16)
            plt.ylabel("Enrollment Rate (%)", fontsize=14)
            plt.xlabel(col.replace('_', ' ').title(), fontsize=14)
            plt.xticks(rotation=45)
            
            for j, p in enumerate(percentage_enrolled):
                plt.text(j, p + 1, f"{p:.1f}%", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "categorical_analysis.png"))
        plt.close()
        
        # --- Plot 5: Tenure Analysis ---
        logger.debug("Generating tenure analysis plots...")
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 2, 1)
        sns.histplot(df["tenure_years"], kde=True, bins=20)
        plt.title("Tenure Distribution (Years)", fontsize=16)
        plt.xlabel("Tenure (Years)", fontsize=14); plt.ylabel("Count", fontsize=14)
        
        plt.subplot(2, 2, 2)
        sns.boxplot(x="enrolled", y="tenure_years", data=df)
        plt.title("Tenure by Enrollment Status", fontsize=16)
        plt.xlabel("Enrolled (1 = Yes, 0 = No)", fontsize=14); plt.ylabel("Tenure (Years)", fontsize=14)
        
        plt.subplot(2, 2, 3)
        sns.boxplot(x="employment_type", y="tenure_years", data=df)
        plt.title("Tenure by Employment Type", fontsize=16)
        plt.xlabel("Employment Type", fontsize=14); plt.ylabel("Tenure (Years)", fontsize=14)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        # Define tenure bins and labels (could be moved to config)
        tenure_bins = [0, 1, 3, 5, 10, 20, 100] 
        tenure_labels = [f'{tenure_bins[i]}-{tenure_bins[i+1]-1}' for i in range(len(tenure_bins)-1)]
        tenure_labels[0] = '<1' 
        tenure_labels[-1] = f'{tenure_bins[-2]}+' 
        
        df["tenure_bin"] = pd.cut(df["tenure_years"], bins=tenure_bins, labels=tenure_labels, right=False) 
        tenure_enrollment = df.groupby("tenure_bin", observed=False)["enrolled"].mean() * 100 
        tenure_enrollment.plot(kind="bar")
        plt.title("Enrollment Rate by Tenure Range", fontsize=16)
        plt.xlabel("Tenure Range (Years)", fontsize=14); plt.ylabel("Enrollment Rate (%)", fontsize=14)
        
        for i, v in enumerate(tenure_enrollment):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tenure_analysis.png"))
        plt.close()
        
        # --- Plot 6: Correlation Matrix ---
        logger.debug("Generating correlation matrix plot...")
        plt.figure(figsize=(12, 10))
        
        numerical_cols_corr = ["age", "salary", "tenure_years", "enrolled"]
        # Ensure only existing numerical columns are used
        numerical_cols_corr = [col for col in numerical_cols_corr if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numerical_cols_corr) > 1:
            corr_matrix = df[numerical_cols_corr].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", annot_kws={"size": 12})
            plt.title("Correlation Matrix of Numerical Features", fontsize=16)
        else:
            logger.warning("Not enough numerical columns found to generate correlation matrix.")
            plt.text(0.5, 0.5, "Not enough numerical data for correlation matrix", ha='center', va='center')
            plt.title("Correlation Matrix (Insufficient Data)", fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()
        
        # --- Plot 7: Feature Interactions ---
        logger.debug("Generating feature interaction plots...")
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 2, 1)
        sns.scatterplot(x="age", y="salary", hue="enrolled", data=df, alpha=0.6)
        plt.title("Age vs Salary by Enrollment Status", fontsize=16)
        plt.xlabel("Age", fontsize=14); plt.ylabel("Salary ($)", fontsize=14)
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(x="tenure_years", y="age", hue="enrolled", data=df, alpha=0.6)
        plt.title("Tenure vs Age by Enrollment Status", fontsize=16)
        plt.xlabel("Tenure (Years)", fontsize=14); plt.ylabel("Age", fontsize=14)
        
        plt.subplot(2, 2, 3)
        sns.scatterplot(x="tenure_years", y="salary", hue="enrolled", data=df, alpha=0.6)
        plt.title("Tenure vs Salary by Enrollment Status", fontsize=16)
        plt.xlabel("Tenure (Years)", fontsize=14); plt.ylabel("Salary ($)", fontsize=14)
        
        plt.subplot(2, 2, 4)
        if "marital_status" in df.columns and "has_dependents" in df.columns:
            enrollment_by_marital_dependents = df.groupby(["marital_status", "has_dependents"], observed=False)["enrolled"].mean() * 100
            enrollment_by_marital_dependents = enrollment_by_marital_dependents.unstack()
            if not enrollment_by_marital_dependents.empty:
                sns.heatmap(enrollment_by_marital_dependents, annot=True, cmap="YlGnBu", fmt=".1f", annot_kws={"size": 12})
                plt.title("Enrollment Rate (%) by Marital Status and Dependents", fontsize=16)
            else:
                 plt.text(0.5, 0.5, "No data for Marital/Dependents heatmap", ha='center', va='center')
                 plt.title("Enrollment Rate (%) by Marital Status and Dependents", fontsize=16)
        else:
            logger.warning("Columns 'marital_status' or 'has_dependents' not found for interaction heatmap.")
            plt.text(0.5, 0.5, "Columns missing for Marital/Dependents heatmap", ha='center', va='center')
            plt.title("Enrollment Rate (%) by Marital Status and Dependents", fontsize=16)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, "feature_interactions.png")
        plt.savefig(save_path)
        plt.close()

        logger.info(f"All EDA visualizations saved to {output_dir}/ directory")
        
        # Return df only if needed downstream, otherwise return None
        # Returning df with added columns might cause issues if not handled
        return None 

    except Exception as e:
        logger.error(f"An error occurred during EDA visualization: {e}", exc_info=True)
        return None


# --- Standalone Execution ---
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Generate EDA visualizations.')
    parser.add_argument('--data_file', type=str, default='employee_data.csv', help='Path to the input CSV data file.')
    parser.add_argument('--output_dir', type=str, default='output/figures', help='Directory to save figures.')
    
    args = parser.parse_args()

    logger.info(f"--- Starting Standalone EDA Run ---")
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        df = pd.read_csv(args.data_file)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        create_eda_visualizations(df, args.output_dir)
        logger.info(f"--- Standalone EDA Run Finished ---")
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data_file}")
    except Exception as e:
        logger.error(f"An error occurred during EDA: {e}", exc_info=True)
