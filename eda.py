# eda.py
# Exploratory data analysis and visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_eda_visualizations(df, output_dir="figures"):
    """
    Create exploratory data analysis visualizations
    
    Args:
        df (pandas.DataFrame): Employee data
        output_dir (str): Directory to save the figures
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style for the plots
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Target Distribution
    plt.figure()
    ax = sns.countplot(x="enrolled", data=df)
    plt.title("Distribution of Enrollment Status", fontsize=16)
    plt.xlabel("Enrolled (1 = Yes, 0 = No)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    
    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = f"{100 * p.get_height() / total:.1f}%"
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/target_distribution.png")
    plt.close()
    
    # 2. Age Distribution and Analysis
    plt.figure(figsize=(15, 12))
    
    # Age histogram
    plt.subplot(2, 2, 1)
    sns.histplot(df["age"], kde=True, bins=20)
    plt.title("Age Distribution", fontsize=16)
    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    
    # Age by enrollment status
    plt.subplot(2, 2, 2)
    sns.boxplot(x="enrolled", y="age", data=df)
    plt.title("Age by Enrollment Status", fontsize=16)
    plt.xlabel("Enrolled (1 = Yes, 0 = No)", fontsize=14)
    plt.ylabel("Age", fontsize=14)
    
    # Age distribution by gender
    plt.subplot(2, 2, 3)
    sns.boxplot(x="gender", y="age", data=df)
    plt.title("Age Distribution by Gender", fontsize=16)
    plt.xlabel("Gender", fontsize=14)
    plt.ylabel("Age", fontsize=14)
    
    # Enrollment rate by age group
    plt.subplot(2, 2, 4)
    df['age_group'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 55, 65, 100], 
                              labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    enrollment_by_age = df.groupby('age_group')['enrolled'].mean() * 100
    enrollment_by_age.plot(kind='bar')
    plt.title("Enrollment Rate by Age Group", fontsize=16)
    plt.xlabel("Age Group", fontsize=14)
    plt.ylabel("Enrollment Rate (%)", fontsize=14)
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for i, v in enumerate(enrollment_by_age):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age_analysis.png")
    plt.close()
    
    # 3. Salary Analysis
    plt.figure(figsize=(15, 12))
    
    # Salary histogram
    plt.subplot(2, 2, 1)
    sns.histplot(df["salary"], kde=True, bins=20)
    plt.title("Salary Distribution", fontsize=16)
    plt.xlabel("Salary ($)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    
    # Salary by enrollment status
    plt.subplot(2, 2, 2)
    sns.boxplot(x="enrolled", y="salary", data=df)
    plt.title("Salary by Enrollment Status", fontsize=16)
    plt.xlabel("Enrolled (1 = Yes, 0 = No)", fontsize=14)
    plt.ylabel("Salary ($)", fontsize=14)
    
    # Salary by employment type
    plt.subplot(2, 2, 3)
    sns.boxplot(x="employment_type", y="salary", data=df)
    plt.title("Salary by Employment Type", fontsize=16)
    plt.xlabel("Employment Type", fontsize=14)
    plt.ylabel("Salary ($)", fontsize=14)
    plt.xticks(rotation=45)
    
    # Enrollment rate by salary band
    plt.subplot(2, 2, 4)
    df['salary_band'] = pd.qcut(df['salary'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    enrollment_by_salary = df.groupby('salary_band')['enrolled'].mean() * 100
    enrollment_by_salary.plot(kind='bar')
    plt.title("Enrollment Rate by Salary Band", fontsize=16)
    plt.xlabel("Salary Band", fontsize=14)
    plt.ylabel("Enrollment Rate (%)", fontsize=14)
    
    # Add percentage labels
    for i, v in enumerate(enrollment_by_salary):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/salary_analysis.png")
    plt.close()
    
    # 4. Categorical Variables Analysis
    categorical_cols = ["gender", "marital_status", "employment_type", "region", "has_dependents"]
    
    plt.figure(figsize=(15, 15))
    for i, col in enumerate(categorical_cols):
        plt.subplot(3, 2, i+1)
        
        # Calculate percentages
        percentage_enrolled = df.groupby(col)["enrolled"].mean() * 100
        
        # Plot
        percentage_enrolled.plot(kind="bar")
        plt.title(f"Enrollment Rate by {col.replace('_', ' ').title()}", fontsize=16)
        plt.ylabel("Enrollment Rate (%)", fontsize=14)
        plt.xlabel(col.replace('_', ' ').title(), fontsize=14)
        plt.xticks(rotation=45)
        
        # Add percentage labels
        for j, p in enumerate(percentage_enrolled):
            plt.text(j, p + 1, f"{p:.1f}%", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/categorical_analysis.png")
    plt.close()
    
    # 5. Tenure Analysis
    plt.figure(figsize=(15, 12))
    
    # Tenure histogram
    plt.subplot(2, 2, 1)
    sns.histplot(df["tenure_years"], kde=True, bins=20)
    plt.title("Tenure Distribution (Years)", fontsize=16)
    plt.xlabel("Tenure (Years)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    
    # Tenure by enrollment status
    plt.subplot(2, 2, 2)
    sns.boxplot(x="enrolled", y="tenure_years", data=df)
    plt.title("Tenure by Enrollment Status", fontsize=16)
    plt.xlabel("Enrolled (1 = Yes, 0 = No)", fontsize=14)
    plt.ylabel("Tenure (Years)", fontsize=14)
    
    # Tenure by employment type
    plt.subplot(2, 2, 3)
    sns.boxplot(x="employment_type", y="tenure_years", data=df)
    plt.title("Tenure by Employment Type", fontsize=16)
    plt.xlabel("Employment Type", fontsize=14)
    plt.ylabel("Tenure (Years)", fontsize=14)
    plt.xticks(rotation=45)
    
    # Enrollment rate by tenure (binned)
    plt.subplot(2, 2, 4)
    df["tenure_bin"] = pd.cut(df["tenure_years"], bins=[0, 1, 3, 5, 10, 20, 100], 
                             labels=["<1", "1-3", "3-5", "5-10", "10-20", "20+"])
    tenure_enrollment = df.groupby("tenure_bin")["enrolled"].mean() * 100
    tenure_enrollment.plot(kind="bar")
    plt.title("Enrollment Rate by Tenure Range", fontsize=16)
    plt.xlabel("Tenure Range (Years)", fontsize=14)
    plt.ylabel("Enrollment Rate (%)", fontsize=14)
    
    # Add percentage labels
    for i, v in enumerate(tenure_enrollment):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tenure_analysis.png")
    plt.close()
    
    # 6. Correlation Matrix
    plt.figure(figsize=(12, 10))
    
    # Select numerical columns
    numerical_cols = ["age", "salary", "tenure_years", "enrolled"]
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", annot_kws={"size": 12})
    plt.title("Correlation Matrix of Numerical Features", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # 7. Feature Interactions
    plt.figure(figsize=(15, 12))
    
    # Age and Salary by Enrollment
    plt.subplot(2, 2, 1)
    sns.scatterplot(x="age", y="salary", hue="enrolled", data=df, alpha=0.6)
    plt.title("Age vs Salary by Enrollment Status", fontsize=16)
    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Salary ($)", fontsize=14)
    
    # Tenure and Age by Enrollment
    plt.subplot(2, 2, 2)
    sns.scatterplot(x="tenure_years", y="age", hue="enrolled", data=df, alpha=0.6)
    plt.title("Tenure vs Age by Enrollment Status", fontsize=16)
    plt.xlabel("Tenure (Years)", fontsize=14)
    plt.ylabel("Age", fontsize=14)
    
    # Tenure and Salary by Enrollment
    plt.subplot(2, 2, 3)
    sns.scatterplot(x="tenure_years", y="salary", hue="enrolled", data=df, alpha=0.6)
    plt.title("Tenure vs Salary by Enrollment Status", fontsize=16)
    plt.xlabel("Tenure (Years)", fontsize=14)
    plt.ylabel("Salary ($)", fontsize=14)
    
    # Enrollment rate by marital status and dependents
    plt.subplot(2, 2, 4)
    
    # Create cross-tabulation of enrollment rate
    enrollment_by_marital_dependents = df.groupby(["marital_status", "has_dependents"])["enrolled"].mean() * 100
    enrollment_by_marital_dependents = enrollment_by_marital_dependents.unstack()
    
    # Plot heatmap
    sns.heatmap(enrollment_by_marital_dependents, annot=True, cmap="YlGnBu", fmt=".1f", annot_kws={"size": 12})
    plt.title("Enrollment Rate (%) by Marital Status and Dependents", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_interactions.png")
    plt.close()

    print(f"All EDA visualizations saved to {output_dir}/ directory")
    
    return df  # Return the dataframe with any added columns for further analysis
