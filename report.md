# Insurance Enrollment Prediction: Project Report

## 1. Executive Summary

This project develops a machine learning pipeline to predict employee enrollment in a voluntary insurance product using demographic and employment-related data. We built and evaluated multiple classification models, including Logistic Regression, Random Forest, and Gradient Boosting.

**Key Findings:**
- Gradient Boosting and Random Forest achieved near-perfect performance, with Gradient Boosting selected as the best model based on a weighted score (Accuracy, Precision, Recall, F1, ROC AUC all 1.0000).
- Key predictors of enrollment include salary, age, employment type, and tenure.
- Employees with dependents and those with higher salaries showed higher enrollment rates.
- The top models achieve near-perfect predictive performance (Accuracy, Precision, Recall, F1, ROC AUC all close to 1.0000), suggesting the task might be simpler than initially anticipated or potential data leakage.

The solution includes an end-to-end pipeline with data preprocessing, visualization, model training, evaluation, and a REST API for serving predictions.

## 2. Data Observations

### 2.1 Data Overview

The dataset consists of approximately 10,000 employee records with the following features:
- `employee_id`: Unique identifier
- `age`: Employee age in years
- `gender`: Employee gender
- `marital_status`: Marital status
- `salary`: Annual salary in USD
- `employment_type`: Type of employment (e.g., Full-time, Part-time)
- `region`: Geographic region
- `has_dependents`: Whether the employee has dependents
- `tenure_years`: Years of employment at the company
- `enrolled`: Target variable (1 = enrolled, 0 = not enrolled)

### 2.2 Data Quality Assessment

The exploratory data analysis revealed:
- No missing values in the dataset
- No duplicate records
- Reasonable value ranges for all features
- Roughly balanced target distribution (40% enrolled, 60% not enrolled)

### 2.3 Feature Distributions and Patterns

#### Numerical Features:
- **Age**: Normally distributed with mean around 40 years
- **Salary**: Right-skewed with median around $75,000
- **Tenure**: Right-skewed with median around 4 years

#### Categorical Features:
- **Gender**: Relatively balanced distribution
- **Marital Status**: Majority married, followed by single
- **Employment Type**: Majority full-time
- **Region**: Relatively even distribution across regions
- **Has Dependents**: About 45% have dependents

### 2.4 Relationship with Target Variable

Key insights from our analysis:
- **Age**: Middle-aged employees (35-55) have higher enrollment rates
- **Salary**: Higher salary bands show increased enrollment likelihood
- **Tenure**: Employees with longer tenure (>5 years) are more likely to enroll
- **Has Dependents**: Employees with dependents are significantly more likely to enroll (~60% vs ~30%)
- **Employment Type**: Full-time employees have higher enrollment rates than part-time or contract
- **Marital Status**: Married employees have higher enrollment rates than single employees

### 2.5 Feature Correlations

- Moderate positive correlation between age and tenure (0.45)
- Weak positive correlation between age and salary (0.25)
- Salary has the strongest correlation with the target variable among numerical features

## 3. Model Development and Evaluation

### 3.1 Preprocessing Pipeline

We implemented a robust preprocessing pipeline that includes:
- Handling of numerical features:
  - Imputation of missing values (median strategy)
  - Standardization (zero mean, unit variance)
- Handling of categorical features:
  - Imputation of missing values (most frequent strategy)
  - One-hot encoding with unknown value handling

### 3.2 Model Selection

We evaluated three different classification models:

1. **Logistic Regression**:
   - Simple, interpretable baseline model
   - Good for understanding feature importance through coefficients
   - Less prone to overfitting on small to medium datasets

2. **Random Forest**:
   - Ensemble method that reduces overfitting
   - Handles non-linear relationships and interactions well
   - Provides feature importance metrics

3. **Gradient Boosting**:
   - Powerful ensemble method that often achieves state-of-the-art performance
   - Strong ability to capture complex patterns
   - More prone to overfitting, requiring careful tuning

### 3.3 Model Comparison

The following table summarizes the performance of the models on the test set based on the initial pipeline run:

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.8965   | 0.9112    | 0.9223 | 0.9167   | 0.9705  |
| Random Forest      | 0.9995   | 0.9992    | 1.0000 | 0.9996   | 1.0000  |
| Gradient Boosting  | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000  |

Gradient Boosting and Random Forest significantly outperformed Logistic Regression, achieving near-perfect scores on the test set. Gradient Boosting was selected as the best model based on the weighted scoring criteria implemented in `main.py`.

### 3.4 Feature Importance

The top predictive features across models were:
1. Salary
2. Age
3. Tenure years
4. Has dependents (True)
5. Employment type (Full-time)

Feature importance was consistent across different models and evaluation methods (model-specific importance, permutation importance, and SHAP values).

### 3.5 Hyperparameter Tuning

Hyperparameter tuning was performed for all models using Randomized Search followed by focused Grid Search, optimizing for ROC AUC. The results were as follows:

- **Logistic Regression:** Best ROC AUC score during tuning was ~0.8918 with parameters `{'model__C': 0.01, 'model__class_weight': 'balanced', 'model__penalty': 'l2', 'model__solver': 'newton-cg'}`. *Note: Convergence warnings and fit failures (due to incompatible penalty/solver combinations) occurred during the grid search.*
- **Random Forest:** Best ROC AUC score during tuning was ~0.9998 with parameters `{'model__class_weight': None, 'model__max_depth': 15, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 4, 'model__min_samples_split': 6, 'model__n_estimators': 236}`. *Note: Fit failures occurred due to the use of `max_features='auto'`, which is deprecated and was included in the initial random search space.*
- **Gradient Boosting:** Achieved a perfect ROC AUC score of 1.0000 during tuning. Best parameters found were `{'model__learning_rate': 0.2346, 'model__max_depth': 2, 'model__n_estimators': 212, 'model__subsample': 0.9085}`.

The near-perfect scores achieved by Random Forest and Gradient Boosting, even before extensive tuning, suggest the dataset might contain highly predictive features or potential target leakage that should be investigated further.

## 4. Model Interpretation

### 4.1 Key Drivers of Enrollment

SHAP analysis revealed the following insights:
- Higher salary consistently increases enrollment probability
- Having dependents substantially increases enrollment likelihood
- Full-time employment is positively associated with enrollment
- Age has a non-linear relationship with enrollment, with middle-aged employees more likely to enroll
- Longer tenure generally increases enrollment probability

### 4.2 Business Implications

These findings suggest several strategies for increasing enrollment:
- Focus outreach efforts on part-time and contract employees
- Provide additional education about insurance benefits to younger employees
- Consider family-oriented messaging for employees without dependents
- Develop specific programs for new employees with shorter tenure

## 5. Model Deployment

We implemented a FastAPI-based REST API that allows:
- Real-time predictions for individual employees
- Feature importance for each prediction
- Health monitoring endpoints
- Clear documentation with Swagger UI

The API serves predictions with the following outputs:
- Enrollment probability (0-1)
- Binary prediction (will enroll or not)
- Feature importance for the specific prediction

## 6. Next Steps and Future Work

With more time, we would explore:

### 6.1 Data Enhancement
- Collect additional features like education level, job role, and previous insurance history
- Explore interaction terms between features (e.g., age × has_dependents)
- Gather time-series data to model enrollment patterns over time

### 6.2 Model Improvements
- Test additional models like XGBoost, LightGBM, and neural networks
- Implement automated feature selection techniques
- Create ensemble models combining multiple algorithms
- Address any class imbalance with techniques like SMOTE or class weighting

### 6.3 Deployment Enhancements
- Implement model monitoring and retraining pipeline
- Add batch prediction capability to the API
- Create a simple front-end interface for non-technical users
- Set up CI/CD for model updates

### 6.4 Business Integration
- Develop a recommendation system for personalized insurance options
- Create dashboards for HR teams to track enrollment trends
- Integrate the model with existing HR systems

## 7. Conclusion

The developed machine learning pipeline successfully predicts employee insurance enrollment with extremely high accuracy. Gradient Boosting and Random Forest demonstrated near-perfect performance, with Gradient Boosting selected based on our holistic evaluation criteria.

The solution includes comprehensive data processing, visualization, model development, and deployment components. The REST API allows easy integration with existing systems, enabling automated predictions for new employees. The exceptionally high performance warrants further investigation into potential data leakage or the simplicity of the underlying prediction task before full production deployment.

By understanding the key factors that influence enrollment decisions, the company can develop targeted strategies to increase participation in their insurance program, ultimately providing better benefits to employees while optimizing business outcomes.
