# hyperparameter_tuning.py
# Hyperparameter tuning for insurance enrollment prediction models

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tune_logistic_regression(X_train, y_train, preprocessor, cv=5, n_jobs=-1, output_dir=None):
    """
    Tune hyperparameters for Logistic Regression
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        preprocessor (ColumnTransformer): Data preprocessing pipeline
        cv (int): Number of cross-validation folds
        n_jobs (int): Number of parallel jobs
        output_dir (str): Directory to save results visualization
        
    Returns:
        tuple: Best Pipeline, best parameters, and CV results
    """
    logger.info("Tuning Logistic Regression hyperparameters")
    
    # Create pipeline with preprocessor
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2', 'elasticnet', None],
        'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'model__class_weight': [None, 'balanced']
    }
    
    # Filter out invalid combinations
    valid_param_grid = []
    for penalty in param_grid['model__penalty']:
        for solver in param_grid['model__solver']:
            # Check validity of combination
            if (penalty == 'l1' and solver in ['newton-cg', 'sag']) or \
               (penalty == 'elasticnet' and solver != 'saga') or \
               (penalty is None and solver in ['liblinear']):
                continue
                
            for C in param_grid['model__C']:
                for class_weight in param_grid['model__class_weight']:
                    valid_param_grid.append({
                        'model__penalty': penalty,
                        'model__solver': solver,
                        'model__C': C,
                        'model__class_weight': class_weight
                    })
    
    # Define the scorer
    scorer = make_scorer(roc_auc_score)
    
    # Run grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=valid_param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    logger.info("Starting grid search for Logistic Regression...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best ROC AUC score: {grid_search.best_score_:.4f}")
    
    # Visualize results if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to DataFrame
        results = pd.DataFrame(grid_search.cv_results_)
        
        # Plot C parameter vs score for different penalties
        plt.figure(figsize=(12, 8))
        
        for penalty in ['l1', 'l2', 'elasticnet', None]:
            if penalty is None:
                penalty_mask = results['param_model__penalty'].isnull()
            else:
                penalty_mask = results['param_model__penalty'] == penalty
                
            if not any(penalty_mask):
                continue
                
            penalty_results = results[penalty_mask].copy()
            
            if not penalty_results.empty:
                # Convert C to numeric for proper plotting
                penalty_results['param_model__C'] = pd.to_numeric(penalty_results['param_model__C'])
                
                # Group by C and calculate mean score
                grouped = penalty_results.groupby('param_model__C')['mean_test_score'].mean().reset_index()
                
                # Plot
                plt.plot(grouped['param_model__C'], grouped['mean_test_score'], 
                         marker='o', label=f"penalty={penalty}")
        
        plt.xlabel('C parameter (log scale)', fontsize=14)
        plt.ylabel('Mean ROC AUC Score', fontsize=14)
        plt.title('Logistic Regression: C parameter vs ROC AUC Score by Penalty', fontsize=16)
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/logreg_tuning_results.png")
        plt.close()
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def tune_random_forest(X_train, y_train, preprocessor, cv=5, n_jobs=-1, output_dir=None):
    """
    Tune hyperparameters for Random Forest
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        preprocessor (ColumnTransformer): Data preprocessing pipeline
        cv (int): Number of cross-validation folds
        n_jobs (int): Number of parallel jobs
        output_dir (str): Directory to save results visualization
        
    Returns:
        tuple: Best Pipeline, best parameters, and CV results
    """
    logger.info("Tuning Random Forest hyperparameters")
    
    # Create pipeline with preprocessor
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter distributions for randomized search
    param_distributions = {
        'model__n_estimators': randint(50, 300),
        'model__max_depth': [None] + list(randint(5, 30).rvs(5)),
        'model__min_samples_split': randint(2, 20),
        'model__min_samples_leaf': randint(1, 10),
        'model__max_features': ['auto', 'sqrt', 'log2', None],
        'model__class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    # Define the scorer
    scorer = make_scorer(roc_auc_score)
    
    # Run randomized search first to narrow down the parameter space
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    logger.info("Starting randomized search for Random Forest...")
    random_search.fit(X_train, y_train)
    logger.info(f"Best parameters from randomized search: {random_search.best_params_}")
    logger.info(f"Best ROC AUC score from randomized search: {random_search.best_score_:.4f}")
    
    # Now define a more focused grid search around the best parameters
    best_params = random_search.best_params_
    
    # Create more focused parameter grid
    param_grid = {
        'model__n_estimators': [max(50, best_params['model__n_estimators'] - 50),
                              best_params['model__n_estimators'],
                              min(300, best_params['model__n_estimators'] + 50)],
        'model__max_depth': [best_params['model__max_depth']],
        'model__min_samples_split': [max(2, best_params['model__min_samples_split'] - 2),
                                   best_params['model__min_samples_split'],
                                   min(20, best_params['model__min_samples_split'] + 2)],
        'model__min_samples_leaf': [max(1, best_params['model__min_samples_leaf'] - 1),
                                  best_params['model__min_samples_leaf'],
                                  min(10, best_params['model__min_samples_leaf'] + 1)],
        'model__max_features': [best_params['model__max_features']],
        'model__class_weight': [best_params['model__class_weight']]
    }
    
    # Run grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    logger.info("Starting focused grid search for Random Forest...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters from grid search: {grid_search.best_params_}")
    logger.info(f"Best ROC AUC score from grid search: {grid_search.best_score_:.4f}")
    
    # Visualize results if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize randomized search results
        results = pd.DataFrame(random_search.cv_results_)
        
        # Plot n_estimators vs score
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different parameters
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='param_model__n_estimators', y='mean_test_score', data=results)
        plt.title('Effect of n_estimators on ROC AUC Score', fontsize=14)
        plt.xlabel('Number of Trees', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        
        plt.subplot(2, 2, 2)
        # Convert max_depth to string for categorical plotting
        results['param_model__max_depth_str'] = results['param_model__max_depth'].astype(str)
        sns.boxplot(x='param_model__max_depth_str', y='mean_test_score', data=results)
        plt.title('Effect of max_depth on ROC AUC Score', fontsize=14)
        plt.xlabel('Max Depth', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='param_model__min_samples_split', y='mean_test_score', data=results)
        plt.title('Effect of min_samples_split on ROC AUC Score', fontsize=14)
        plt.xlabel('Min Samples Split', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        
        plt.subplot(2, 2, 4)
        sns.boxplot(x='param_model__max_features', y='mean_test_score', data=results)
        plt.title('Effect of max_features on ROC AUC Score', fontsize=14)
        plt.xlabel('Max Features', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rf_randomized_search_results.png")
        plt.close()
        
        # Visualize grid search results
        grid_results = pd.DataFrame(grid_search.cv_results_)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='param_model__n_estimators', y='mean_test_score', data=grid_results, marker='o')
        plt.title('Random Forest: Number of Trees vs ROC AUC Score (Focused Grid Search)', fontsize=16)
        plt.xlabel('Number of Trees', fontsize=14)
        plt.ylabel('ROC AUC Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rf_grid_search_results.png")
        plt.close()
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def tune_gradient_boosting(X_train, y_train, preprocessor, cv=5, n_jobs=-1, output_dir=None):
    """
    Tune hyperparameters for Gradient Boosting
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        preprocessor (ColumnTransformer): Data preprocessing pipeline
        cv (int): Number of cross-validation folds
        n_jobs (int): Number of parallel jobs
        output_dir (str): Directory to save results visualization
        
    Returns:
        tuple: Best Pipeline, best parameters, and CV results
    """
    logger.info("Tuning Gradient Boosting hyperparameters")
    
    # Create pipeline with preprocessor
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=42))
    ])
    
    # Define parameter distributions for randomized search
    param_distributions = {
        'model__n_estimators': randint(50, 300),
        'model__learning_rate': uniform(0.01, 0.3),
        'model__max_depth': randint(2, 10),
        'model__min_samples_split': randint(2, 20),
        'model__min_samples_leaf': randint(1, 10),
        'model__subsample': uniform(0.6, 0.4)
    }
    
    # Define the scorer
    scorer = make_scorer(roc_auc_score)
    
    # Run randomized search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    logger.info("Starting randomized search for Gradient Boosting...")
    random_search.fit(X_train, y_train)
    logger.info(f"Best parameters from randomized search: {random_search.best_params_}")
    logger.info(f"Best ROC AUC score from randomized search: {random_search.best_score_:.4f}")
    
    # Define more focused parameter grid based on randomized search results
    best_params = random_search.best_params_
    
    # Create more focused parameter grid
    param_grid = {
        'model__n_estimators': [best_params['model__n_estimators']],
        'model__learning_rate': [max(0.01, best_params['model__learning_rate'] - 0.02),
                                best_params['model__learning_rate'],
                                min(0.3, best_params['model__learning_rate'] + 0.02)],
        'model__max_depth': [max(2, best_params['model__max_depth'] - 1),
                            best_params['model__max_depth'],
                            min(10, best_params['model__max_depth'] + 1)],
        'model__subsample': [best_params['model__subsample']]
    }
    
    # Run grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    logger.info("Starting focused grid search for Gradient Boosting...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters from grid search: {grid_search.best_params_}")
    logger.info(f"Best ROC AUC score from grid search: {grid_search.best_score_:.4f}")
    
    # Visualize results if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize randomized search results
        results = pd.DataFrame(random_search.cv_results_)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='param_model__n_estimators', y='mean_test_score', data=results)
        plt.title('Effect of n_estimators on ROC AUC Score', fontsize=14)
        plt.xlabel('Number of Trees', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='param_model__learning_rate', y='mean_test_score', data=results)
        plt.title('Effect of learning_rate on ROC AUC Score', fontsize=14)
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='param_model__max_depth', y='mean_test_score', data=results)
        plt.title('Effect of max_depth on ROC AUC Score', fontsize=14)
        plt.xlabel('Max Depth', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='param_model__subsample', y='mean_test_score', data=results)
        plt.title('Effect of subsample on ROC AUC Score', fontsize=14)
        plt.xlabel('Subsample Ratio', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gb_randomized_search_results.png")
        plt.close()
        
        # Visualize grid search results
        grid_results = pd.DataFrame(grid_search.cv_results_)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='param_model__learning_rate', y='mean_test_score', data=grid_results, marker='o')
        plt.title('Gradient Boosting: Learning Rate vs ROC AUC Score (Focused Grid Search)', fontsize=16)
        plt.xlabel('Learning Rate', fontsize=14)
        plt.ylabel('ROC AUC Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gb_grid_search_results.png")
        plt.close()
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def tune_all_models(X_train, y_train, preprocessor, cv=5, n_jobs=-1, output_dir=None):
    """
    Tune hyperparameters for all models
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        preprocessor (ColumnTransformer): Data preprocessing pipeline
        cv (int): Number of cross-validation folds
        n_jobs (int): Number of parallel jobs
        output_dir (str): Directory to save results visualization
        
    Returns:
        dict: Dictionary with tuned models and their parameters
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Tune each model
    lr_model, lr_params, lr_results = tune_logistic_regression(
        X_train, y_train, preprocessor, cv, n_jobs, output_dir
    )
    
    rf_model, rf_params, rf_results = tune_random_forest(
        X_train, y_train, preprocessor, cv, n_jobs, output_dir
    )
    
    gb_model, gb_params, gb_results = tune_gradient_boosting(
        X_train, y_train, preprocessor, cv, n_jobs, output_dir
    )
    
    # Compare the best models
    if output_dir:
        models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
        best_scores = [
            max(pd.DataFrame(lr_results)['mean_test_score']),
            max(pd.DataFrame(rf_results)['mean_test_score']),
            max(pd.DataFrame(gb_results)['mean_test_score'])
        ]
        
        plt.figure(figsize=(10, 6))
        colors = ['#FF9999', '#99FF99', '#9999FF']
        plt.bar(models, best_scores, color=colors)
        plt.title('Best ROC AUC Score by Model after Hyperparameter Tuning', fontsize=16)
        plt.ylabel('ROC AUC Score', fontsize=14)
        plt.ylim(min(best_scores) - 0.05, 1.0)
        
        for i, score in enumerate(best_scores):
            plt.text(i, score + 0.01, f"{score:.4f}", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison.png")
        plt.close()
    
    # Return results
    return {
        "LogisticRegression": {
            "model": lr_model,
            "best_params": lr_params,
            "cv_results": lr_results
        },
        "RandomForest": {
            "model": rf_model,
            "best_params": rf_params,
            "cv_results": rf_results
        },
        "GradientBoosting": {
            "model": gb_model,
            "best_params": gb_params,
            "cv_results": gb_results
        }
    }
