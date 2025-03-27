# hyperparameter_tuning.py
# Performs hyperparameter tuning for specified models using configuration settings.

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.impute import SimpleImputer 
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml 
import joblib 
import datetime 
from scipy.stats import randint, uniform 
from typing import Dict, Any, Tuple, Optional 

# --- Logging Setup ---
def setup_logging(log_dir: str = "logs/tuning", level: int = logging.INFO) -> logging.Logger: 
    """Configures logging to console and a timestamped file for the tuning script."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tuning_run_{timestamp}.log")

    # Remove existing handlers to avoid duplication if re-run
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure root logger
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format,
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler() 
                        ])
    logger = logging.getLogger(__name__) 
    logger.info(f"Logging configured. Log file: {log_file}")
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

# --- Data Loading ---
def load_processed_data(processed_data_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads the processed training data splits (X_train, y_train) saved by main.py."""
    logger.info(f"Loading processed training data from {processed_data_dir}")
    try:
        X_train_path = os.path.join(processed_data_dir, 'X_train.pkl')
        y_train_path = os.path.join(processed_data_dir, 'y_train.pkl')
        X_train = pd.read_pickle(X_train_path)
        y_train = pd.read_pickle(y_train_path)
        logger.info("Processed training data loaded successfully.")
        return X_train, y_train
    except FileNotFoundError as e:
        logger.error(f"Error loading processed data file: {e}. Ensure main.py was run first.")
        raise
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise

# --- Preprocessor Definition ---
def define_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Defines the preprocessing steps (imputation, scaling, encoding) based on training data columns.
    Note: This definition should match the preprocessor used in main.py.
    """
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"Preprocessor: Found {len(numerical_cols)} numerical, {len(categorical_cols)} categorical columns.")

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep unspecified columns
    )
    return preprocessor


# --- Tuning Functions ---
def tune_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                             preprocessor: ColumnTransformer, config: Dict[str, Any]) -> Tuple[Optional[Pipeline], Dict, Dict]:
    """
    Tunes Logistic Regression using GridSearchCV with parameters from config.

    Args:
        X_train: Training features.
        y_train: Training target.
        preprocessor: Preprocessing pipeline definition (ColumnTransformer).
        config: Configuration dictionary.

    Returns:
        Tuple containing the best fitted pipeline, best parameters, and CV results dictionary. 
        Returns (None, {}, {}) if tuning is skipped or fails.
    """
    logger.info("Tuning Logistic Regression hyperparameters...")

    tuning_config = config.get('tuning', {})
    logreg_config = tuning_config.get('logreg', {})
    param_grid = logreg_config.get('param_grid', {}) 
    cv = tuning_config.get('cv_folds', 5)
    scoring = tuning_config.get('scoring_metric', 'roc_auc')
    n_jobs = tuning_config.get('n_jobs', -1)
    tuning_dir = config.get('tuning_dir', 'output/tuning')
    random_state = config.get('random_state', 42)

    if not param_grid:
        logger.warning("No parameter grid found for Logistic Regression in config. Skipping tuning.")
        return None, {}, {}

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(random_state=random_state, max_iter=config.get('models',{}).get('LogisticRegression',{}).get('max_iter', 5000))) 
    ])

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid, 
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1, 
        return_train_score=True,
        error_score='raise' 
    )
    
    try:
        logger.info("Starting grid search for Logistic Regression...")
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best {scoring} score: {grid_search.best_score_:.4f}")
        
        # Visualize results
        os.makedirs(tuning_dir, exist_ok=True)
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        plt.figure(figsize=(12, 8))
        results_df['param_model__C'] = pd.to_numeric(results_df['param_model__C'])
        results_df['solver_penalty'] = results_df['param_model__solver'].astype(str) + "_" + results_df['param_model__penalty'].fillna('none').astype(str)

        sns.lineplot(data=results_df, x='param_model__C', y='mean_test_score', hue='solver_penalty', marker='o', errorbar=None) 
        plt.xscale('log')
        plt.xlabel('C parameter (log scale)', fontsize=14)
        plt.ylabel(f'Mean {scoring.upper()} Score', fontsize=14)
        plt.title('Logistic Regression Tuning: C vs Score', fontsize=16)
        plt.legend(title='Solver_Penalty', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(tuning_dir, "logreg_tuning_results.png"))
        plt.close()

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

    except ValueError as e:
        logger.error(f"Grid search failed for Logistic Regression: {e}")
        logger.error("Check config.yaml for valid Logistic Regression solver/penalty combinations.")
        return None, {}, {}


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                         preprocessor: ColumnTransformer, config: Dict[str, Any]) -> Tuple[Optional[Pipeline], Dict, Dict]:
    """
    Tunes Random Forest using RandomizedSearchCV followed by GridSearchCV with parameters from config.

    Args:
        X_train: Training features.
        y_train: Training target.
        preprocessor: Preprocessing pipeline definition.
        config: Configuration dictionary.

    Returns:
        Tuple containing the best fitted pipeline, best parameters, and CV results dictionary. 
        Returns (None, {}, {}) if tuning is skipped or fails.
    """
    logger.info("Tuning Random Forest hyperparameters...")

    tuning_config = config.get('tuning', {})
    rf_config = tuning_config.get('rf', {})
    param_dist = rf_config.get('param_dist', {}) 
    param_grid_focused = rf_config.get('focused_grid', {}) 
    cv = tuning_config.get('cv_folds', 5)
    scoring = tuning_config.get('scoring_metric', 'roc_auc')
    n_iter = tuning_config.get('random_search_iterations', 50)
    n_jobs = tuning_config.get('n_jobs', -1)
    tuning_dir = config.get('tuning_dir', 'output/tuning')
    random_state = config.get('random_state', 42)

    if not param_dist:
        logger.warning("No parameter distribution found for Random Forest random search in config. Skipping tuning.")
        return None, {}, {}

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=random_state))
    ])

    # --- Randomized Search ---
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist, 
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state,
        return_train_score=True,
        error_score='raise'
    )
    
    try:
        logger.info("Starting randomized search for Random Forest...")
        random_search.fit(X_train, y_train)
        logger.info(f"Best parameters from randomized search: {random_search.best_params_}")
        logger.info(f"Best {scoring} score from randomized search: {random_search.best_score_:.4f}")
    except ValueError as e:
         logger.error(f"Randomized search failed for Random Forest: {e}")
         return None, {}, {} 

    # --- Focused Grid Search ---
    if not param_grid_focused:
        logger.warning("No 'focused_grid' defined for Random Forest in config. Using best model from random search.")
        best_estimator = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        final_cv_results = random_search.cv_results_ 
    else:
        logger.info("Starting focused grid search for Random Forest...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grid=param_grid_focused, 
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
            error_score='raise'
        )
    
        try:
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters from grid search: {grid_search.best_params_}")
            logger.info(f"Best {scoring} score from grid search: {grid_search.best_score_:.4f}")
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            final_cv_results = grid_search.cv_results_
        except ValueError as e:
             logger.error(f"Focused grid search failed for Random Forest: {e}")
             logger.warning("Using best model from random search instead due to grid search failure.")
             best_estimator = random_search.best_estimator_
             best_params = random_search.best_params_
             best_score = random_search.best_score_
             final_cv_results = random_search.cv_results_ 

    # --- Visualize Tuning Results ---
    os.makedirs(tuning_dir, exist_ok=True)
    try:
        # Visualize randomized search results
        random_results_df = pd.DataFrame(random_search.cv_results_)
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='param_model__n_estimators', y='mean_test_score', data=random_results_df)
        plt.title('Random Search: n_estimators vs Score', fontsize=14)
        plt.xlabel('Number of Trees', fontsize=12); plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        
        plt.subplot(2, 2, 2)
        random_results_df['param_model__max_depth_str'] = random_results_df['param_model__max_depth'].fillna('None').astype(str)
        sns.boxplot(x='param_model__max_depth_str', y='mean_test_score', data=random_results_df)
        plt.title('Random Search: max_depth vs Score', fontsize=14)
        plt.xlabel('Max Depth', fontsize=12); plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='param_model__min_samples_split', y='mean_test_score', data=random_results_df)
        plt.title('Random Search: min_samples_split vs Score', fontsize=14)
        plt.xlabel('Min Samples Split', fontsize=12); plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        
        plt.subplot(2, 2, 4)
        random_results_df['param_model__max_features_str'] = random_results_df['param_model__max_features'].fillna('None').astype(str)
        sns.boxplot(x='param_model__max_features_str', y='mean_test_score', data=random_results_df)
        plt.title('Random Search: max_features vs Score', fontsize=14)
        plt.xlabel('Max Features', fontsize=12); plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(tuning_dir, "rf_randomized_search_results.png"))
        plt.close()

        # Visualize focused grid search results if applicable
        if 'grid_search' in locals() and hasattr(grid_search, 'cv_results_'):
            grid_results_df = pd.DataFrame(grid_search.cv_results_)
            focus_param_keys = list(param_grid_focused.keys())
            if focus_param_keys:
                focus_param = focus_param_keys[0] 
                if len(param_grid_focused[focus_param]) > 1: 
                     plt.figure(figsize=(10, 6))
                     try:
                         grid_results_df[f'param_{focus_param}'] = pd.to_numeric(grid_results_df[f'param_{focus_param}'])
                         sns.lineplot(x=f'param_{focus_param}', y='mean_test_score', data=grid_results_df, marker='o', errorbar=None)
                     except ValueError: 
                         sns.boxplot(x=f'param_{focus_param}', y='mean_test_score', data=grid_results_df)
                         
                     plt.title(f'Focused Grid Search: {focus_param.split("__")[-1]} vs Score', fontsize=16)
                     plt.xlabel(focus_param.split('__')[-1], fontsize=14)
                     plt.ylabel(f'{scoring.upper()} Score', fontsize=14)
                     plt.grid(True, alpha=0.3)
                     plt.tight_layout()
                     plt.savefig(os.path.join(tuning_dir, "rf_grid_search_results.png"))
                     plt.close()

    except Exception as plot_err:
        logger.warning(f"Could not generate tuning plots for Random Forest: {plot_err}")

    return best_estimator, best_params, final_cv_results


def tune_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series, 
                             preprocessor: ColumnTransformer, config: Dict[str, Any]) -> Tuple[Optional[Pipeline], Dict, Dict]:
    """
    Tunes Gradient Boosting using RandomizedSearchCV followed by GridSearchCV with parameters from config.

     Args:
        X_train: Training features.
        y_train: Training target.
        preprocessor: Preprocessing pipeline definition.
        config: Configuration dictionary.

    Returns:
        Tuple containing the best fitted pipeline, best parameters, and CV results dictionary. 
        Returns (None, {}, {}) if tuning is skipped or fails.
    """
    logger.info("Tuning Gradient Boosting hyperparameters...")

    # Get relevant configuration settings
    tuning_config = config.get('tuning', {})
    gb_config = tuning_config.get('gb', {})
    param_dist = gb_config.get('param_dist', {}) 
    param_grid_focused = gb_config.get('focused_grid', {}) 
    cv = tuning_config.get('cv_folds', 5)
    scoring = tuning_config.get('scoring_metric', 'roc_auc')
    n_iter = tuning_config.get('random_search_iterations', 50)
    n_jobs = tuning_config.get('n_jobs', -1)
    tuning_dir = config.get('tuning_dir', 'output/tuning')
    random_state = config.get('random_state', 42)

    if not param_dist:
        logger.warning("No parameter distribution found for Gradient Boosting random search in config. Skipping tuning.")
        return None, {}, {}

    # Create pipeline with preprocessor and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=random_state))
    ])

    # --- Randomized Search ---
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist, 
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state,
        return_train_score=True,
        error_score='raise'
    )
    
    try:
        logger.info("Starting randomized search for Gradient Boosting...")
        random_search.fit(X_train, y_train)
        logger.info(f"Best parameters from randomized search: {random_search.best_params_}")
        logger.info(f"Best {scoring} score from randomized search: {random_search.best_score_:.4f}")
    except ValueError as e:
         logger.error(f"Randomized search failed for Gradient Boosting: {e}")
         return None, {}, {} 

    # --- Focused Grid Search ---
    if not param_grid_focused:
        logger.warning("No 'focused_grid' defined for Gradient Boosting in config. Using best model from random search.")
        best_estimator = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        final_cv_results = random_search.cv_results_
    else:
        logger.info("Starting focused grid search for Gradient Boosting...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid_focused, 
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
            error_score='raise'
        )
    
        try:
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters from grid search: {grid_search.best_params_}")
            logger.info(f"Best {scoring} score from grid search: {grid_search.best_score_:.4f}")
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            final_cv_results = grid_search.cv_results_
        except ValueError as e:
             logger.error(f"Focused grid search failed for Gradient Boosting: {e}")
             logger.warning("Using best model from random search instead due to grid search failure.")
             best_estimator = random_search.best_estimator_
             best_params = random_search.best_params_
             best_score = random_search.best_score_
             final_cv_results = random_search.cv_results_

    # --- Visualize Tuning Results ---
    os.makedirs(tuning_dir, exist_ok=True)
    try:
        # Visualize randomized search results
        random_results_df = pd.DataFrame(random_search.cv_results_)
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='param_model__n_estimators', y='mean_test_score', data=random_results_df)
        plt.title('Random Search: n_estimators vs Score', fontsize=14)
        plt.xlabel('Number of Trees', fontsize=12); plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='param_model__learning_rate', y='mean_test_score', data=random_results_df)
        plt.title('Random Search: learning_rate vs Score', fontsize=14)
        plt.xlabel('Learning Rate', fontsize=12); plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='param_model__max_depth', y='mean_test_score', data=random_results_df)
        plt.title('Random Search: max_depth vs Score', fontsize=14)
        plt.xlabel('Max Depth', fontsize=12); plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='param_model__subsample', y='mean_test_score', data=random_results_df)
        plt.title('Random Search: subsample vs Score', fontsize=14)
        plt.xlabel('Subsample Ratio', fontsize=12); plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(tuning_dir, "gb_randomized_search_results.png"))
        plt.close()

        # Visualize focused grid search results if applicable
        if 'grid_search' in locals() and hasattr(grid_search, 'cv_results_'):
            grid_results_df = pd.DataFrame(grid_search.cv_results_)
            focus_param_keys = list(param_grid_focused.keys())
            if focus_param_keys:
                focus_param = focus_param_keys[0]
                if len(param_grid_focused[focus_param]) > 1:
                     plt.figure(figsize=(10, 6))
                     try:
                         grid_results_df[f'param_{focus_param}'] = pd.to_numeric(grid_results_df[f'param_{focus_param}'])
                         sns.lineplot(x=f'param_{focus_param}', y='mean_test_score', data=grid_results_df, marker='o', errorbar=None)
                     except ValueError: 
                         sns.boxplot(x=f'param_{focus_param}', y='mean_test_score', data=grid_results_df)

                     plt.title(f'Focused Grid Search: {focus_param.split("__")[-1]} vs Score', fontsize=16)
                     plt.xlabel(focus_param.split('__')[-1], fontsize=14)
                     plt.ylabel(f'{scoring.upper()} Score', fontsize=14)
                     plt.grid(True, alpha=0.3)
                     plt.tight_layout()
                     plt.savefig(os.path.join(tuning_dir, "gb_grid_search_results.png"))
                     plt.close()

    except Exception as plot_err:
        logger.warning(f"Could not generate tuning plots for Gradient Boosting: {plot_err}")

    return best_estimator, best_params, final_cv_results


def tune_all_models(X_train, y_train, preprocessor, config):
    """Tunes all models defined in config and saves the best tuned pipelines.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        preprocessor (ColumnTransformer): Preprocessing pipeline definition.
        config (dict): Configuration dictionary.

    Returns:
        dict: Dictionary containing results for each tuned model.
    """
    tuning_dir = config.get('tuning_dir', 'output/tuning')
    models_dir = config.get('models_dir', 'output/models')
    scoring = config.get('tuning', {}).get('scoring_metric', 'roc_auc')
    os.makedirs(tuning_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    tuned_results = {}
    best_scores = {}

    # Tune models based on presence in the 'tuning' section of config
    if config.get('tuning', {}).get('logreg'):
        lr_model, lr_params, lr_cv_results = tune_logistic_regression(
            X_train, y_train, preprocessor, config
        )
        if lr_model:
            tuned_results["LogisticRegression"] = {"model": lr_model, "best_params": lr_params, "cv_results": lr_cv_results}
            best_scores['Logistic Regression'] = max(pd.DataFrame(lr_cv_results)['mean_test_score'].dropna().replace([np.inf, -np.inf], np.nan).dropna(), default=0)
            save_path = os.path.join(models_dir, "LogisticRegression_tuned_pipeline.joblib")
            joblib.dump(lr_model, save_path)
            logger.info(f"Saved tuned Logistic Regression pipeline to {save_path}")
        else:
             best_scores['Logistic Regression'] = 0

    if config.get('tuning', {}).get('rf'):
        rf_model, rf_params, rf_cv_results = tune_random_forest(
            X_train, y_train, preprocessor, config
        )
        if rf_model:
            tuned_results["RandomForest"] = {"model": rf_model, "best_params": rf_params, "cv_results": rf_cv_results}
            best_scores['Random Forest'] = max(pd.DataFrame(rf_cv_results)['mean_test_score'].dropna().replace([np.inf, -np.inf], np.nan).dropna(), default=0)
            save_path = os.path.join(models_dir, "RandomForest_tuned_pipeline.joblib")
            joblib.dump(rf_model, save_path)
            logger.info(f"Saved tuned Random Forest pipeline to {save_path}")
        else:
             best_scores['Random Forest'] = 0

    if config.get('tuning', {}).get('gb'):
        gb_model, gb_params, gb_cv_results = tune_gradient_boosting(
            X_train, y_train, preprocessor, config
        )
        if gb_model:
            tuned_results["GradientBoosting"] = {"model": gb_model, "best_params": gb_params, "cv_results": gb_cv_results}
            best_scores['Gradient Boosting'] = max(pd.DataFrame(gb_cv_results)['mean_test_score'].dropna().replace([np.inf, -np.inf], np.nan).dropna(), default=0)
            save_path = os.path.join(models_dir, "GradientBoosting_tuned_pipeline.joblib")
            joblib.dump(gb_model, save_path)
            logger.info(f"Saved tuned Gradient Boosting pipeline to {save_path}")
        else:
             best_scores['Gradient Boosting'] = 0

    # Plot comparison of best scores achieved during tuning
    if best_scores:
        models = list(best_scores.keys())
        scores = list(best_scores.values())
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        plt.bar(models, scores, color=colors)
        plt.title(f'Best {scoring.upper()} Score by Model after Tuning', fontsize=16)
        plt.ylabel(f'{scoring.upper()} Score', fontsize=14)
        # Adjust y-axis limits dynamically
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        plt.ylim(max(0, min_score - 0.05 * (max_score - min_score)), min(1.0, max_score + 0.1 * (max_score - min_score)))

        for i, score in enumerate(scores):
            plt.text(i, score + 0.01 * (max_score - min_score), f"{score:.4f}", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(tuning_dir, "tuned_model_comparison.png"))
        plt.close()
        logger.info(f"Tuned model comparison plot saved to {tuning_dir}")

    return tuned_results


if __name__ == "__main__":
    
    logger.info("--- Starting Hyperparameter Tuning Script ---")
    pipeline_config = load_config()
    
    proc_data_dir = pipeline_config.get('processed_data_dir')
    train_X, train_y = load_processed_data(proc_data_dir)
    
    # Define the preprocessor based on loaded training data for consistency
    proc_preprocessor = define_preprocessor(train_X)
    
    # Run tuning for all models specified in the config
    tuning_summary = tune_all_models(train_X, train_y, proc_preprocessor, pipeline_config)
    
    # Log summary of tuning results
    logger.info("--- Hyperparameter Tuning Summary ---")
    scoring_metric = pipeline_config.get('tuning',{}).get('scoring_metric','N/A')
    models_dir_path = pipeline_config.get('models_dir')
    for name, result in tuning_summary.items():
        if result.get('model'):
             # Safely calculate best score from CV results
             cv_scores = pd.DataFrame(result['cv_results'])['mean_test_score'].dropna()
             cv_scores = cv_scores[~np.isinf(cv_scores)] # Remove inf/-inf
             best_cv_score = cv_scores.max() if not cv_scores.empty else np.nan
             
             logger.info(f"  {name}: Best CV Score ({scoring_metric}) = {best_cv_score:.4f}")
             logger.info(f"     Best Params: {result['best_params']}")
             logger.info(f"     Tuned Model Saved: {models_dir_path}/{name}_tuned_pipeline.joblib")
        else:
             logger.info(f"  {name}: Tuning skipped or failed.")
    logger.info("--- Hyperparameter Tuning Script Finished ---")
