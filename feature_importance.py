# feature_importance.py
# Feature importance analysis for insurance enrollment prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, partial_dependence
import shap
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_feature_names(column_transformer):
    """
    Get feature names from column transformer
    
    Args:
        column_transformer: Fitted column transformer
        
    Returns:
        list: List of feature names
    """
    # Get feature names from all transformers
    output_features = []
    
    for name, trans, cols in column_transformer.transformers_:
        if name != 'drop':
            if hasattr(trans, 'get_feature_names_out'):
                output_features.extend(trans.get_feature_names_out(cols))
            else:
                output_features.extend(cols)
    
    return output_features

def analyze_feature_importance(model_results, X, y, output_dir="figures"):
    """
    Analyze and visualize feature importance for different models
    
    Args:
        model_results (dict): Dictionary with model evaluation results
        X (pandas.DataFrame): Features dataframe
        y (pandas.Series): Target variable
        output_dir (str): Directory to save output visualizations
        
    Returns:
        dict: Dictionary with feature importance results for each model
    """
    logger.info("Analyzing feature importance for different models")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    feature_importance_results = {}
    
    for model_name, result in model_results.items():
        logger.info(f"Calculating feature importance for {model_name}")
        
        pipeline = result["pipeline"]
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        
        # Get feature names after preprocessing
        feature_names = get_feature_names(preprocessor)
        
        # Transform the input data
        X_transformed = preprocessor.transform(X)
        
        # Calculate feature importance based on model type
        if model_name == "LogisticRegression":
            # For logistic regression, use coefficients
            coefficients = model.coef_[0]
            importance = np.abs(coefficients)
            feature_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
            feature_imp = feature_imp.sort_values('Importance', ascending=False)
            
            # Visualize coefficients (direction matters)
            plt.figure(figsize=(12, 10))
            top_features = feature_imp.head(15)
            colors = ['red' if c < 0 else 'green' for c in coefficients[feature_imp.head(15).index]]
            sns.barplot(x='Importance', y='Feature', data=top_features, palette=colors)
            plt.title(f"Feature Importance - {model_name} (Coefficients)", fontsize=16)
            plt.xlabel("Coefficient Magnitude", fontsize=14)
            plt.ylabel("Feature", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name}_coefficients.png")
            plt.close()
            
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models, use built-in feature importance
            importance = model.feature_importances_
            feature_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
            feature_imp = feature_imp.sort_values('Importance', ascending=False)
            
            # Visualize
            plt.figure(figsize=(12, 10))
            top_features = feature_imp.head(15)
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title(f"Feature Importance - {model_name}", fontsize=16)
            plt.xlabel("Importance", fontsize=14)
            plt.ylabel("Feature", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name}_feature_importance.png")
            plt.close()
        
        # Calculate permutation importance for all models
        perm_importance = permutation_importance(pipeline, X, y, n_repeats=10, random_state=42)
        
        perm_imp_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importance.importances_mean
        })
        perm_imp_df = perm_imp_df.sort_values('Importance', ascending=False)
        
        # Visualize permutation importance
        plt.figure(figsize=(12, 10))
        top_features = perm_imp_df.head(15)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f"Permutation Importance - {model_name}", fontsize=16)
        plt.xlabel("Mean Decrease in Accuracy", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_permutation_importance.png")
        plt.close()
        
        # Store results
        feature_importance_results[model_name] = {
            "model_specific_importance": feature_imp if 'feature_imp' in locals() else None,
            "permutation_importance": perm_imp_df
        }
        
        # Try to generate SHAP values for model explanation
        try:
            # Only use a subset of data for SHAP to avoid long computation times
            sample_size = min(500, len(X))
            X_sample = X.sample(sample_size, random_state=42)
            X_sample_transformed = preprocessor.transform(X_sample)
            
            # Create SHAP explainer
            if model_name == "LogisticRegression":
                explainer = shap.LinearExplainer(model, X_sample_transformed)
                shap_values = explainer.shap_values(X_sample_transformed)
                
                # For LogisticRegression, shap_values might be a list with one element
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample_transformed)
                
                # For binary classification with tree models, shap_values might be a list with two elements
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Focus on the positive class
            
            # Plot SHAP summary
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample_transformed, feature_names=feature_names, show=False)
            plt.title(f"SHAP Feature Importance - {model_name}", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name}_shap_summary.png")
            plt.close()
            
            # Plot SHAP dependence plots for top 3 features
            top_features_idx = np.argsort(-np.abs(shap_values).mean(0))[:3]
            
            for idx in top_features_idx:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(idx, shap_values, X_sample_transformed, feature_names=feature_names, show=False)
                plt.title(f"SHAP Dependence Plot - {feature_names[idx]}", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{model_name}_shap_dependence_{feature_names[idx].replace(' ', '_')}.png")
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not generate SHAP visualizations for {model_name}: {e}")
    
    # Create a consolidated view of feature importance across models
    top_features_by_model = []
    
    for model_name, result in feature_importance_results.items():
        perm_imp = result["permutation_importance"].head(10)[["Feature", "Importance"]]
        perm_imp.columns = ["Feature", f"{model_name}"]
        top_features_by_model.append(perm_imp)
    
    # Merge all feature importance dataframes
    consolidated_importance = top_features_by_model[0][["Feature"]]
    for df in top_features_by_model:
        consolidated_importance = consolidated_importance.merge(df, on="Feature", how="outer")
    
    consolidated_importance = consolidated_importance.fillna(0)
    
    # Save consolidated feature importance to CSV
    consolidated_importance.to_csv(f"{output_dir}/consolidated_feature_importance.csv", index=False)
    
    logger.info("Feature importance analysis completed")
    return feature_importance_results
