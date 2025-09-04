"""
Explainable Boosting Machine (EBM) Analysis for Data Quality Impact on Classifier Performance

This script analyzes the relationship between data quality metrics and classifier performance
using Explainable Boosting Machine models. It trains separate EBM models for different 
classifiers to predict balanced accuracy based on data quality characteristics.

Author: [Your Name]
Date: [Date]
Purpose: Replication kit for journal paper on data quality impact analysis

Dependencies:
- pandas
- numpy  
- matplotlib
- scikit-learn
- interpret (EBM library)
- pathlib
- warnings

Input Files:
- Quality_Issue_Extraction_Output/quality_issues_dataset.csv: Data quality metrics
- Performance_Analysis_Results/performanc_results.csv: Classifier performance results

Output:
- EBM_Plots/: Directory containing shape function comparison plots
- Individual CSV files with feature importance scores for each classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from interpret.glassbox import ExplainableBoostingRegressor
import warnings
from pathlib import Path

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration constants
OUTPUT_PLOTS_DIR = './EBM_Influence_Analysis/Plots'
OUTPUT_DATA_DIR = './EBM_Influence_Analysis/Result_Data'
QUALITY_ISSUES_FILE = 'Quality_Issue_Extraction_Output/Result_Data/quality_issues_result.csv'
PERFORMANCE_RESULTS_FILE = 'Performance_Analysis_Results/performance_results.csv'
TEST_SIZE_RATIO = 0.2

# Data quality feature columns (independent variables)
# Data quality feature columns (independent variables)
DATA_QUALITY_FEATURES = [
    'Class Imbalance', 
    'Class Overlap', 
    'Outlier', 
    'Noise', 
    'Irrelevant Features'
]

# Create output directories if they don't exist
Path(OUTPUT_PLOTS_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DATA_DIR).mkdir(parents=True, exist_ok=True)


def load_and_merge_datasets():
    """
    Load data quality metrics and classifier performance results, then merge them.
    
    This function reads two CSV files:
    1. Quality issues dataset containing data quality metrics for each dataset
    2. Performance results containing balanced accuracy scores for different classifiers
    
    Returns:
        pd.DataFrame: Merged dataset with quality metrics and performance scores
                     joined on 'dataset_id' column
    
    Raises:
        FileNotFoundError: If either input CSV file is not found
        KeyError: If 'dataset_id' column is missing from either file
    """
    # Load data quality metrics
    quality_issues_df = pd.read_csv(QUALITY_ISSUES_FILE)
    
    # Load classifier performance results
    model_performance_df = pd.read_csv(PERFORMANCE_RESULTS_FILE)

    # Merge datasets on dataset_id using inner join
    # Only keeps records that exist in both datasets
    merged_dataset = pd.merge(
        quality_issues_df, 
        model_performance_df, 
        on='dataset_id', 
        how='inner'
    )

    # Sort by dataset_id in ascending order
    merged_dataset = merged_dataset.sort_values(by='dataset_id', ascending=True)

    merged_dataset.to_csv(f'{OUTPUT_DATA_DIR}/issue_performance_merged.csv', index=False)
    
    return merged_dataset


def train_ebm_models_for_all_classifiers(merged_dataset):
    """
    Train individual EBM models for each classifier to predict balanced accuracy.
    
    This function creates separate EBM regression models for each classifier found in
    the dataset. Each model uses data quality metrics as features to predict the
    balanced accuracy of that specific classifier.
    
    Args:
        merged_dataset (pd.DataFrame): Dataset containing quality metrics and performance scores
        
    Returns:
        dict: Dictionary mapping classifier names to trained EBM model objects
              Format: {classifier_name: trained_ebm_model, ...}
    
    Raises:
        ValueError: If no balanced accuracy columns are found in the dataset
    """
    # Identify target columns (balanced accuracy for each classifier)
    balanced_accuracy_columns = [
        col for col in merged_dataset.columns 
        if col.startswith('Balanced_Acc_')
    ]
    
    if not balanced_accuracy_columns:
        raise ValueError("No balanced accuracy columns found in the merged dataset")
    
    # Dictionary to store trained models for each classifier
    trained_ebm_models = {}
    
    # List to store evaluation results for summary
    model_evaluation_results = []
    
    # Train one EBM model per classifier
    for target_column in balanced_accuracy_columns:
        # Extract classifier name from column name (e.g., 'Balanced_Acc_SVM' -> 'SVM')
        classifier_name = target_column.replace('Balanced_Acc_', '')
        print(f"\nTraining EBM model for {classifier_name} classifier...")
        
        # Skip classifiers with no valid data
        valid_data_mask = merged_dataset[target_column].notna()
        if valid_data_mask.sum() == 0:
            print(f"No valid data for {classifier_name}, skipping this classifier.")
            continue
            
        # Prepare features (X) and target (y) with only valid data
        feature_matrix = merged_dataset.loc[valid_data_mask, DATA_QUALITY_FEATURES]
        target_vector = merged_dataset.loc[valid_data_mask, target_column]
        
        # Handle missing values in features by filling with median
        feature_matrix_cleaned = _handle_missing_feature_values(feature_matrix)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix_cleaned, 
            target_vector, 
            test_size=TEST_SIZE_RATIO, 
            random_state=RANDOM_SEED
        )
        
        # Create and configure EBM model
        ebm_model = ExplainableBoostingRegressor(
            feature_names=DATA_QUALITY_FEATURES,
            interactions=0  # Disable interactions for clearer interpretation
        )
        
        # Train the EBM model
        ebm_model.fit(X_train, y_train)
        
        # Apply monotonicity constraints to improve model interpretability
        _apply_monotonicity_constraints(ebm_model, classifier_name)
        
        # Evaluate model performance on test set
        test_predictions = ebm_model.predict(X_test)
        evaluation_metrics = _calculate_regression_metrics(y_test, test_predictions)
        
        # Display evaluation results
        print(f"{classifier_name} - MAE: {evaluation_metrics['MAE']:.4f}, "
              f"MSE: {evaluation_metrics['MSE']:.4f}, "
              f"RMSE: {evaluation_metrics['RMSE']:.4f}")
        
        # Store results for summary
        model_evaluation_results.append({
            'Model': classifier_name,
            'MAE': evaluation_metrics['MAE'],
            'MSE': evaluation_metrics['MSE'],
            'RMSE': evaluation_metrics['RMSE'],
            'Train Size': len(X_train),
            'Test Size': len(X_test)
        })
        
        # Store the trained model
        trained_ebm_models[classifier_name] = ebm_model
    
    # Display comprehensive results summary
    _display_model_performance_summary(model_evaluation_results)
    
    return trained_ebm_models


def _handle_missing_feature_values(feature_matrix):
    """
    Handle missing values in feature matrix by filling with median values.
    
    Args:
        feature_matrix (pd.DataFrame): Feature matrix potentially containing NaN values
        
    Returns:
        pd.DataFrame: Feature matrix with missing values filled
    """
    feature_matrix_cleaned = feature_matrix.copy()
    
    for column_name in feature_matrix_cleaned.columns:
        if feature_matrix_cleaned[column_name].isna().any():
            median_value = feature_matrix_cleaned[column_name].median()
            missing_count = feature_matrix_cleaned[column_name].isna().sum()
            feature_matrix_cleaned[column_name] = feature_matrix_cleaned[column_name].fillna(median_value)
            print(f"Filled {missing_count} missing values in {column_name} with median {median_value:.4f}")
    
    return feature_matrix_cleaned


def _apply_monotonicity_constraints(ebm_model, classifier_name):
    """
    Apply monotonicity constraints to EBM model features for better interpretability.
    
    Args:
        ebm_model: Trained EBM model object
        classifier_name (str): Name of the classifier for logging purposes
    """
    try:
        print(f"Applying monotonization to {classifier_name} model features...")
        for feature_index, feature_name in enumerate(DATA_QUALITY_FEATURES):
            try:
                # Try to monotonize each feature (can be increasing or decreasing)
                ebm_model.monotonize(term=feature_index)
                print(f"  - Successfully monotonized {feature_name}")
            except Exception as feature_error:
                print(f"  - Could not monotonize {feature_name}: {feature_error}")
    except Exception as general_error:
        print(f"Error during monotonization: {general_error}")


def _calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        dict: Dictionary containing MAE, MSE, and RMSE metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }


def _display_model_performance_summary(evaluation_results):
    """
    Display a formatted summary of all model performance metrics.
    
    Args:
        evaluation_results (list): List of dictionaries containing evaluation metrics
    """
    results_dataframe = pd.DataFrame(evaluation_results)
    if not results_dataframe.empty:
        print("\nModel Performance Summary:")
        print(results_dataframe.to_string(index=False))


def generate_shape_function_comparison_plots(trained_ebm_models):
    """
    Generate comparison plots of EBM shape functions across all classifiers.
    
    This function creates one plot per data quality feature, showing how each
    classifier's EBM model responds to that feature. Shape functions show the
    contribution of each feature value to the predicted balanced accuracy.
    
    Args:
        trained_ebm_models (dict): Dictionary of trained EBM models for each classifier
        
    Note:
        - Plots are saved to the OUTPUT_PLOTS_DIR directory
        - Also exports feature importance scores to individual CSV files
        - Handles EBM dimension mismatch issues gracefully
    """
    if not trained_ebm_models:
        print("No trained models available for plotting shape functions.")
        return
    
    # Generate one comparison plot per data quality feature
    for feature_index, feature_name in enumerate(DATA_QUALITY_FEATURES):
        plt.figure(figsize=(10, 6))
        has_valid_plot_data = False
        
        # Plot shape function for each classifier's model
        for classifier_name, ebm_model in trained_ebm_models.items():
            try:
                # Extract and save feature importance scores
                _extract_and_save_feature_influence(ebm_model, classifier_name)
                
                # Get shape function data for current feature
                global_explanation = ebm_model.explain_global()
                feature_data = global_explanation.data(feature_index)
                
                # Extract feature values and their contributions
                feature_values = np.array(feature_data['names'])
                feature_contributions = np.array(feature_data['scores'])
                
                # Handle EBM dimension mismatch (common issue where names has one more element than scores)
                feature_values, feature_contributions = _handle_ebm_dimension_mismatch(
                    feature_values, feature_contributions
                )
                
                # Plot the shape function for this classifier
                plt.plot(
                    feature_values, 
                    feature_contributions, 
                    label=classifier_name, 
                    alpha=0.7
                )
                has_valid_plot_data = True
                
            except Exception as plot_error:
                print(f"Error plotting shape function for {classifier_name}, "
                      f"feature {feature_name}: {plot_error}")
                continue

        # Finalize and save the plot if we have valid data
        if has_valid_plot_data:
            _format_and_save_shape_function_plot(feature_name)
        else:
            print(f"Warning: No valid shape functions to plot for feature {feature_name}.")
        
        plt.close()


def _extract_and_save_feature_influence(ebm_model, classifier_name):
    """
    Extract feature influence scores from EBM model and save to CSV.
    
    Args:
        ebm_model: Trained EBM model
        classifier_name (str): Name of classifier for filename
    """
    global_explanation = ebm_model.explain_global()
    importance_dataframe = pd.DataFrame(global_explanation.data())
    importance_dataframe = importance_dataframe.sort_values(by="scores", ascending=False)
    importance_dataframe['type'] = classifier_name
    importance_dataframe.to_csv(f'{OUTPUT_DATA_DIR}/{classifier_name}_issue_influence.csv', index=False)


def _handle_ebm_dimension_mismatch(feature_values, feature_contributions):
    """
    Handle dimension mismatch between feature values and contributions in EBM.
    
    This is a common issue where the 'names' array is one element longer than 'scores'.
    
    Args:
        feature_values (np.array): Array of feature values
        feature_contributions (np.array): Array of feature contributions
        
    Returns:
        tuple: Aligned feature_values and feature_contributions arrays
    """
    if len(feature_values) != len(feature_contributions):
        if len(feature_values) == len(feature_contributions) + 1:
            # Common case: drop the last element in feature_values
            feature_values = feature_values[:-1]
        else:
            # General case: take the minimum common length
            min_length = min(len(feature_values), len(feature_contributions))
            feature_values = feature_values[:min_length]
            feature_contributions = feature_contributions[:min_length]
    
    return feature_values, feature_contributions


def _format_and_save_shape_function_plot(feature_name):
    """
    Apply formatting to shape function plot and save to file.
    
    Args:
        feature_name (str): Name of the feature being plotted
    """
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(feature_name, weight='bold', fontsize=23)
    plt.xticks(weight='bold', fontsize=23)
    plt.ylabel('Contribution', weight='bold', fontsize=23)
    plt.yticks(weight='bold', fontsize=23)
    plt.legend(loc='best', prop={'weight': 'bold', 'size': 16})
    plt.tight_layout()
    
    # Save the plot
    output_filename = f'{OUTPUT_PLOTS_DIR}/{feature_name}_shape_function.png'
    plt.savefig(output_filename, dpi=300)


def generate_influence_score_heatmap(trained_ebm_models):
    """
    Generate a heatmap showing feature influence scores across all classifiers.
    
    This function extracts feature importance scores from all trained EBM models
    and creates a heatmap visualization showing how each data quality feature
    contributes to the prediction of balanced accuracy for each classifier.
    
    Args:
        trained_ebm_models (dict): Dictionary of trained EBM models for each classifier
        
    Returns:
        pd.DataFrame: Feature importance matrix used for the heatmap
    """
    if not trained_ebm_models:
        print("No trained models available for generating feature importance heatmap.")
        return None
    
    # Dictionary to store feature importance scores for each model
    importance_matrix = {}
    
    # Extract feature importance scores from each model
    for classifier_name, ebm_model in trained_ebm_models.items():
        try:
            # Get global explanation from the EBM model
            global_explanation = ebm_model.explain_global()
            
            # Extract feature names and their importance scores
            feature_names = global_explanation.feature_names
            feature_scores = []
            
            # Get importance score for each feature
            for feature_idx in range(len(DATA_QUALITY_FEATURES)):
                feature_data = global_explanation.data(feature_idx)
                # Calculate absolute mean of scores as feature importance
                importance_score = np.abs(np.array(feature_data['scores'])).mean()
                feature_scores.append(importance_score)
            
            # Convert to percentage using your specified formula
            feature_scores = np.array(feature_scores)
            feature_scores_percentage = np.round((feature_scores / feature_scores.sum()) * 100, 2)
            
            # Store percentage scores for this classifier
            importance_matrix[classifier_name] = feature_scores_percentage.tolist()
            
        except Exception as e:
            print(f"Error extracting feature influence for {classifier_name}: {e}")
            continue
    
    # Convert to DataFrame for easier manipulation
    importance_df = pd.DataFrame(importance_matrix, index=DATA_QUALITY_FEATURES)
    
    # Transpose so classifiers are on y-axis and features on x-axis
    importance_df = importance_df.T
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    
    # Create heatmap with custom styling to match your example
    heatmap = sns.heatmap(
        importance_df,
        annot=True,           # Show values in cells
        fmt='.2f',           # Format numbers to 2 decimal places
        cmap='Blues',        # Use blue color scheme
        cbar_kws={'label': 'Influence Score (%)'},
        square=False,        # Don't force square cells
        linewidths=0.5,      # Add thin lines between cells
        linecolor='white'    # White lines between cells
    )
    
    # Customize the plot appearance
    #plt.title('Feature Importance Heatmap Across Classifiers', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Data Quality Issue', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_filename = f'{OUTPUT_PLOTS_DIR}/influence_score_heatmap.png'
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"Influence score heatmap saved to: {heatmap_filename}")
    
    # Display the heatmap
    #plt.show()
    
    # Also save the importance matrix as CSV
    csv_filename = f'{OUTPUT_DATA_DIR}/influence_score_matrix_percentage.csv'
    importance_df.to_csv(csv_filename)
    print(f"Influence score percentage matrix saved to: {csv_filename}")
    
    return importance_df


def execute_complete_ebm_analysis():
    """
    Execute the complete EBM analysis workflow.
    
    This is the main orchestration function that:
    1. Loads and merges the input datasets
    2. Trains EBM models for all classifiers
    3. Generates shape function comparison plots
    4. Provides comprehensive error handling
    
    Raises:
        Exception: Any errors during execution are caught and reported
    """
    try:
        # Step 1: Load and merge input datasets
        print("Loading and merging datasets...")
        merged_dataset = load_and_merge_datasets()
        print(f"Successfully merged datasets. Total records: {len(merged_dataset)}")
        
        # Step 2: Train EBM models for all classifiers
        print("\nTraining EBM models for all classifiers...")
        trained_models = train_ebm_models_for_all_classifiers(merged_dataset)
        print(f"Successfully trained {len(trained_models)} EBM models")
        
        # Step 3: Generate shape function comparison plots
        print("\nGenerating shape function comparison plots...")
        generate_shape_function_comparison_plots(trained_models)
        
        # Step 4: Generate influence score heatmap
        print("\nGenerating influence score heatmap...")
        importance_matrix = generate_influence_score_heatmap(trained_models)
        
        print(f"\nAnalysis completed successfully!")
        print(f"- Shape function plots saved to: {OUTPUT_PLOTS_DIR}/")
        print(f"- Influence score heatmap saved to: {OUTPUT_PLOTS_DIR}/influence_score_heatmap.png")
        print(f"- Influence score matrix saved to: {OUTPUT_DATA_DIR}/influence_score_matrix_percentage.csv")
        print(f"- Individual influence score CSV files saved to current directory")
        
    except Exception as execution_error:
        print(f"Error during analysis execution: {execution_error}")
        raise


if __name__ == "__main__":
    execute_complete_ebm_analysis()



