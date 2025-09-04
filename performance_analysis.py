# %%
"""
This script automatically evaluates multiple SDP models on all datasets
found in the SDP_Dataset_Pool folder. It supports the following algorithms:
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes (NB)
- Multi-Layer Perceptron (MLP)

Usage:
    python ml_classification_kit.py
    
The script will automatically:
1. Discover all CSV files in the SDP_Dataset_Pool folder
2. Run all 5 algorithms on each dataset
3. Save results for each algorithm-dataset combination

Output:
    Saves results to CSV files: all_results.csv
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score


def train_decision_tree_model(features_train, target_train):
    """
    Train a Decision Tree classifier with fixed random state for reproducibility.
    
    Args:
        features_train (array-like): Training feature matrix
        target_train (array-like): Training target vector
        
    Returns:
        sklearn.tree.DecisionTreeClassifier: Trained Decision Tree model
    """
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(features_train, target_train)
    return decision_tree_model


def train_random_forest_model(features_train, target_train):
    """
    Train a Random Forest classifier with fixed random state for reproducibility.
    
    Args:
        features_train (array-like): Training feature matrix
        target_train (array-like): Training target vector
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained Random Forest model
    """
    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(features_train, target_train)
    return random_forest_model


def train_svm_model(features_train, target_train):
    """
    Train a Support Vector Machine classifier with fixed random state for reproducibility.
    
    Args:
        features_train (array-like): Training feature matrix
        target_train (array-like): Training target vector
        
    Returns:
        sklearn.svm.SVC: Trained SVM model
    """
    svm_model = SVC(random_state=42)
    svm_model.fit(features_train, target_train)
    return svm_model


def train_gaussian_naive_bayes_model(features_train, target_train):
    """
    Train a Gaussian Naive Bayes classifier.
    Note: GaussianNB doesn't have a random_state parameter as it's deterministic.
    
    Args:
        features_train (array-like): Training feature matrix
        target_train (array-like): Training target vector
        
    Returns:
        sklearn.naive_bayes.GaussianNB: Trained Gaussian Naive Bayes model
    """
    gaussian_nb_model = GaussianNB()
    gaussian_nb_model.fit(features_train, target_train)
    return gaussian_nb_model


def train_mlp_model(features_train, target_train):
    """
    Train a Multi-Layer Perceptron classifier with fixed random state for reproducibility.
    
    Args:
        features_train (array-like): Training feature matrix
        target_train (array-like): Training target vector
        
    Returns:
        sklearn.neural_network.MLPClassifier: Trained MLP model
    """
    mlp_model = MLPClassifier(random_state=42)
    mlp_model.fit(features_train, target_train)
    return mlp_model


def evaluate_classifier_performance(trained_model, features_test, target_test):
    """
    Evaluate a trained classifier's performance using balanced accuracy.
    
    Args:
        trained_model: Trained sklearn classifier
        features_test (array-like): Test feature matrix
        target_test (array-like): Test target vector
        
    Returns:
        float: Balanced accuracy score on test data
    """
    predicted_labels = trained_model.predict(features_test)
    balanced_accuracy = balanced_accuracy_score(target_test, predicted_labels)
    return balanced_accuracy


def load_dataset_by_id(dataset_identifier):
    """
    Load dataset from CSV file based on dataset ID.
    
    Args:
        dataset_identifier (int): Dataset ID number
        
    Returns:
        tuple: (features_dataframe, target_series) - X and y data
    """
    dataset_path = f"SDP_Dataset_Pool/{dataset_identifier}.csv"
    dataset_dataframe = pd.read_csv(dataset_path)
    
    # Extract features (all columns except 'target')
    features_dataframe = dataset_dataframe.drop(['target'], axis=1)
    # Extract target variable
    target_series = dataset_dataframe['target']
    
    return features_dataframe, target_series


def split_data_for_training(features_data, target_data, test_proportion=0.2, random_seed=42):
    """
    Split data into training and testing sets with stratification.
    
    Args:
        features_data (array-like): Feature matrix
        target_data (array-like): Target vector
        test_proportion (float): Proportion of data to use for testing (default: 0.2)
        random_seed (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Split datasets
    """
    features_train, features_test, target_train, target_test = train_test_split(
        features_data, target_data, 
        test_size=test_proportion, 
        stratify=target_data, 
        random_state=random_seed
    )
    return features_train, features_test, target_train, target_test


def discover_datasets_in_folder(dataset_folder_path="SDP_Dataset_Pool"):
    """
    Discover all CSV files in the dataset folder and extract dataset identifiers.
    
    Args:
        dataset_folder_path (str): Path to the folder containing datasets
        
    Returns:
        list: List of dataset identifiers (integers) found in the folder
    """
    # Get all CSV files in the dataset folder
    csv_files_pattern = os.path.join(dataset_folder_path, "*.csv")
    csv_files_list = glob.glob(csv_files_pattern)
    
    # Extract dataset identifiers from filenames
    dataset_identifiers = []
    for csv_file_path in csv_files_list:
        # Get filename without path and extension
        filename_base = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        # Check if filename is a number
        if filename_base.isdigit():
            dataset_identifiers.append(int(filename_base))
    
    # Sort dataset identifiers for consistent processing order
    dataset_identifiers.sort()
    
    return dataset_identifiers


def get_all_algorithm_functions():
    """
    Return dictionary mapping algorithm names to their training functions.
    
    Returns:
        dict: Dictionary with algorithm names as keys and training functions as values
    """
    algorithm_training_functions = {
        'DT': train_decision_tree_model,
        'RF': train_random_forest_model,
        'SVM': train_svm_model,
        'NB': train_gaussian_naive_bayes_model,
        'MLP': train_mlp_model
    }
    return algorithm_training_functions


def run_all_algorithms_on_dataset(dataset_identifier, algorithm_functions):
    """
    Run all algorithms on a single dataset and collect results.
    
    Args:
        dataset_identifier (int): Dataset ID number
        algorithm_functions (dict): Dictionary mapping algorithm names to training functions
        
    Returns:
        dict: Dictionary with algorithm names as keys and balanced accuracy scores as values
    """
    dataset_results = {}
    
    try:
        # Load dataset once for all algorithms
        print(f"  Loading dataset {dataset_identifier}...")
        features_data, target_data = load_dataset_by_id(dataset_identifier)
        print(f"    Dataset shape: {features_data.shape[0]} samples, {features_data.shape[1]} features")
        
        # Split data once for all algorithms
        features_train, features_test, target_train, target_test = split_data_for_training(
            features_data, target_data
        )
        print(f"    Data split: {len(features_train)} training, {len(features_test)} test samples")
        
        # Run each algorithm on the same train/test split
        for algorithm_name, training_function in algorithm_functions.items():
            try:
                print(f"  Running {algorithm_name}...")
                
                # Train the model
                trained_model = training_function(features_train, target_train)
                
                # Evaluate model performance on test data
                test_balanced_accuracy = evaluate_classifier_performance(
                    trained_model, features_test, target_test
                )
                
                # Store result
                dataset_results['Balanced_Acc_'+algorithm_name] = test_balanced_accuracy
                print(f"    {algorithm_name}: Balanced Accuracy = {test_balanced_accuracy:.4f}")
                
            except Exception as algorithm_error:
                print(f"    Error with {algorithm_name}: {algorithm_error}")
                # Store NaN for failed algorithms to maintain consistent structure
                dataset_results['Balanced_Acc_'+algorithm_name] = np.nan
        
        return dataset_results
        
    except Exception as dataset_error:
        print(f"  Error loading dataset {dataset_identifier}: {dataset_error}")
        # Return NaN for all algorithms if dataset loading fails
        return {alg_name: np.nan for alg_name in algorithm_functions.keys()}


def save_all_results(all_results_list):
    """
    Save experimental results for all algorithms on all datasets to a single CSV file.
    
    Args:
        all_results_list (list): List of dictionaries, each containing results for one dataset
    """

    # Create results folder if it doesn't exist
    results_folder = "Performance_Analysis_Results"
    os.makedirs(results_folder, exist_ok=True)

    # Create DataFrame from all results
    results_dataframe = pd.DataFrame(all_results_list)
    
    # Sort by dataset_id for consistent ordering
    results_dataframe = results_dataframe.sort_values('dataset_id').reset_index(drop=True)
    
    output_filename = os.path.join(results_folder, 'performance_results.csv')
    results_dataframe.to_csv(output_filename, index=False)
    print(f"\nAll results saved to: {output_filename}")


    # Generate performance statistics summary
    algorithm_columns = [col for col in results_dataframe.columns if col != 'dataset_id']
    
    # Create summary statistics DataFrame
    summary_stats = []
    for algorithm in algorithm_columns:
        alg_data = results_dataframe[algorithm].dropna()  # Remove NaN values
        if len(alg_data) > 0:
            stats = {
                'Algorithm': algorithm,
                'Min': alg_data.min(),
                'Max': alg_data.max(),
                'Mean': alg_data.mean(),
                'S.D.': alg_data.std(),
                '25%': alg_data.quantile(0.25),
                '75%': alg_data.quantile(0.75)
            }
            summary_stats.append(stats)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_stats)
    summary_filename = os.path.join(results_folder, 'performance_statistics_summary.csv')
    summary_df.to_csv(summary_filename, index=False)
    print(f"Performance statistics saved to: {summary_filename}")
    
    # Display summary of the results
    print(f"Results summary:")
    print(f"  - Total datasets: {len(results_dataframe)}")
    print(f"  - Columns: {list(results_dataframe.columns)}")
    print(f"  - Dataset IDs: {sorted(results_dataframe['dataset_id'].tolist())}")


def main():
    """
    Main execution function that runs all algorithms on all discovered datasets.
    """
    print("=" * 60)
    print("SDP Model Performance Analysis")
    print("=" * 60)
    
    # Discover all datasets in the folder
    print("Discovering datasets...")
    dataset_identifiers = discover_datasets_in_folder()
    
    if not dataset_identifiers:
        print("No datasets found in SDP_Dataset_Pool folder!")
        print("Make sure the folder exists and contains CSV files named with numbers (e.g., 1.csv, 2.csv)")
        sys.exit(1)
    
    print(f"Found {len(dataset_identifiers)} datasets: {dataset_identifiers}")
    
    # Get all algorithm functions
    algorithm_functions = get_all_algorithm_functions()
    algorithm_names = list(algorithm_functions.keys())
    
    print(f"Algorithms to run: {algorithm_names}")
    print("-" * 60)
    
    # Initialize counters for tracking progress
    total_datasets = len(dataset_identifiers)
    processed_datasets = 0
    successful_algorithms = 0
    failed_algorithms = 0
    
    # List to store all results
    all_results_list = []
    
    # Run experiments for each dataset (all algorithms per dataset)
    for dataset_id in dataset_identifiers:
        print(f"\nProcessing Dataset {dataset_id} ({processed_datasets + 1}/{total_datasets}):")
        
        # Run all algorithms on this dataset
        dataset_results = run_all_algorithms_on_dataset(dataset_id, algorithm_functions)
        
        # Count successful and failed algorithms
        for alg_name, result in dataset_results.items():
            if pd.isna(result):
                failed_algorithms += 1
            else:
                successful_algorithms += 1
        
        # Add dataset_id to results and append to master list
        dataset_results['dataset_id'] = dataset_id
        all_results_list.append(dataset_results)
        
        processed_datasets += 1
    
    # Save all results to a single CSV file
    save_all_results(all_results_list)
    
    # Print final summary
    total_experiments = total_datasets * len(algorithm_names)
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total datasets processed: {processed_datasets}")
    print(f"Total algorithm runs: {total_experiments}")
    print(f"Successful algorithm runs: {successful_algorithms}")
    print(f"Failed algorithm runs: {failed_algorithms}")
    print(f"Success rate: {(successful_algorithms/total_experiments)*100:.1f}%")
    
    if failed_algorithms == 0:
        print("\nAll experiments completed successfully!")
    else:
        print(f"\n{failed_algorithms} algorithm runs failed. Check error messages above.")
    
    print(f"\nAll results consolidated in 'performance_results.csv'")
    print("Each row represents one dataset with results from all algorithms.")
    print("=" * 60)


if __name__ == "__main__":
    main()


