"""
Data Quality Issue Detection for Software Defect Prediction Datasets
====================================================================

This script detects data quality issues in Software Defect Prediction datasets for research purposes.
It processes CSV files containing datasets with a 'target' column and extracts various
data quality indicators: four measures using pymfe's complexity meta-features and one 
additional measure (outlier count) using a custom IQR-based implementation.

Requirements:
- pandas
- numpy
- scikit-learn
- pymfe
- matplotlib

Author: Emmanuel Charleson Dapaah
Date: 05.09.2025
Version: 1.1

Usage:
    python quality_issue_extraction.py

Input:
    - CSV files in INPUT_FOLDER, each containing a 'target' column
    
Output:
    - quality_issues_dataset.csv in OUTPUT_FOLDER containing detected quality issues
      (4 measures from pymfe + 1 custom outlier count measure)
    - quality_issues_statistics_summary.csv in OUTPUT_FOLDER containing statistical
      descriptions (Min, Max, Mean, S.D., 25%, 75%) for each quality issue
    - Individual boxplot PNG files in OUTPUT_FOLDER for each quality issue distribution
      (e.g., Class_Imbalance_boxplot.png, Class_Overlap_boxplot.png, etc.)
"""

import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pymfe.mfe import MFE
from sklearn.model_selection import train_test_split
from pathlib import Path
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration - Update these paths for your environment
INPUT_FOLDER = os.path.join("SDP_Dataset_Pool")  # Dataset folder outside current directory
OUTPUT_PLOT_FOLDER = "Quality_Issue_Extraction_Output/Plots"  # Output plot folder in current directory
OUTPUT_DATA_FOLDER = "Quality_Issue_Extraction_Output/Result_Data"  # Output data folder in current directory

# Create output directory if it doesn't exist
Path(OUTPUT_PLOT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DATA_FOLDER).mkdir(parents=True, exist_ok=True)


def extract_data_quality_issues(feature_matrix_train, target_vector_train):
    """
    Extract data quality issues from training data using pymfe complexity meta-features.
    
    This function uses pymfe's complexity measures as indicators of data quality issues
    that can affect Software Defect Prediciton model performance.
    
    Parameters
    ----------
    feature_matrix_train : numpy.ndarray
        Training feature matrix (samples × features)
    target_vector_train : numpy.ndarray
        Training target vector (samples,)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing data quality issue measures
        
    Notes
    -----
    Quality issues measured (using pymfe complexity features):
    - c2: Class imbalance - Compute the imbalance ratio.
    - n1: Class overlap - Compute the fraction of borderline points.
    - ns_ratio: Attribute noise - Compute the noisiness of attributes.
    - mut_inf: Irrelevant features - Compute the mutual information between each attribute and target.
    
    Additional quality issue (computed separately):
    - outlier_count: Number of outliers detected using IQR method (custom implementation)
    """
    # Initialize MFE with complexity measures used as quality issue indicators
    quality_issue_extractor = MFE(
        features=["nr_inst", "nr_attr", "c2", "n1", "ns_ratio", "mut_inf"], 
        groups=["all"], 
        summary=["mean"], 
        random_state=42
    )
    
    # Fit the extractor and extract quality issue measures
    quality_issue_extractor.fit(feature_matrix_train, target_vector_train)
    extracted_quality_measures = quality_issue_extractor.extract()
    
    # Convert to DataFrame for easier handling
    quality_issues_dataframe = pd.DataFrame(
        extracted_quality_measures, 
        columns=extracted_quality_measures[0]
    )
    
    # Remove the first row (contains feature names, not values)
    quality_issues_dataframe.drop(index=quality_issues_dataframe.index[0], axis=0, inplace=True)
    
    return quality_issues_dataframe


def calculate_outlier_count_iqr_method(feature_matrix, outlier_threshold_multiplier=1.5):
    """
    Calculate the total number of outlier values in the dataset using IQR method.
    
    This is a custom implementation to detect outliers as a data quality issue,
    separate from the pymfe-based quality measures. An outlier is defined as a value 
    outside the interval: [Q1 - threshold_multiplier * IQR, Q3 + threshold_multiplier * IQR]
    where IQR = Q3 - Q1 (Interquartile Range).
    
    Parameters
    ----------
    feature_matrix : numpy.ndarray
        Numerical feature matrix (samples × features)
    outlier_threshold_multiplier : float, optional (default=1.5)
        Multiplier for IQR to define outlier boundaries.
        Higher values = more tolerance for outliers
        Lower values = less tolerance for outliers
        
    Returns
    -------
    int
        Total count of outlier values across all features in the dataset
        
    Notes
    -----
    This method applies the standard boxplot rule for outlier detection,
    commonly used in exploratory data analysis. This quality issue measure
    is computed independently from the pymfe-based measures.
    """
    # Calculate first and third quartiles for each feature (column-wise)
    first_quartile, third_quartile = np.percentile(feature_matrix, (25, 75), axis=0)
    
    # Calculate Interquartile Range (IQR) for each feature
    interquartile_range = third_quartile - first_quartile
    
    # Calculate outlier boundary adjustment
    outlier_boundary_adjustment = outlier_threshold_multiplier * interquartile_range
    
    # Define outlier boundaries for each feature
    lower_outlier_boundary = first_quartile - outlier_boundary_adjustment
    upper_outlier_boundary = third_quartile + outlier_boundary_adjustment
    
    # Create boolean mask identifying outliers
    # A value is an outlier if it's below lower boundary OR above upper boundary
    outlier_detection_mask = np.logical_or(
        feature_matrix < lower_outlier_boundary, 
        feature_matrix > upper_outlier_boundary
    )
    
    # Count total number of outlier values across entire dataset
    total_outlier_count = np.sum(outlier_detection_mask)
    
    return total_outlier_count


def process_single_dataset(dataset_file_path, dataset_identifier):
    """
    Process a single dataset file and extract all data quality issues.
    
    This function loads a dataset, performs train-test split, and extracts
    both quality issue measures and outlier count.
    
    Parameters
    ----------
    dataset_file_path : str
        Full path to the CSV dataset file
    dataset_identifier : str
        Unique identifier for the dataset (typically filename without extension)
        
    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing extracted quality issue measures for the dataset,
        or None if processing failed
        
    Notes
    -----
    Expected dataset format:
    - CSV file with 'target' column containing class labels
    - All other columns treated as features
    - Stratified split used to maintain class distribution
    """
    try:
        # Load dataset from CSV file
        dataset = pd.read_csv(dataset_file_path)
        
        # Separate features and target variable
        feature_matrix = dataset.drop(['target'], axis=1)
        target_vector = dataset['target']
        
        # Perform stratified train-test split to maintain class distribution
        (feature_matrix_train, feature_matrix_test, 
         target_vector_train, target_vector_test) = train_test_split(
            feature_matrix, target_vector, 
            test_size=0.2, 
            stratify=target_vector, 
            random_state=42
        )
        
        # Calculate outlier count using custom IQR method (separate quality issue measure)
        dataset_outlier_count = calculate_outlier_count_iqr_method(
            feature_matrix_train.to_numpy()
        )
        
        # Extract data quality issues from pymfe (4 complexity-based measures)
        data_quality_issues = extract_data_quality_issues(
            feature_matrix_train.to_numpy(), 
            target_vector_train.to_numpy()
        )

        # Invert mutual information to better represent irrelevant features quality issue
        # Higher values now indicate more irrelevant features (worse quality)
        data_quality_issues['mut_inf_inverted'] = 1 - data_quality_issues['mut_inf.mean']
        data_quality_issues = data_quality_issues.drop(['mut_inf.mean'], axis=1)
        
        # Add custom relative outlier count measure and dataset identifier
        data_quality_issues['outlier_count'] = dataset_outlier_count
        data_quality_issues['outlier_count'] = data_quality_issues['outlier_count']/(data_quality_issues['nr_inst'] * data_quality_issues['nr_attr'])
        data_quality_issues['dataset_id'] = dataset_identifier

        data_quality_issues = data_quality_issues.drop(['nr_inst', 'nr_attr'], axis=1)

        # Rename quality issue features to more descriptive names
        quality_issue_column_mapping = {
            "c2": "Class Imbalance",
            "n1": "Class Overlap", 
            "outlier_count": "Outlier",
            "ns_ratio": "Noise",
            "mut_inf_inverted": "Irrelevant Features"
        }
        data_quality_issues.rename(columns=quality_issue_column_mapping, inplace=True)
        
        return data_quality_issues
        
    except Exception as processing_error:
        print(f"[ERROR] Dataset {dataset_identifier} processing failed: {processing_error}")
        return None
    

def plot_issue_cooccurrence_heatmap(df, save_path):
    """
    Compute and plot a co-occurrence heatmap for data quality issues, then save it to a file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data quality issue values.
        If a 'dataset_id' column exists, it will be ignored.
    save_path : str
        Path (including filename) where the plot will be saved (e.g., 'output/heatmap.png').

    Returns
    -------
    co_occurrence : pd.DataFrame
        A DataFrame representing the co-occurrence counts between issues.

    Notes
    -----
    - Issues are binarized: value > 0 → 1 (issue present), value = 0 → 0 (issue absent).
    - The heatmap shows the number of datasets where two issues occur together.
    """

    # Drop 'dataset_id' if present
    if 'dataset_id' in df.columns:
        df_issues = df.drop(columns=['dataset_id'])
    else:
        df_issues = df.copy()

    # Binarize: issue present (1) if value > 0, absent (0) otherwise
    df_binary = (df_issues > 0).astype(int)

    # Compute co-occurrence matrix
    co_occurrence = df_binary.T.dot(df_binary)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(co_occurrence, annot=True, fmt='d', cmap="Oranges",
                cbar_kws={'label': 'Co-occurrence Count'})

    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
   

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return co_occurrence


def generate_quality_issue_boxplots(quality_issues_dataframe, output_directory_path):
    """
    Generate simple individual boxplot visualizations for each quality issue distribution.
    
    This function creates clean, minimal boxplot files for each quality issue, showing
    the distribution across all processed datasets with simple styling.
    
    Parameters
    ----------
    quality_issues_dataframe : pandas.DataFrame
        DataFrame containing quality issue measures for all datasets
    output_directory_path : str
        Path to directory where individual boxplot images will be saved
        
    Returns
    -------
    None
        Individual boxplot visualizations are saved as separate PNG files in output directory
        
    Notes
    -----
    Each boxplot shows:
    - Simple box plot for the specific quality issue
    - Clean minimal styling with grid lines
    - No titles, just ylabel with quality issue name
    - Hidden x-axis ticks for cleaner appearance
    """
    # Define quality issue columns (exclude dataset_id)
    quality_issue_columns = [col for col in quality_issues_dataframe.columns if col != 'dataset_id']
    
    # Create a box plot for each data quality issue
    for quality_issue in quality_issue_columns:
        plt.figure(figsize=(10, 5))
        
        # Drop NaN values in case there are any in the column
        quality_issue_values = pd.to_numeric(quality_issues_dataframe[quality_issue], errors='coerce').dropna()
        
        # Create simple boxplot
        plt.boxplot(quality_issue_values)
        
        # Simple styling
        plt.ylabel(quality_issue, fontsize=23, fontweight='bold')
        plt.xticks([])  # Hide x-axis ticks as it's a single box plot for each plot
        plt.yticks(fontsize=23, fontweight='bold')
        plt.grid(axis='y', linestyle='--')
        
        # Create safe filename (replace spaces and special characters)
        safe_filename = quality_issue.replace(' ', '_').replace('/', '_').replace('\\', '_')
        boxplot_output_path = os.path.join(output_directory_path, f"{safe_filename}_boxplot.png")
        
        # Save the plot
        plt.savefig(boxplot_output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Display success message for each file
        print(f"[SUCCESS] {quality_issue} boxplot saved to: {boxplot_output_path}")
        
        # Close the figure to free memory
        plt.close()
    
    print(f"[INFO] Generated {len(quality_issue_columns)} individual boxplot files")


def generate_quality_issue_statistics_summary(quality_issues_dataframe):
    """
    Generate comprehensive statistical summary for all quality issues.
    
    This function computes descriptive statistics (Min, Max, Mean, S.D., 25%, 75%)
    for each quality issue measure across all processed datasets.
    
    Parameters
    ----------
    quality_issues_dataframe : pandas.DataFrame
        DataFrame containing quality issue measures for all datasets
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing statistical summaries for each quality issue,
        with quality issues as rows and statistics as columns
        
    Notes
    -----
    Generated statistics:
    - Min: Minimum value across all datasets
    - Max: Maximum value across all datasets  
    - Mean: Average value across all datasets
    - S.D.: Standard deviation across all datasets
    - 25%: First quartile (25th percentile)
    - 75%: Third quartile (75th percentile)
    """
    # Define quality issue columns (exclude dataset_id)
    quality_issue_columns = [col for col in quality_issues_dataframe.columns if col != 'dataset_id']
    
    # Initialize dictionary to store statistics for each quality issue
    statistics_summary = {}
    
    # Calculate statistics for each quality issue
    for quality_issue in quality_issue_columns:
        # Extract values for current quality issue (convert to numeric to handle any string values)
        quality_issue_values = pd.to_numeric(quality_issues_dataframe[quality_issue], errors='coerce')
        
        # Remove any NaN values that might result from conversion
        quality_issue_values = quality_issue_values.dropna()
        
        # Calculate descriptive statistics
        statistics_summary[quality_issue] = {
            'Min': quality_issue_values.min(),
            'Max': quality_issue_values.max(),
            'Mean': quality_issue_values.mean(),
            'S.D.': quality_issue_values.std(),
            '25%': quality_issue_values.quantile(0.25),
            '75%': quality_issue_values.quantile(0.75)
        }
    
    # Convert to DataFrame with quality issues as rows and statistics as columns
    statistics_summary_dataframe = pd.DataFrame(statistics_summary).T
    
    # Round values to 4 decimal places for better readability
    statistics_summary_dataframe = statistics_summary_dataframe.round(4)
    
    # Add quality issue names as a column for better identification
    statistics_summary_dataframe.index.name = 'Quality Issue'
    statistics_summary_dataframe = statistics_summary_dataframe.reset_index()
    
    return statistics_summary_dataframe


def process_all_datasets_in_folder(input_directory_path, output_plot_directory_path, output_data_directory_path):
    """
    Process all CSV datasets in input folder and save combined quality issue measures.
    
    This function processes each CSV file in the input directory, extracts
    data quality issues, combines results, and generates statistical summaries.
    
    Parameters
    ----------
    input_directory_path : str
        Path to directory containing input CSV dataset files
    output_directory_path : str
        Path to directory where output files will be saved
        
    Returns
    -------
    None
        Results are saved to multiple CSV files in output directory:
        - 'quality_issues_dataset.csv': Raw quality issue measures for each dataset
        - 'quality_issues_statistics_summary.csv': Statistical summaries for each quality issue
        - Individual boxplot PNG files for each quality issue distribution
        
    Notes
    -----
    - Only processes files with .csv extension
    - Files are processed in sorted order for reproducibility
    - Failed datasets are logged but don't stop processing
    - Final results combined into single CSV file with statistical summary
    """
    successfully_processed_datasets = []
    
    # Get all files in input directory, sorted for reproducibility
    all_files_in_directory = sorted(os.listdir(input_directory_path))
    
    # Process each CSV file in the directory
    for filename in all_files_in_directory:
        if filename.endswith(".csv"):
            # Extract dataset identifier from filename (remove .csv extension)
            dataset_identifier = os.path.splitext(filename)[0]
            
            # Construct full file path
            full_dataset_file_path = os.path.join(input_directory_path, filename)
            
            # Process the dataset and extract quality issue measures
            dataset_quality_issues = process_single_dataset(
                full_dataset_file_path, 
                dataset_identifier
            )
            
            # Add to results list if processing was successful
            if dataset_quality_issues is not None:
                successfully_processed_datasets.append(dataset_quality_issues)
    
    # Combine and save results if any datasets were successfully processed
    if successfully_processed_datasets:
        # Concatenate all quality issue DataFrames
        combined_quality_issues_dataframe = pd.concat(
            successfully_processed_datasets, 
            ignore_index=True
        )

        # Normalize attribute noise
        combined_quality_issues_dataframe['Noise'] = (combined_quality_issues_dataframe['Noise'] - combined_quality_issues_dataframe['Noise'].min()) / (combined_quality_issues_dataframe['Noise'].max() - combined_quality_issues_dataframe['Noise'].min())

        
        # Save combined results to CSV file
        output_file_path = os.path.join(output_data_directory_path, "quality_issues_result.csv")
        combined_quality_issues_dataframe.to_csv(output_file_path, index=False)
        
        print(f"[SUCCESS] Data quality issues extracted and saved to: {output_file_path}")
        print(f"[INFO] Successfully processed {len(successfully_processed_datasets)} datasets")
        
        # Generate and save statistical summary
        print("\n[INFO] Generating statistical summary for quality issues...")
        statistics_summary_dataframe = generate_quality_issue_statistics_summary(
            combined_quality_issues_dataframe
        )
        
        # Save statistical summary to CSV file
        statistics_output_file_path = os.path.join(
            output_data_directory_path, 
            "quality_issues_statistics_summary.csv"
        )
        statistics_summary_dataframe.to_csv(statistics_output_file_path, index=False)
        
        print(f"[SUCCESS] Statistical summary saved to: {statistics_output_file_path}")
        
        # Generate and save boxplot visualizations
        print("\n[INFO] Generating boxplot visualizations for quality issues...")
        generate_quality_issue_boxplots(combined_quality_issues_dataframe, output_plot_directory_path)

        # Generate and save cooccurrence heatmap visualizations
        plot_issue_cooccurrence_heatmap(combined_quality_issues_dataframe.round(2), output_plot_directory_path)
        
        # Display summary statistics in console for quick review
        print("\n" + "="*80)
        print("QUALITY ISSUES STATISTICAL SUMMARY")
        print("="*80)
        print(statistics_summary_dataframe.to_string(index=False))
        print("="*80)
        
    else:
        print("[WARNING] No datasets were successfully processed. Check input directory and file formats.")


if __name__ == "__main__":
    """
    Main execution block - runs when script is executed directly.
    
    This block initiates the data quality issue detection process for all
    datasets in the configured input folder, generates statistical summaries,
    and creates boxplot visualizations.
    """
    print("Starting data quality issue detection process...")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output Plot folder: {OUTPUT_PLOT_FOLDER}")
    print(f"Output Data folder: {OUTPUT_DATA_FOLDER}")
    print("-" * 60)
    
    # Process all datasets in the input folder
    process_all_datasets_in_folder(INPUT_FOLDER, OUTPUT_PLOT_FOLDER, OUTPUT_DATA_FOLDER)
    
    print("-" * 60)
    print("Data quality issue detection process completed.")


