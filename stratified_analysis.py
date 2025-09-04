import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# File paths
QUALITY_ISSUES_FILE = 'Quality_Issue_Extraction_Output/Result_Data/quality_issues_result.csv'
PERFORMANCE_RESULTS_FILE = 'Performance_Analysis_Results/performance_results.csv'

def create_output_directory(primary_issue, model_name):
    """
    Create directory structure: Stratified_Analysis/primary_issue/model_name/
    
    Args:
        primary_issue (str): Primary quality issue name
        model_name (str): Model name
    
    Returns:
        str: Path to the created directory
    """
    # Clean names for folder creation (remove spaces and special characters)
    clean_primary = primary_issue.replace(' ', '_').replace('/', '_')
    clean_model = model_name.replace(' ', '_').replace('/', '_')
    
    # Create directory path including the "Stratified_Analysis" base folder
    output_dir = os.path.join("Stratified_Analysis", clean_primary, clean_model)
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Created output directory: {output_dir}")
    return output_dir

def save_boxplot(fig, output_dir, filename, dpi=300, bbox_inches='tight'):
    """
    Save a matplotlib figure to the specified directory.
    
    Args:
        fig: matplotlib figure object
        output_dir (str): Directory to save the plot
        filename (str): Name of the file (without extension)
        dpi (int): Resolution for saved image
        bbox_inches (str): Bounding box option for saving
    
    Returns:
        str: Full path of saved file
    """
    # Ensure filename has .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    
    # Create full path
    filepath = os.path.join(output_dir, filename)
    
    # Save the figure
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    #print(f"💾 Saved plot: {filepath}")
    
    return filepath

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
    try:
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
        
        return merged_dataset
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        raise
    except KeyError as e:
        print(f"Error: Missing required column - {e}")
        raise

def multi_stratified_interaction_analysis(merged_dataset, primary_quality_issue, model_name):
    """
    Perform stratified interaction analysis between one primary quality issue and all other quality issues.
    
    Args:
        merged_dataset (pd.DataFrame): The merged dataset from load_and_merge_datasets()
        primary_quality_issue (str): Primary quality issue for x-axis stratification
                                     Options: 'Class Imbalance', 'Class Overlap', 'Noise', 
                                              'Irrelevant Features', 'Outlier'
        model_name (str): Name of the classification model to analyze
                          Options: 'DT', 'RF', 'SVM', 'NB', 'MLP'
    
    Returns:
        dict: Dictionary containing analysis results for all interactions
    """
    
    # Validate inputs
    quality_issues = ['Class Imbalance', 'Class Overlap', 'Noise', 'Irrelevant Features', 'Outlier']
    models = ['DT', 'RF', 'SVM', 'NB', 'MLP']
    
    if primary_quality_issue not in quality_issues:
        raise ValueError(f"Invalid primary quality issue. Choose from: {quality_issues}")
    
    if model_name not in models:
        raise ValueError(f"Invalid model name. Choose from: {models}")
    
    # Create output directory
    output_dir = create_output_directory(primary_quality_issue, model_name)
    
    # Get all secondary quality issues (all except primary)
    secondary_quality_issues = [issue for issue in quality_issues if issue != primary_quality_issue]
    
    # Store results for all interactions
    all_results = {}
    
    print(f"🔍 MULTI-STRATIFIED INTERACTION ANALYSIS")
    print(f"Primary Quality Issue: {primary_quality_issue}")
    print(f"Model: {model_name}")
    print(f"Secondary Issues to analyze: {secondary_quality_issues}")
    print("=" * 90)
    
    # Perform analysis for each secondary quality issue
    for secondary_issue in secondary_quality_issues:
        print(f"\n📊 Analyzing interaction: {primary_quality_issue} vs {secondary_issue}")
        print("-" * 60)
        
        try:
            # Perform individual stratified analysis
            results = stratified_interaction_analysis(
                merged_dataset, 
                primary_quality_issue, 
                secondary_issue, 
                model_name,
                output_dir, # Pass output directory
                show_plots=False # Don't show individual plots yet as we'll handle saving in multi-stratified
            )
            
            all_results[secondary_issue] = results
            
            # Print brief summary for this interaction
            print_brief_interaction_summary(primary_quality_issue, secondary_issue, results)
            
        except Exception as e:
            print(f"❌ Error analyzing {primary_quality_issue} vs {secondary_issue}: {e}")
            all_results[secondary_issue] = {'error': str(e)}
    
    # Create comprehensive visualizations and save them (now as individual plots)
    create_multi_stratified_visualizations(
        merged_dataset, 
        primary_quality_issue, 
        secondary_quality_issues, 
        model_name,
        output_dir # Pass output directory
    )
    
    # Print comprehensive summary
    #print_comprehensive_summary(primary_quality_issue, model_name, all_results)
    
    return all_results

def stratified_interaction_analysis(merged_dataset, primary_quality_issue, secondary_quality_issue, model_name, output_dir, show_plots=True):
    """
    Perform stratified interaction analysis between two quality issues on model performance.
    Modified to save plots to specified directory.
    """
    
    # Validate inputs
    quality_issues = ['Class Imbalance', 'Class Overlap', 'Noise', 'Irrelevant Features', 'Outlier']
    models = ['DT', 'RF', 'SVM', 'NB', 'MLP']
    
    if primary_quality_issue not in quality_issues:
        raise ValueError(f"Invalid primary quality issue. Choose from: {quality_issues}")
    
    if secondary_quality_issue not in quality_issues:
        raise ValueError(f"Invalid secondary quality issue. Choose from: {quality_issues}")
        
    if primary_quality_issue == secondary_quality_issue:
        raise ValueError("Primary and secondary quality issues must be different")
    
    if model_name not in models:
        raise ValueError(f"Invalid model name. Choose from: {models}")
    
    # Create a copy of the dataset for analysis
    df = merged_dataset.copy()
    
    # Define the target performance column
    performance_col = f'Balanced_Acc_{model_name}'
    
    if performance_col not in df.columns:
        raise ValueError(f"Performance column {performance_col} not found in dataset")
    
    # Create tertiles for primary quality issue (x-axis)
    try:
        df[f'{primary_quality_issue}_tertile'] = pd.qcut(
            df[primary_quality_issue], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
    except ValueError as e:
        raise ValueError(f"Cannot create tertiles for {primary_quality_issue} due to duplicate values. "
                         f"The data may not have sufficient variation for tertile analysis. "
                         f"Original error: {str(e)}")
    
    # Create tertiles for secondary quality issue (hue)
    try:
        df[f'{secondary_quality_issue}_tertile'] = pd.qcut(
            df[secondary_quality_issue], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
    except ValueError as e:
        raise ValueError(f"Cannot create tertiles for {secondary_quality_issue} due to duplicate values. "
                         f"The data may not have sufficient variation for tertile analysis. "
                         f"Original error: {str(e)}")
    
    # Remove any NaN values in tertiles
    df = df.dropna(subset=[f'{primary_quality_issue}_tertile', f'{secondary_quality_issue}_tertile'])
    
    # Perform statistical analysis
    results = perform_interaction_analysis(df, primary_quality_issue, secondary_quality_issue, performance_col)
    
    # Create visualizations and save them
    if show_plots:
        create_stratified_visualizations(df, primary_quality_issue, secondary_quality_issue, performance_col, model_name, output_dir)
    
    return results



def create_multi_stratified_visualizations(merged_dataset, primary_issue, secondary_issues, model_name, output_dir):
    """
    Create comprehensive visualizations showing primary issue stratified against all secondary issues.
    Modified to save each plot individually to the specified directory.
    """
    performance_col = f'Balanced_Acc_{model_name}'
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a separate subplot for each secondary issue
    for secondary_issue in secondary_issues:
        fig, ax = plt.subplots(figsize=(10, 6)) # Create a new figure and axes for each plot
        try:
            # Prepare data for this interaction
            df = merged_dataset.copy()
            
            # Create tertiles
            df[f'{primary_issue}_tertile'] = pd.qcut(
                df[primary_issue], q=3, labels=['Low', 'Medium', 'High']
            )
            df[f'{secondary_issue}_tertile'] = pd.qcut(
                df[secondary_issue], q=3, labels=['Low', 'Medium', 'High']
            )
            
            # Remove NaN values
            df = df.dropna(subset=[f'{primary_issue}_tertile', f'{secondary_issue}_tertile'])
            
            # Create boxplot on the current axes (ax)
            sns.boxplot(
                data=df,
                x=f'{primary_issue}_tertile',
                y=performance_col,
                hue=f'{secondary_issue}_tertile',
                ax=ax, # Plot on the specific axis
                palette='Set2'
            )
            
            #ax.set_title(f'Stratified Analysis: {primary_issue} vs {secondary_issue}\n{model_name} Performance', fontweight='bold')
            
            # Correct way to set tick label formatting
            ax.tick_params(axis='x', labelsize=23)
            ax.tick_params(axis='y', labelsize=23)
            
            # Make tick labels bold
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')


            ax.set_xlabel(f'{primary_issue} Level', fontsize=23, fontweight='bold')
            ax.set_ylabel('Balanced Accuracy', fontsize=23, fontweight='bold')
            ax.legend(title=f'{secondary_issue}', title_fontsize=15, prop={'size': 15})
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                             ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{primary_issue} vs {secondary_issue} (Error)')
        
        plt.tight_layout()
        
        # Save each individual boxplot
        clean_secondary = secondary_issue.replace(' ', '_').replace('/', '_')
        save_boxplot(fig, output_dir, f'{primary_issue.replace(" ", "_")}_vs_{clean_secondary}_{model_name}_boxplot')
        #plt.show() # Show the plot
        plt.close(fig) # Close the figure to free memory
        
    # Create interaction strength comparison chart (this will still be a single plot)
    create_interaction_strength_comparison(merged_dataset, primary_issue, secondary_issues, model_name, output_dir)


def create_interaction_strength_comparison(merged_dataset, primary_issue, secondary_issues, model_name, output_dir):
    """
    Create a comparison chart of interaction strengths between primary issue and all secondary issues.
    Modified to save the plot.
    """
    performance_col = f'Balanced_Acc_{model_name}'
    interaction_strengths = []
    
    # Calculate interaction strength for each secondary issue
    for secondary_issue in secondary_issues:
        try:
            df = merged_dataset.copy()
            
            # Create tertiles
            df[f'{primary_issue}_tertile'] = pd.qcut(
                df[primary_issue], q=3, labels=['Low', 'Medium', 'High']
            )
            df[f'{secondary_issue}_tertile'] = pd.qcut(
                df[secondary_issue], q=3, labels=['Low', 'Medium', 'High']
            )
            
            # Calculate interaction strength
            strength_results = calculate_interaction_strength(
                df, f'{primary_issue}_tertile', f'{secondary_issue}_tertile', performance_col
            )
            
            if 'average_interaction' in strength_results:
                interaction_strengths.append({
                    'Secondary_Issue': secondary_issue,
                    'Interaction_Strength': strength_results['average_interaction']
                })
        except:
            interaction_strengths.append({
                'Secondary_Issue': secondary_issue,
                'Interaction_Strength': 0
            })
    
 

def print_brief_interaction_summary(primary_issue, secondary_issue, results):
    """
    Print a brief summary for individual interaction analysis.
    """
    print(f"✅ Analysis completed for {primary_issue} vs {secondary_issue}")
    
    if 'anova' in results and 'p_value' in results['anova']:
        p_val = results['anova']['p_value']
        significance = "Significant" if p_val < 0.05 else "Not significant"
        print(f"    ANOVA p-value: {p_val:.4f} ({significance})")
    
    if 'interaction_strength' in results and 'average_interaction' in results['interaction_strength']:
        strength = results['interaction_strength']['average_interaction']
        #print(f"    Interaction strength: {strength:.4f}")

def print_comprehensive_summary(primary_issue, model_name, all_results):
    """
    Print comprehensive summary comparing all interactions.
    """
    print("\n" + "=" * 100)
    print(f"🎯 COMPREHENSIVE MULTI-STRATIFIED ANALYSIS SUMMARY")
    print(f"Primary Quality Issue: {primary_issue}")
    print(f"Model: {model_name}")
    print("=" * 100)
    
    # Create summary table
    summary_data = []
    for secondary_issue, results in all_results.items():
        if 'error' not in results:
            # ANOVA results
            anova_p = results.get('anova', {}).get('p_value', 'N/A')
            anova_sig = "Yes" if (anova_p != 'N/A' and anova_p < 0.05) else "No"
            
            # Interaction strength
            interaction_strength = results.get('interaction_strength', {}).get('average_interaction', 'N/A')
            
            # Strength category
            if interaction_strength != 'N/A':
                if interaction_strength < 0.02:
                    strength_cat = "Weak"
                elif interaction_strength < 0.05:
                    strength_cat = "Moderate"
                else:
                    strength_cat = "Strong"
            else:
                strength_cat = "N/A"
            
            summary_data.append({
                'Secondary_Issue': secondary_issue,
                'ANOVA_p_value': f"{anova_p:.4f}" if anova_p != 'N/A' else 'N/A',
                'Significant': anova_sig,
                'Interaction_Strength': f"{interaction_strength:.4f}" if interaction_strength != 'N/A' else 'N/A',
                'Strength_Category': strength_cat
            })
        else:
            summary_data.append({
                'Secondary_Issue': secondary_issue,
                'ANOVA_p_value': 'Error',
                'Significant': 'Error',
                'Interaction_Strength': 'Error',
                'Strength_Category': 'Error'
            })
    
    # Print summary table
    summary_df = pd.DataFrame(summary_data)
    print("\n📋 INTERACTION SUMMARY TABLE:")
    print(summary_df.to_string(index=False))
    
    # Identify strongest interactions
    valid_interactions = [row for row in summary_data if row['Strength_Category'] not in ['N/A', 'Error']]
    if valid_interactions:
        # Sort by interaction strength
        sorted_interactions = sorted(valid_interactions, 
                                     key=lambda x: float(x['Interaction_Strength']), 
                                     reverse=True)
        
        print(f"\n🏆 RANKING BY INTERACTION STRENGTH:")
        for i, interaction in enumerate(sorted_interactions, 1):
            print(f"{i}. {primary_issue} × {interaction['Secondary_Issue']}: "
                  f"{interaction['Interaction_Strength']} ({interaction['Strength_Category']})")
        
        # Identify significant interactions
        significant_interactions = [row for row in summary_data if row['Significant'] == 'Yes']
        if significant_interactions:
            print(f"\n📊 STATISTICALLY SIGNIFICANT INTERACTIONS:")
            for interaction in significant_interactions:
                print(f"    • {primary_issue} × {interaction['Secondary_Issue']} "
                      f"(p = {interaction['ANOVA_p_value']})")
        else:
            print(f"\n📊 No statistically significant interactions found (p < 0.05)")

def perform_interaction_analysis(df, primary_issue, secondary_issue, performance_col):
    """
    Perform statistical tests for interaction effects between two quality issues.
    """
    results = {}
    
    primary_tertile = f'{primary_issue}_tertile'
    secondary_tertile = f'{secondary_issue}_tertile'
    
    # Create combination groups for detailed analysis
    df['combination_group'] = df[primary_tertile].astype(str) + '_' + df[secondary_tertile].astype(str)
    
    # Descriptive statistics by combination groups
    stats_by_combination = df.groupby(['combination_group'])[performance_col].agg([
        'count', 'mean', 'median', 'std'
    ]).round(4)
    results['combination_stats'] = stats_by_combination
    
    # Descriptive statistics by primary issue tertiles
    stats_by_primary = df.groupby(primary_tertile)[performance_col].agg([
        'count', 'mean', 'median', 'std'
    ]).round(4)
    results['primary_stats'] = stats_by_primary
    
    # Descriptive statistics by secondary issue tertiles  
    stats_by_secondary = df.groupby(secondary_tertile)[performance_col].agg([
        'count', 'mean', 'median', 'std'
    ]).round(4)
    results['secondary_stats'] = stats_by_secondary
    
    # Two-way ANOVA for interaction effects
    try:
        # Prepare data for ANOVA
        groups_data = []
        group_labels = []
        
        for primary_level in ['Low', 'Medium', 'High']:
            for secondary_level in ['Low', 'Medium', 'High']:
                group_data = df[
                    (df[primary_tertile] == primary_level) & 
                    (df[secondary_tertile] == secondary_level)
                ][performance_col].dropna()
                
                if len(group_data) > 0:
                    groups_data.append(group_data.values)
                    group_labels.append(f"{primary_level}_{secondary_level}")
        
        # Perform ANOVA if we have enough groups
        if len(groups_data) >= 3:
            f_stat, p_value = stats.f_oneway(*groups_data)
            results['anova'] = {
                'f_statistic': f_stat, 
                'p_value': p_value,
                'groups_tested': len(groups_data)
            }
    except Exception as e:
        results['anova'] = {'error': str(e)}
    
    # Calculate interaction effect strength
    interaction_strength = calculate_interaction_strength(df, primary_tertile, secondary_tertile, performance_col)
    results['interaction_strength'] = interaction_strength
    
    return results

def calculate_interaction_strength(df, primary_tertile, secondary_tertile, performance_col):
    """
    Calculate a measure of interaction strength between the two quality issues.
    """
    interaction_effects = {}
    
    for primary_level in ['Low', 'Medium', 'High']:
        # Get performance means for each secondary level within this primary level
        means_within_primary = []
        for secondary_level in ['Low', 'Medium', 'High']:
            subset = df[
                (df[primary_tertile] == primary_level) & 
                (df[secondary_tertile] == secondary_level)
            ][performance_col]
            
            if len(subset) > 0:
                means_within_primary.append(subset.mean())
            
        # Calculate range of means within this primary level
        if len(means_within_primary) > 1:
            range_within_primary = max(means_within_primary) - min(means_within_primary)
            interaction_effects[f'{primary_level}_range'] = range_within_primary
    
    # Overall interaction strength is the average range across primary levels
    if interaction_effects:
        avg_interaction = np.mean(list(interaction_effects.values()))
        interaction_effects['average_interaction'] = avg_interaction
    
    return interaction_effects

def create_stratified_visualizations(df, primary_issue, secondary_issue, performance_col, model_name, output_dir):
    """
    Create visualizations for stratified interaction analysis.
    Modified to save plots to specified directory.
    """
    primary_tertile = f'{primary_issue}_tertile'
    secondary_tertile = f'{secondary_issue}_tertile'
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    #fig.suptitle(f'Stratified Interaction Analysis: {primary_issue} vs {secondary_issue}\n{model_name} Performance', fontsize=16, fontweight='bold')
    
    # 1. Main stratified boxplot
    sns.boxplot(
        data=df,
        x=primary_tertile,
        y=performance_col,
        hue=secondary_tertile,
        ax=axes[0, 0],
        palette='Set2'
    )
    axes[0, 0].set_title(f'Stratified Analysis: {model_name} Performance')
    axes[0, 0].set_xlabel(f'{primary_issue} Level', fontweight='bold')
    axes[0, 0].set_ylabel('Balanced Accuracy', fontweight='bold')
    axes[0, 0].legend(title=f'{secondary_issue} Level', title_fontsize=12, prop={'size': 12})
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Interaction plot (line plot)
    for secondary_level in ['Low', 'Medium', 'High']:
        means_by_primary = []
        primary_levels = ['Low', 'Medium', 'High']
        
        for primary_level in primary_levels:
            subset = df[
                (df[primary_tertile] == primary_level) & 
                (df[secondary_tertile] == secondary_level)
            ][performance_col]
            
            if len(subset) > 0:
                means_by_primary.append(subset.mean())
            else:
                means_by_primary.append(np.nan)
        
        axes[0, 1].plot(primary_levels, means_by_primary, marker='o', linewidth=2, 
                         label=f'{secondary_issue}: {secondary_level}')
    
    axes[0, 1].set_title('Interaction Effect Plot')
    axes[0, 1].set_xlabel(f'{primary_issue} Level', fontweight='bold')
    axes[0, 1].set_ylabel('Mean Balanced Accuracy', fontweight='bold')
    axes[0, 1].legend(title=f'{secondary_issue} Level')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Heatmap of mean performance by combination
    pivot_data = df.groupby([secondary_tertile, primary_tertile])[performance_col].mean().unstack()
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        ax=axes[1, 0],
        cbar_kws={'label': 'Mean Balanced Accuracy'}
    )
    axes[1, 0].set_title('Performance Heatmap by Quality Issue Combination')
    axes[1, 0].set_xlabel(f'{primary_issue} Level', fontweight='bold')
    axes[1, 0].set_ylabel(f'{secondary_issue} Level', fontweight='bold')
    
    # 4. Violin plot for distribution comparison
    sns.violinplot(
        data=df,
        x=primary_tertile,
        y=performance_col,
        hue=secondary_tertile,
        ax=axes[1, 1],
        palette='Set2'
    )
    axes[1, 1].set_title('Performance Distribution by Stratified Groups')
    axes[1, 1].set_xlabel(f'{primary_issue} Level', fontweight='bold')
    axes[1, 1].set_ylabel('Balanced Accuracy', fontweight='bold')
    axes[1, 1].legend(title=f'{secondary_issue} Level', title_fontsize=12, prop={'size': 12})
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the detailed stratified analysis plot
    clean_secondary = secondary_issue.replace(' ', '_').replace('/', '_')
    save_boxplot(fig, output_dir, f'detailed_analysis_{clean_secondary}')
    #plt.show()
    plt.close()

# Main execution function
def main():
    """
    Main function to demonstrate the multi-stratified interaction analysis.
    """
    try:
        # Load and merge datasets
        print("Loading and merging datasets...")
        merged_data = load_and_merge_datasets()
        print(f"Dataset loaded successfully. Shape: {merged_data.shape}")
        
        # Example analysis - modify these parameters
        primary_quality_issue = "Outlier"   # Primary quality issue to stratify against all others
        model_name = "SVM"                           # Model to analyze
        
        print(f"\nPerforming multi-stratified interaction analysis...")
        print(f"Primary: {primary_quality_issue} vs ALL other quality issues for {model_name}")
        
        # Perform multi-stratified interaction analysis
        results = multi_stratified_interaction_analysis(
            merged_data, 
            primary_quality_issue, 
            model_name
        )
        
        print("\nMulti-stratified analysis completed successfully!")
        # Updated message for new folder structure
        print(f"All plots saved in: Stratified_Analysis/{primary_quality_issue.replace(' ', '_')}/{model_name}/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()

# Example usage:
"""
# Load data once
merged_data = load_and_merge_datasets()

# Analyze how 'Outlier' interacts with all other quality issues for Random Forest
# Plots will be saved in: Stratified_Analysis/Outlier/RF/
results1 = multi_stratified_interaction_analysis(merged_data, "Outlier", "RF")

# Analyze how 'Class Imbalance' interacts with all other quality issues for SVM
# Plots will be saved in: Stratified_Analysis/Class_Imbalance/SVM/

"""


