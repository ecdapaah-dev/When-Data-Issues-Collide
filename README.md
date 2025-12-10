# When Data Issues Collide ‒ Reproduction Guide

This guide explains the purpose of the **When‑Data‑Issues‑Collide** project and provides detailed instructions to help researchers reproduce the analyses contained in the repository. The project explores how different data quality problems—such as class imbalance, class overlap, noise, irrelevant features and outliers—affect the performance of machine‑learning classifiers on software defect prediction (SDP) datasets. The analyses include extracting data quality metrics, training multiple classifiers, building Explainable Boosting Machine (EBM) models to study how data issues influence accuracy, and conducting stratified interaction analyses.

### Project overview

The repository contains a collection of scripts and datasets used in a data‑quality study. Each dataset in the `SDP_Dataset_Pool` directory is a **CSV** file containing code metrics and a `target` column indicating whether a software module is defective. An example dataset shows features such as McCC, CLOC, PDA and other metrics along with the `target` column.

The study is divided into four main stages:

**1. Data quality issue extraction** – Compute five data quality indicators for each dataset.

**2. Performance analysis** – Evaluate five standard classification algorithms on each dataset and record balanced accuracy.

**3. EBM influence analysis** – Merge the quality and performance data and train Explainable Boosting Machine models to understand how quality issues influence accuracy.

**4. Stratified interaction analysis** – Examine how pairs of quality issues jointly affect classifier performance using statistical tests and visualizations.

This guide walks through the required dependencies and the procedure to reproduce each stage.

### Repository structure

The top‑level repository contains the following key folders and scripts:

| Item                               | Purpose                                                                                                                                                                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SDP_Dataset_Pool/`                | Raw SDP datasets. Each file (e.g., `0.csv`, `1.csv`, …) contains software metrics and a `target` column indicating whether the instance is defective.                                                                                         |
| `Quality_Issue_Extraction_Output/` | Contains the results of quality issue extraction. `Result_Data/quality_issues_result.csv` holds the computed quality measures per dataset, and `quality_issues_statistics_summary.csv` provides descriptive statistics.                       |
| `Performance_Analysis_Results/`    | Contains classifier performance results. `performance_results.csv` lists the balanced accuracy of each algorithm for every dataset, and `performance_statistics_summary.csv` summarizes minimum, maximum, mean and quartiles across datasets. |
| `EBM_Influence_Analysis/`          | Output directory for EBM models. Contains plots and feature‑importance files produced by `ebm_influence_analysis.py`.                                                                                                                         |
| `Stratified_Analysis/`             | The script `stratified_analysis.py` creates sub‑folders under this directory when producing stratified interaction plots.                                                                                                                     |
| `quality_issue_extraction.py`      | Extracts data quality metrics using `pymfe` and a custom outlier detector.                                                                                                                                                                    |
| `performance_analysis.py`          | Runs multiple classifiers on each dataset and computes balanced accuracy.                                                                                                                                                                     |
| `ebm_influence_analysis.py`        | Trains EBM models to predict accuracy from quality metrics.                                                                                                                                                                                   |
| `stratified_analysis.py`           | Performs stratified interaction analysis between pairs of quality issues and model performance.                                                                                                                                               |

### Environment setup

The scripts require **Python 3.8–3.11** and several scientific libraries. The following command creates a reproducible environment using `pip` (a Conda environment may also be used):

Create and activate a virtual environment (optional but recommended):

 `python3 -m venv env `
 
 `source env/bin/activate `

Upgrade pip and install dependencies:

 `python -m pip install --upgrade pip `

 `python -m pip install pandas numpy scikit-learn matplotlib seaborn pymfe interpret scipy statsmodels `

 The `interpret` package provides the Explainable Boosting Machine (EBM) model used in later stages, while `pymfe` extracts complexity‑based meta‑features for quality‑issue detection. If using `pymfe`, also install its optional dependencies (e.g., `scikit-learn`) to avoid runtime warnings.


### Stage 1 – Extracting data quality issues

The script `quality_issue_extraction.py` computes five data quality indicators for each dataset:

- **Class imbalance** (`c2`) – imbalance ratio (lower values indicate balanced classes).

- **Class overlap** (`n1`) – fraction of borderline instances representing class overlap.

- **Noise** (`ns_ratio`) – attribute noise level; the script normalizes this measure between 0 and 1 before saving.

- **Irrelevant features** (`mut_inf`) – average mutual information between each feature and the target; lower values indicate more irrelevant features.

- **Outlier count** – number of outliers detected using an interquartile‑range (IQR) method.

### Running the extraction script

1. Place all raw datasets (CSV files with a `target` column) inside `SDP_Dataset_Pool/` or update the `INPUT_FOLDER` in the script.

2. Run the script from the project root: `python quality_issue_extraction.py`

The script iterates through every CSV file, splits the data into train and test subsets and computes the five quality measures for the training portion. Results for each dataset are concatenated and saved to `Quality_Issue_Extraction_Output/Result_Data/quality_issues_result.csv`. It also produces summary statistics (`quality_issues_statistics_summary.csv`) and individual box plots for each quality issue in `Quality_Issue_Extraction_Output/Plots/`.

The resulting `quality_issues_result.csv` contains columns `Class Imbalance`, `Class Overlap`, `Noise`, `Irrelevant Features`, `Outlier` and `dataset_id`. These columns are essential for subsequent analyses.


### Stage 2 – Evaluating classifier performance

The script `performance_analysis.py` evaluates five classifiers on every dataset:

- **Decision Tree (DT)**

- **Random Forest (RF)**

- **Support Vector Machine (SVM)**

- **Gaussian Naive Bayes (NB)**

- **Multi‑Layer Perceptron (MLP)**

For each dataset, the script loads the features and target, performs a 80/20 stratified train–test split, trains each classifier using a fixed random seed to ensure reproducibility, predicts on the test set and records the **balanced accuracy**. All results are compiled into a single DataFrame and saved to Performance_Analysis_Results/performance_results.csv. A summary file performance_statistics_summary.csv reports the minimum, maximum, mean and quartiles for each algorithm across datasets.


### Running the performance analysis script

Run the script from the project root: `python performance_analysis.py`.

The script automatically discovers all numerical filenames in `SDP_Dataset_Pool/`. Make sure the dataset files are named with integers (e.g., `0.csv`, `1.csv`, …). The script prints progress messages as it trains each algorithm on each dataset and writes the consolidated results to `Performance_Analysis_Results/performance_results.csv`.


### Stage 3 – EBM influence analysis

After obtaining the quality‑issue metrics and performance results, `ebm_influence_analysis.py` explores how data quality issues relate to classifier performance using Explainable Boosting Machines. The script merges `quality_issues_result.csv` and `performance_results.csv` on the dataset_id, yielding a combined dataset with quality features and balanced accuracy scores.

For each classifier (e.g., `Balanced_Acc_DT`, `Balanced_Acc_RF`, etc.), the script trains an **Explainable Boosting Regressor** to predict balanced accuracy from the five quality metrics. The EBM models use monotonicity constraints to improve interpretability. After training, the script reports mean absolute error, mean squared error and root mean squared error for each model and then generates **shape‑function plots** showing how each quality issue influences accuracy. Feature‑importance values are also exported to CSV files.


### Running the EBM analysis

Ensure that `Quality_Issue_Extraction_Output/Result_Data/quality_issues_result.csv` and `Performance_Analysis_Results/performance_results.csv` exist. Then run:

`python ebm_influence_analysis.py`

The script writes merged data to `EBM_Influence_Analysis/Result_Data/issue_performance_merged.csv`, produces shape‑function comparison plots for each quality issue in `EBM_Influence_Analysis/Plots/`, and stores feature‑importance tables in the same directory. Reviewing these plots helps interpret whether increases in, say, class imbalance or noise generally improve or degrade balanced accuracy.


### Stage 4 – Stratified interaction analysis

Whereas the EBM models provide overall trends, the stratified analysis examines how two quality issues jointly influence performance. The script `stratified_analysis.py` merges the quality and performance data, then divides a primary quality issue and a secondary quality issue into **tertiles** (low/medium/high) using quantile cuts. It then performs a two‑way ANOVA to test interaction effects and computes an **interaction strength** metric—the average range of performance differences within the tertiles. Visualizations include box plots, interaction plots and heatmaps to illustrate the relationships between the issues.


### Running a stratified analysis

This script is more exploratory and requires specifying both a primary quality issue and a classifier. For example, to study how *class imbalance interacts* with other issues for the Decision Tree model:

`from stratified_analysis import load_and_merge_datasets, multi_stratified_interaction_analysis`

`merged = load_and_merge_datasets()`
`multi_stratified_interaction_analysis(merged_dataset=merged, primary_quality_issue='Class Imbalance', model_name='DT')`

The function creates a sub‑folder `Stratified_Analysis/Class_Imbalance/DT/` and saves multiple plots there. The primary issue must be one of `Class Imbalance`, `Class Overlap`, `Noise`, `Irrelevant Features` or `Outlier`, and the model name must match one of `DT`, `RF`, `SVM`, `NB`, or `MLP`. When the data contain insufficient variation to create tertiles, the script raises an informative error. The results dictionary returned by `multi_stratified_interaction_analysis` includes descriptive statistics, ANOVA results and interaction strength for each pair.


### Suggested workflow

To reproduce the full study:

1. **Install dependencies** using the commands in the *Environment setup* section.

2. **Prepare datasets** in `SDP_Dataset_Pool/`. Ensure each CSV includes a `target` column with binary labels and is named with an integer identifier.

3. **Run** `quality_issue_extraction.py` to compute and save quality metrics. Examine the generated box plots and the `quality_issues_statistics_summary.csv` for an overview of quality characteristics.

4. **Run** `performance_analysis.py` to train and evaluate the five classifiers. Review `performance_statistics_summary.csv` to understand baseline performance.

5. **Run** `ebm_influence_analysis.py` to build EBM models that relate quality metrics to performance. Inspect the shape‑function plots and feature‑importance tables to identify which issues most influence each classifier.

6. **Conduct targeted stratified analyses** using `stratified_analysis.py` to explore interactions between specific pairs of quality issues for the classifier(s) of interest. Use the `multi_stratified_interaction_analysis` function to automatically generate all pairwise analyses for a given primary issue and model.

Following this sequence ensures that each stage has the required inputs and yields outputs compatible with subsequent analyses.


### Additional notes & troubleshooting

- **Dataset format:** Each CSV must contain only numeric feature columns and a binary `target` column. Non‑numeric data or missing labels will cause the scripts to fail.

- **Naming convention:** The performance analysis script identifies datasets by numeric file names. Files with non‑numeric names will be ignored.

- **Random seeds:** All training procedures set `random_state` to 42 to ensure reproducibility. If you wish to evaluate different seeds, modify the script accordingly.

- **Dependencies:** The `interpret` package can be heavy to install; ensure a compatible version of `scikit‑learn` is present. The `pymfe` library may issue warnings if optional dependencies are missing.

- **Large datasets:** When processing large datasets, extracting meta‑features and training models can be time‑consuming. Consider limiting the number of datasets or using a machine with sufficient memory.


### Citation

Please cite the repository and associated publication if you use this reproduction guide in your own work.

