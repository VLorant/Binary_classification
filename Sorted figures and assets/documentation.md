# Research Data and Figures Documentation

This directory contains the organized and descriptively named data and plots for the Binary Classification research. The files have been extracted from the `scriptek/` directory for better structural overview and accessibility.

## Directory Structure

### `data/`
Contains the datasets used for analysis and visualization.
- **`raw_results/`**: Original CSV result files for the 17 classifiers.
    - *Naming format:* `Raw_Results_[Classifier].csv`
- **`processed/`**: Aggregated results grouped by noise level (`Z`) and consolidated findings.
    - *General format:* `Aggregated_Metrics_By_Noise_[Classifier].csv`
    - `Consolidated_Balanced_Results.csv`: Aggregated results for Imbalance Ratio < 4.
    - `Consolidated_NoiseFree_Results.csv`: Aggregated results for Noise Level Z = 0.
- **`summaries/`**: High-level statistical summaries.
    - `Global_Aggregate_Metrics.csv`: Combined aggregate metrics across all classifiers.
    - `Metric_Correlation_Summary.csv`: Correlation analysis between different performance metrics.
    - `Mean_Performance_Overview.csv`: Averaged performance metrics for each classifier.
    - `Classifier_Rank_Comparison.csv`: Numerical ranking of classifiers across all metrics.

### `plots/`
Contains the generated visualizations categorized by their research context.
- **`balanced/`**: Analysis of scenarios with low imbalance (IR < 4).
    - `Critical_Difference_Balanced_[Metric]_cd_aeon.png`: CD diagrams for specific metrics.
    - `Violin_Distribution_Balanced.png`: Violin plot showing score distributions.
- **`noise_free/`**: Analysis of scenarios with no noise (Z = 0).
    - `Critical_Difference_NoiseFree_[Metric]_cd_aeon.png`: CD diagrams for specific metrics.
- **`general/`**: Global visualizations across all experimental variables.
    - `Critical_Difference_Global_[Metric]_cd_aeon.png`: CD diagrams for global performance.
    - `Global_Violin_All_Classifiers.png`: Comprehensive distribution plot.
    - `Classifier_Performance_Ranks_Table.png`: Visual table of classifier rankings.
    - `Metric_Distribution_Violin_Plot.png`: Primary violin plot of performance metrics.

## Data Provenance
All data and plots in this directory are derived from the files in `scriptek/eredmenyek_teljes_IR/`.
- The `processed/` data is generated via aggregation scripts (e.g., `agg_new.py`).
- The `summaries/` are produced by statistical evaluation scripts (e.g., `for_test.py`, `stat_test2.py`).
- The plots are generated using various visualization scripts (e.g., `violin.py`, `saj_test_frid_cd_diag.py`, `for_noise_free_CD.py`).

---
