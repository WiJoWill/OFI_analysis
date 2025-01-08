# Order Flow Imbalance Analysis and Visualization

## Description
This project analyzes and visualizes the impact of **Order Flow Imbalance (OFI)** on equity price movements. The pipeline integrates data processing, regression analysis, and visualization to evaluate both contemporaneous and predictive relationships between OFI metrics and short-term price changes.

### Key Features
- **Heatmaps**: Display regression coefficients across stocks and time horizons.
- **Scatter Plots**: Show \( R^2 \) values over multiple trading days.
- **Pipeline**: Easy-to-use scripts for data preprocessing, analysis, and visualization.

## Steps to Run the Analysis

### Prerequisites
Ensure you have Python (version >= 3.8) and the required libraries installed. Use `requirements.txt` to set up your environment:
```bash
pip install -r requirements.txt
```

### 1. Prepare Input Data
- Place raw data files for OFI computation in `../data/raw_data/`.
- Run the OFI calculation script to process the raw data and compute OFI metrics:
```bash
python OFI_calculation.py --file_path ../data/raw_data/sample_data.csv --pca --levels 5
```
Explanation:
- `--file_path`: Specifies the input file path for raw data.
- `--pca`: Optional flag to integrate OFI metrics using PCA.
- `--levels`: Defines the number of levels in the order book for computation.

The processed OFI metrics will be saved in `../data/ofi_metrics/`.
- Store processed OFI metrics in `../data/ofi_metrics/`.

### 2. Run Cross-Impact Analysis
Run the script to perform contemporaneous and predictive regression analysis:
```bash
python cross_impact_analysis.py
```
Explanation:
- This script evaluates the explanatory power of OFI metrics for contemporaneous price changes and the predictive power over different time horizons (e.g., 1 and 5 minutes).
- Results are saved in `../data/contemporaneous_regression_results/` and `../data/predictive_regression_results/`.

### 3. Generate Visualizations
Run the script to create heatmaps and scatter plots for the analysis results:
```bash
python visualization.py
```
Explanation:
- Heatmaps display regression coefficients for cross-impact and predictive analyses.
- Scatter plots illustrate \( R^2 \) values across trading days for different time horizons.
- Outputs are saved in `../data/visualizations/`.

### Output
- Regression results are saved in:
  - `../data/contemporaneous_regression_results/`
  - `../data/predictive_regression_results/`
- Visualizations (heatmaps and scatter plots) are saved in `../data/visualizations/`.

## Summary of Findings

### Explanatory Power
- Stocks like **GILD** and **V** exhibit strong explanatory relationships with \( R^2 \) values of 0.1179 and 0.1698, respectively.
- Heatmaps indicate significant cross-impact coefficients for stocks like **AMGN** and **JPM**.

### Predictive Power
- Short-term horizons (1 minute) show strong predictive power, with peak \( R^2 \) values for stocks like **GILD** and **V**.
- Longer horizons (5 minutes) exhibit diminishing predictive strength but remain consistent for stocks like **AMGN** and **MSFT**.
- Coefficient heatmaps reveal dynamic relationships, suggesting inverse or direct correlations depending on stock and horizon.

## Files and Scripts
- **`regression_analysis.py`**: Computes contemporaneous and predictive regression analysis.
- **`visualization.py`**: Generates heatmaps and scatter plots for analysis results.
- **`requirements.txt`**: Specifies required Python libraries.

## Remarks
The data time range: Dec.03, 2024 - Dec.16 2024

## License
This project is licensed under the MIT License. See the LICENSE file for details.
