import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_contemporaneous_cross_impact(daily_data):
    """
    Analyze contemporaneous cross-impact of OFI on short-term price changes across stocks.

    Args:
        daily_data (pd.DataFrame): DataFrame containing the daily OFI metrics.

    Returns:
        list[dict]: Results containing R^2, coefficients, and intercepts for each stock.
    """
    results = []

    # Prepare integrated OFI data for all stocks
    integrated_ofi = daily_data.pivot(index='timestamp', columns='symbol', values='integrated_ofi')
    mid_prices = daily_data.pivot(index='timestamp', columns='symbol', values='mid_price')
    mid_price_changes = np.log(mid_prices).diff()[1:]

    for stock in mid_price_changes.columns:
        Y = mid_price_changes[stock].dropna()
        X = integrated_ofi.loc[Y.index].ffill().bfill()  # Match X and Y by index

        if X.empty or Y.empty:
            logging.warning(f"Skipping stock {stock}: insufficient data for regression.")
            continue

        model = LinearRegression()
        model.fit(X, Y)

        r2 = r2_score(Y, model.predict(X))
        coefficients = model.coef_
        intercept = model.intercept_

        result = {'stock': stock, 'r2': r2, 'intercept': intercept}
        for idx, coeff in enumerate(coefficients):
            result[f'coef_{integrated_ofi.columns[idx]}'] = coeff

        results.append(result)

    return results

def analyze_predictive_power(daily_data, horizons=[1, 5]):
    """
    Evaluate the predictive power of lagged cross-asset OFI on future price changes.

    Args:
        daily_data (pd.DataFrame): DataFrame containing the daily OFI metrics.
        horizons (list): List of time horizons (in minutes) for predictive analysis.

    Returns:
        list[dict]: Results containing R^2, coefficients, and intercepts for each stock and horizon.
    """
    results = []

    integrated_ofi = daily_data.pivot(index='timestamp', columns='symbol', values='integrated_ofi')
    mid_prices = daily_data.pivot(index='timestamp', columns='symbol', values='mid_price')

    for horizon in horizons:
        future_price_changes = np.log(mid_prices.shift(-horizon)) - np.log(mid_prices)

        for stock in future_price_changes.columns:
            Y = future_price_changes[stock].dropna()
            X = integrated_ofi.loc[Y.index].ffill().bfill()

            if X.empty or Y.empty:
                logging.warning(f"Skipping stock {stock} for horizon {horizon}: insufficient data for regression.")
                continue

            model = LinearRegression()
            model.fit(X, Y)

            r2 = r2_score(Y, model.predict(X))
            coefficients = model.coef_
            intercept = model.intercept_

            result = {'stock': stock, 'horizon': horizon, 'r2': r2, 'intercept': intercept}
            for idx, coeff in enumerate(coefficients):
                result[f'coef_{integrated_ofi.columns[idx]}'] = coeff

            results.append(result)

    return results

def process_files(input_dir, output_dir, analysis_function, file_suffix, **kwargs):
    """
    Generalized function to process files, apply an analysis function, and save results.

    Args:
        input_dir (str): Directory containing input files.
        output_dir (str): Directory to save output files.
        analysis_function (callable): Function to apply for analysis.
        file_suffix (str): Suffix for output files.
        kwargs (dict): Additional arguments for the analysis function.

    Returns:
        None
    """
    file_paths = glob.glob(os.path.join(input_dir, "*_ofi_metrics.csv"))
    os.makedirs(output_dir, exist_ok=True)

    for file_path in file_paths:
        date = os.path.basename(file_path).split('_')[0]  # Extract date from filename
        daily_data = pd.read_csv(file_path)
        results = analysis_function(daily_data, **kwargs)
        results_df = pd.DataFrame(results)

        output_path = os.path.join(output_dir, f"{date}_{file_suffix}.csv")
        results_df.to_csv(output_path, index=False)
        logging.info(f"Saved results for {date} to {output_path}")

def evaluate_explanatory_power(cross_impact_results):
    """
    Evaluate explanatory power of contemporaneous OFI models.

    Args:
        cross_impact_results (pd.DataFrame): DataFrame containing R^2 and coefficients from contemporaneous models.

    Returns:
        pd.DataFrame: Aggregated metrics for explanatory power.
    """
    grouped = cross_impact_results.groupby('stock').mean()

    # Prepare a readable matrix output
    summary_matrix = grouped.filter(like='coef_').copy()
    summary_matrix['avg_r2'] = grouped['r2']

    return summary_matrix.reset_index()


def evaluate_predictive_power(predictive_power_results):
    """
    Evaluate predictive power of lagged OFI models.

    Args:
        predictive_power_results (pd.DataFrame): DataFrame containing R^2 and coefficients from predictive models.
        output_excel_path (str): Path to save the Excel file with results for each horizon.

    Returns:
        None
    """
    summary_matrix_dictionary = {}
    for horizon in predictive_power_results['horizon'].unique():
        subset = predictive_power_results[predictive_power_results['horizon'] == horizon]
        explanatory_summary = evaluate_explanatory_power(subset)
        summary_matrix_dictionary[horizon] = explanatory_summary

    return summary_matrix_dictionary


def main():
    input_dir = "../data/ofi_metrics"
    contemporaneous_output_dir = "../data/contemporaneous_regression_results"
    predictive_output_dir = "../data/predictive_regression_results"
    explanatory_summary_path = os.path.join(contemporaneous_output_dir, "explanatory_power_summary.csv")
    predictive_summary_excel = os.path.join(predictive_output_dir, "predictive_power_summary.xlsx")

    # Step 1: Analyze contemporaneous cross-impact
    process_files(input_dir, contemporaneous_output_dir, analyze_contemporaneous_cross_impact, "result")

    # Step 2: Analyze predictive power
    process_files(input_dir, predictive_output_dir, analyze_predictive_power, "result", horizons=[1, 5])

    # Step 3: Summarize explanatory power
    file_paths = glob.glob(os.path.join(contemporaneous_output_dir, "*_result.csv"))
    cross_impact_results = []

    for file_path in file_paths:
        daily_result = pd.read_csv(file_path)
        cross_impact_results.append(daily_result)

    cross_impact_combined = pd.concat(cross_impact_results, ignore_index=True)
    explanatory_summary = evaluate_explanatory_power(cross_impact_combined)
    explanatory_summary.to_csv(explanatory_summary_path, index=False)
    logging.info(f"Explanatory power summary saved to {explanatory_summary_path}")

    # Step 4: Summarize predictive power
    file_paths = glob.glob(os.path.join(predictive_output_dir, "*_result.csv"))
    predictive_regression_results = []

    for file_path in file_paths:
        daily_result = pd.read_csv(file_path)
        predictive_regression_results.append(daily_result)

    predictive_power_combined = pd.concat(predictive_regression_results, ignore_index=True)
    predictive_summary_dict = evaluate_predictive_power(predictive_power_combined)

    with pd.ExcelWriter(predictive_summary_excel, engine="openpyxl") as writer:
        for horizon, summary in predictive_summary_dict.items():
            summary.to_excel(writer, sheet_name=f"Horizon_{horizon}", index=False)
    logging.info(f"Predictive power summary saved to {predictive_summary_excel}")

if __name__ == "__main__":
    main()
