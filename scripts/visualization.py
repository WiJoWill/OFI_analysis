import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import openpyxl

pd.set_option('display.max_columns', None)

def visualize_heatmap(data, title="heatmap", output_dir="../results/"):
    """
    Generate a heatmap image based on the given dataframe.

    Args:
        data (pd.DataFrame): Summary DataFrame.
        title (str): Title for the heatmap and filename.
        output_dir (str): Directory to save the heatmap.

    Returns:
        None
    """
    try:
        output_path = os.path.join(output_dir, f"{title}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        heatmap_data = data.set_index('stock').filter(like='coef_').fillna(0)
        annotation_data = heatmap_data.map(lambda x: f"{x:.2e}")

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=annotation_data,
            fmt="",
            cmap='coolwarm',
            linewidths=0.5,
            cbar_kws={"format": "%.1e"}
        )
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Heatmap saved to {output_path}.")
    except Exception as e:
        print(f"Error generating heatmap for {title}: {e}")

def visualize_scatter_plot(data, x_variable, y_variable, title="scatter_plot", output_dir="../results/"):
    """
    Generate scatter plots for data with X and Y variables, and legends for different stocks.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot.
        x_variable (str): Column name for X-axis variable.
        y_variable (str): Column name for Y-axis variable.
        title (str): Title for the scatter plot and filename.
        output_dir (str): Directory to save the scatter plots.

    Returns:
        None
    """
    try:
        output_path = os.path.join(output_dir, f"{title}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=(12, 8))
        for stock, group in data.groupby('stock'):
            plt.plot(group[x_variable], group[y_variable], marker='o', label=stock)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title(title)
        plt.legend(title='Stock')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Scatter plot saved to {output_path}.")
    except Exception as e:
        print(f"Error generating scatter plot for {title}: {e}")

def load_and_combine_csv(file_pattern):
    """
    Load and combine multiple CSV files matching a pattern into a single DataFrame.

    Args:
        file_pattern (str): Glob pattern to match file paths.

    Returns:
        pd.DataFrame: Combined DataFrame from all matching files.
    """
    try:
        file_paths = sorted(glob.glob(file_pattern))
        data_frames = []
        for file_path in file_paths:
            date = os.path.basename(file_path).split('_')[0]  # Extract date from filename
            df = pd.read_csv(file_path)
            df['date'] = date  # Add trading day column
            data_frames.append(df)
        combined_data = pd.concat(data_frames, ignore_index=True)
        combined_data = combined_data.loc[:, ~combined_data.columns.str.contains('^Unnamed')]
        return combined_data
    except Exception as e:
        print(f"Error loading CSV files with pattern {file_pattern}: {e}")
        return pd.DataFrame()

def main():
    try:
        time_horizons = [1, 5]

        # Visualize heatmaps for explanatory power
        explanatory_summary_path = "../data/contemporaneous_regression_results/explanatory_power_summary.csv"
        if os.path.exists(explanatory_summary_path):
            summary_df = pd.read_csv(explanatory_summary_path)
            visualize_heatmap(summary_df, title="explanatory_cross_impact")
        else:
            print(f"Explanatory summary file not found: {explanatory_summary_path}")

        # Visualize heatmaps for predictive power
        predictive_summary_path = "../data/predictive_regression_results/predictive_power_summary.xlsx"
        if os.path.exists(predictive_summary_path):
            with pd.ExcelFile(predictive_summary_path) as xls:
                for i, horizon in enumerate(time_horizons):
                    summary_df = pd.read_excel(xls, sheet_name=i)
                    visualize_heatmap(summary_df, title=f"predictive_power_horizon_{horizon}")
        else:
            print(f"Predictive summary file not found: {predictive_summary_path}")

        # Visualize scatter plots for contemporaneous impact
        contemporaneous_combined = load_and_combine_csv("../data/contemporaneous_regression_results/*_result.csv")
        if not contemporaneous_combined.empty:
            visualize_scatter_plot(
                contemporaneous_combined, 
                x_variable="date", 
                y_variable="r2", 
                title="contemporaneous_impact_by_trading_day"
            )
        else:
            print("No data found for contemporaneous impact scatter plot.")

        # Visualize scatter plots for predictive power
        predictive_combined = load_and_combine_csv("../data/predictive_regression_results/*_result.csv")
        if not predictive_combined.empty:
            for horizon in time_horizons:
                subset = predictive_combined[predictive_combined['horizon'] == horizon]
                visualize_scatter_plot(
                    subset, 
                    x_variable="date", 
                    y_variable="r2", 
                    title=f"predictive_power_horizon_{horizon}"
                )
        else:
            print("No data found for predictive power scatter plots.")

        print("Visualization process completed successfully.")
    except Exception as e:
        print(f"An error occurred in the main visualization process: {e}")

if __name__ == "__main__":
    main()
