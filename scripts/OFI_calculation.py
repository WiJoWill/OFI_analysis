import pandas as pd
import numpy as np
import glob
from sklearn.decomposition import PCA
import argparse
import os

def process_minute_level(data):
    """
    Process data at the minute level by keeping the last state within each minute.

    Args:
        data (pd.DataFrame): DataFrame containing order book data.

    Returns:
        pd.DataFrame: Minute-level processed data.
    """
    data.loc[:, 'timestamp'] = pd.to_datetime(data.loc[:,'ts_recv']).dt.floor('T')
    # Use groupby and last directly without sorting
    minute_level_data = data.groupby(['symbol', 'timestamp']).last().reset_index()
    minute_level_data.fillna(0, inplace=True)
    return minute_level_data


def compute_ofi(data, levels=5):
    """
    Compute multi-level OFI metrics.
    
    Args:
        data (pd.DataFrame): Input data containing LOB states.
        levels (int): Number of levels to compute OFI for.

    Returns:
        pd.DataFrame: DataFrame containing OFI metrics for each stock.
    """
    ofi_results = []

    for symbol, group in data.groupby('symbol'):
        # Calculate mid-price
        group['mid_price'] = (group['bid_px_00'] + group['ask_px_00']) / 2

        # Filter out rows where mid_price is zero
        group = group[group['mid_price'] > 0].reset_index(drop=True)

        # Initialize dictionary for storing OFI and mid-price
        symbol_ofi = {
            'symbol': group['symbol'],
            'timestamp': group['timestamp'],
            'mid_price': group['mid_price']
        }

        for level in range(0, levels):
            bid_price_col = f'bid_px_0{level}'
            bid_size_col = f'bid_sz_0{level}'
            ask_price_col = f'ask_px_0{level}'
            ask_size_col = f'ask_sz_0{level}'

            # Compute bid and ask OFIs
            bid_ofi = np.where(
                group[bid_price_col].diff() > 0, group[bid_size_col],
                np.where(
                    group[bid_price_col].diff() == 0,
                    group[bid_size_col].diff(),
                    -group[bid_size_col]
                )
            )

            ask_ofi = np.where(
                group[ask_price_col].diff() > 0, -group[ask_size_col],
                np.where(
                    group[ask_price_col].diff() == 0,
                    group[ask_size_col].diff(),
                    group[ask_size_col]
                )
            )

            level_ofi = bid_ofi - ask_ofi
            symbol_ofi[f'ofi_level_{level}'] = level_ofi

        # Convert to DataFrame and append to results
        ofi_results.append(pd.DataFrame(symbol_ofi))

    return pd.concat(ofi_results, ignore_index=True)

def integrate_ofi_with_pca(daily_data, date, levels=5):
    """
    Integrate multi-level OFIs into a single metric using PCA for each stock in a given trading day.

    Args:
        daily_data (pd.DataFrame): DataFrame containing the daily OFI metrics.
        date (str): Trading day date as a string.
        levels (int): Number of levels used in OFI computation.

    Returns:
        pd.DataFrame: Updated DataFrame with integrated OFI and explained variance.
        list: A list of dictionaries containing explained variance data for each stock on the given day.
    """
    explained_variances = []

    for symbol, group in daily_data.groupby('symbol'):
        pca_columns = [f'ofi_level_{i}' for i in range(levels)]
        original_indices = group.index  # Track original indices for correct assignment later

        # Check if all columns are null or constant
        if group[pca_columns].isnull().all().any() or group[pca_columns].nunique().max() == 1:
            print(f"Skipping PCA for symbol {symbol} on {date}: insufficient data.")
            daily_data.loc[original_indices, 'integrated_ofi'] = np.nan
            explained_variances.append({
                'date': date,
                'symbol': symbol,
                'explained_variance': np.nan
            })
            continue

        # Perform PCA on multi-level OFI columns
        pca = PCA(n_components=1)
        group['integrated_ofi'] = pca.fit_transform(group[pca_columns])

        # Update the original daily_data DataFrame with the calculated 'integrated_ofi'
        daily_data.loc[original_indices, 'integrated_ofi'] = group['integrated_ofi']

        # Collect explained variance
        explained_variances.append({
            'date': date,
            'symbol': symbol,
            'explained_variance': pca.explained_variance_ratio_[0]
        })

    return daily_data, explained_variances



def main():
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", 
        type=str, 
        required=True, 
        help="Path to the raw data file (e.g., ../data/raw_data/dbeq-basic-YYYYMMDD.csv.zst)"
    )
    parser.add_argument(
        "--pca", 
        action="store_true", 
        help="Include this flag to apply PCA for integrating multi-level OFIs."
    )
    parser.add_argument(
        "--levels", 
        type=int, 
        default=5, 
        help="Number of levels to compute OFI for (default is 5)."
    )
    args = parser.parse_args()

    # Load the raw data
    print(f"Reading file: {args.file_path}")
    raw_data = pd.read_csv(args.file_path)

    # Process the data to minute level
    print("Processing data to minute level...")
    minute_level_data = process_minute_level(raw_data)

    # Compute OFI
    print(f"Computing OFI for {args.levels} levels...")
    ofi_data = compute_ofi(minute_level_data, levels=args.levels)
    date = os.path.basename(args.file_path).split('.')[0].split('-')[-1]

    # Save processed OFI data
    output_path = f"../data/ofi_metrics/{date}_ofi_metrics.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ofi_data.to_csv(output_path, index=False)
    print(f"Processed OFI metrics saved to {output_path}.")

    if args.pca:
        # Extract date from file name for PCA output
        file_paths = glob.glob("../data/ofi_metrics/*_ofi_metrics.csv")
        all_explained_variances = []

        for file_path in file_paths:
            date = file_path.split('/')[-1].split('_')[0]  # Extract date from filename
            daily_data = pd.read_csv(file_path)
            daily_data, daily_explained_variances = integrate_ofi_with_pca(daily_data, date, levels=5)

            # Save updated daily data with integrated OFI
            daily_data.to_csv(file_path, index=False)

            # Collect explained variance data
            all_explained_variances.extend(daily_explained_variances)
            
        explained_variance_df = pd.DataFrame(all_explained_variances)
        explained_variance_matrix = explained_variance_df.pivot(index='date', columns='symbol', values='explained_variance')
        explained_variance_matrix.to_csv('../data/ofi_metrics/explained_variance_matrix.csv')
        print(f"PCA process has been finished. Original ofi metrics data are modified and an explained variance matrix is saved.")
        

if __name__ == "__main__":
    main()