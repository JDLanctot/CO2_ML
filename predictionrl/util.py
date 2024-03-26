import os
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.io import savemat
import torch

__all__ = ['init_weights', 'set_seed', 'set_mpl']

def init_weights(module):
    """ Set all weights to a small, uniform range. Set all biases to zero. """
    def _init_weights(m):
        try:
            # torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        except AttributeError:
            pass
        try:
            torch.nn.init.zeros_(m.bias) #torch.nn.init.uniform_(m.bias, -0.01, 0.01)
        except AttributeError:
            pass

    module.apply(_init_weights)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)

def set_mpl():
    # change defaults to be less ugly for matplotlib
    mpl.rc('xtick', labelsize=14, color="#222222")
    mpl.rc('ytick', labelsize=14, color="#222222")
    mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    mpl.rc('font', size=16)
    mpl.rc('xtick.major', size=6, width=1)
    mpl.rc('xtick.minor', size=3, width=1)
    mpl.rc('ytick.major', size=6, width=1)
    mpl.rc('ytick.minor', size=3, width=1)
    mpl.rc('axes', linewidth=1, edgecolor="#222222", labelcolor="#222222")
    mpl.rc('text', usetex=False, color="#222222")

def parse_data(folder_path):
    # Initialize empty lists to hold the data for each column and additional columns for tracking
    columns = ['MarketId', 'MarketName', 'ContractId', 'ContractName', 'HistoryDateET',
               'OpenSharePrice', 'CloseSharePrice', 'LowSharePrice', 'HighSharePrice',
               'AverageTradePrice', 'TradeVolume', 'IdSeries', 'TimeSeries']
    data = {column: [] for column in columns}

    # Set to keep track of (ContractId, HistoryDateET) tuples to avoid duplicates
    existing_entries = set()

    # Dictionary to keep track of the IdSeries value for each ContractId
    contract_id_series = {}

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Remove trailing spaces in any string columns and commas
            df['HistoryDateET'] = df['HistoryDateET'].str.strip()
            df['MarketName'] = df['MarketName'].str.strip()
            df['ContractName'] = df['ContractName'].str.strip()
            df['TradeVolume'] = df['TradeVolume'].astype(str).str.strip()
            df['TradeVolume'] = df['TradeVolume'].str.replace(',', '', regex=False)

            # Change column types
            df['HistoryDateET'] = pd.to_datetime(df['HistoryDateET'])
            df['TradeVolume'] = pd.to_numeric(df['TradeVolume'])

            # Ensure 'TradeVolume' is present; if not, add it with default 0 values
            if 'TradeVolume' not in df.columns:
                df['TradeVolume'] = 0

            # Rename a column that has a wrong name in one sheets
            if 'AverageTradePrice' not in df.columns:
                df.rename(columns={'AverageSharePrice': 'AverageTradePrice'}, inplace=True)

            # Process each unique ContractId in the dataframe
            for contract_id in df['ContractId'].unique():
                contract_df = df[df['ContractId'] == contract_id].copy()
                contract_df.sort_values(by='HistoryDateET', inplace=True)  # Ensure chronological order

                # Assign IdSeries value
                id_series_value = contract_id_series.get(contract_id, len(contract_id_series) + 1)
                contract_id_series[contract_id] = id_series_value

                # Generate TimeSeries values
                time_series_values = list(range(1, len(contract_df) + 1))

                # Check for duplicates and append data if unique
                for _, row in contract_df.iterrows():
                    entry_key = (row['ContractId'], row['HistoryDateET'])
                    if entry_key not in existing_entries:
                        existing_entries.add(entry_key)
                        for column in columns[:-2]:  # Exclude IdSeries and TimeSeries for now
                            data[column].append(row[column])
                        data['IdSeries'].append(id_series_value)  # Same IdSeries value for the contract
                        data['TimeSeries'].append(time_series_values.pop(0))  # Sequential TimeSeries value

    # Save the data arrays into a .mat file
    savemat('data_python.mat', data)


