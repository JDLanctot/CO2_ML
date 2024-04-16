import os
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.io import loadmat, savemat
import torch
from typing import List, Dict

__all__ = ['init_weights', 'set_seed', 'set_mpl', 'parse_data', 'extract_series',
           'get_ids', 'get_data_dict', 'data_to_dictionary', 'dictionary_metrics']

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


def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)

def set_mpl() -> None:
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

def parse_data(folder_path: str) -> None:
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

#Function for getting only data associated with a particular contract id
def extract_series(k: int, price_series: np.ndarray, contract_series: np.ndarray) -> np.ndarray:
    return price_series[contract_series == k]

#Function for getting how large of a contract set each contract id is in
# def n_contracts(k: int, market_series: np.ndarray, contract_series: np.ndarray, series_len: int) -> int:
#     market_id = market_series[contract_series == k][0]
#     return int(len(contract_series[market_series == market_id])/series_len)

#Get only a subset based on market contract number
# def filter_sets(data: Dict, Ns: np.ndarray, unique_ids: np.ndarray, condition: int) -> Tuple[Dict, np.ndarray]:
#     if condition <= 2:
#         filter_idx = [unique_ids[k] for k in range(len(unique_ids)) if Ns[k] == condition]
#         filter_Ns = [Ns[k] for k in range(len(unique_ids)) if Ns[k] == condition]
#     else:
#         filter_idx = [unique_ids[k] for k in range(len(unique_ids)) if Ns[k] >= condition]
#         filter_Ns = [Ns[k] for k in range(len(unique_ids)) if Ns[k] >= condition]
#     data = {k: data[k] for k in filter_idx}
#     return data, np.array(filter_Ns)

#Get unique contract ids from Matlab data file
def get_ids(contract_series: np.ndarray) -> List:
    return list(map(int, np.unique(contract_series)))

#Save the data corresponding to each contract to a keyed data structure
def get_data_dict(unique_ids: List, price_series: np.ndarray, contract_series: np.ndarray) -> Dict:
    return {k: extract_series(k, price_series, contract_series) for k in unique_ids}

#Save the number of contracts a contract is part of corresponding to each contract to a keyed data structure
def get_number_contracts(unique_ids: np.ndarray, market_series: np.ndarray, contract_series: np.ndarray, series_len: int) -> np.ndarray:
    return  np.array([n_contracts(k, market_series, contract_series, series_len) for k in unique_ids])

def data_to_dictionary(filename: str) -> Dict:
    data = loadmat(filename)

    # columns = ['MarketId', 'MarketName', 'ContractId', 'ContractName', 'HistoryDateET',
    #            'OpenSharePrice', 'CloseSharePrice', 'LowSharePrice', 'HighSharePrice',
    #            'AverageTradePrice', 'TradeVolume', 'IdSeries', 'TimeSeries']

    market_series = data["MarketId"]
    contract_series = data["ContractId"]
    price_series = data["CloseSharePrice"]

    return get_data_dict(get_ids(contract_series), price_series, contract_series)

def data_to_dataframe(filename: str) -> pd.DataFrame:
    data = loadmat(filename)
    return pd.DataFrame({
        'MarketId': data['MarketId'][0],
        # 'MarketName': data['MarketName'],
        'ContractId': data['ContractId'][0],
        # 'ContractName': data['ContractName'],
        # 'HistoryDateET': data['HistoryDateET'][0],
        # 'OpenSharePrice': data["OpenSharePrice"][0],
        # 'CloseSharePrice': data["CloseSharePrice"][0],
        # 'LowSharePrice': data["LowSharePrice"][0],
        # 'HighSharePrice': data["HighSharePrice"][0],
        'AverageTradePrice': data["AverageTradePrice"][0],
        'TradeVolume': data["TradeVolume"][0],
        # 'IdSeries': data["IdSeries"][0],
        'TimeSeries': data["TimeSeries"][0],
    })

def dictionary_metrics(data: dict, df: pd.DataFrame) -> pd.DataFrame:
    # contract_ids = [k for k in data.keys()]
    means = [np.mean(s) for s in data.values()]
    stds = [np.std(s) for s in data.values()]
    maxs = [np.max(s) for s in data.values()]
    mins = [np.min(s) for s in data.values()]
    results = [np.round(s[-1]) for s in data.values()]

    # Step 1: Calculate unique ContractIds per MarketId
    unique_contracts = df.groupby('MarketId')['ContractId'].nunique().reset_index(name='N')

    # Step 2: Merge this with the original DataFrame
    contract_counts = df.drop_duplicates(['MarketId', 'ContractId']).merge(unique_contracts, on='MarketId', how='left')

    # Step 3: Create DataFrame from data.keys() to preserve order
    order_df = pd.DataFrame(list(data.keys()), columns=['ContractId'])

    # Step 4: Merge to align results with the order_df; use outer join to ensure all keys are used
    metrics_df = order_df.merge(contract_counts, on='ContractId', how='left')

    # Step 5: Add the other metrics
    metrics_df['Mean'] = means
    metrics_df['StDev'] = stds
    metrics_df['Max'] = maxs
    metrics_df['Min'] = mins
    metrics_df['Result'] = results

    return metrics_df

