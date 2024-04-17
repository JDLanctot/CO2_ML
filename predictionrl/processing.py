import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import List

__all__ = ['clean_string_column', 'drop_unneeded_columns', 'fill_missing_values',
           'encode_categories', 'scale_features', 'prep_data']

def clean_string_column(s: pd.Series) -> pd.Series:
    # APPLY IF COLUMN IS OF TYPE STRING
    if s.dtype == "object":
        # CLEANING LEADING AND TRAINING SPACES, THEN REMOVE THE EXTRA RETURNS
        return s.str.strip().str.replace(r'\r', '', regex=True)
    # DO NOTHING
    else:
        return s

def drop_unneeded_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.drop(columns=cols)

def fill_missing_values(df: pd.DataFrame, num_strategy: str = 'median', cat_strategy: str = 'most_frequent') -> pd.DataFrame:
    # FOR NUMERIC COLUMNS FILL WITH VALUES BASED ON STRATEGY INPUT
    num_imputer = SimpleImputer(strategy=num_strategy)
    num_columns = df.select_dtypes(include=[np.number]).columns
    if len(num_columns) > 0:
        df[num_columns] = num_imputer.fit_transform(df[num_columns])

    # Categorical columns: Fill with most frequent value
    cat_imputer = SimpleImputer(strategy=cat_strategy)
    cat_columns = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_columns) > 0:
        df[cat_columns] = cat_imputer.fit_transform(df[cat_columns])

    return df

def encode_categories(df: pd.DataFrame, sparse: bool = False) -> pd.DataFrame:
    # CONVERT THE CATEGORICAL VALUES TO TYPES WE CAN USE FOR ML
    # NEED TO SPARSE AS AN OPTION TO AVOID MEMORY OVERFLOW IN SOME CASES
    df = pd.get_dummies(df, drop_first=True, sparse=sparse)
    return df

def scale_features(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    # SCALE THE VALUES SO WE DON'T HAVE POORLY BOUNDED SCALES ON ANY GIVEN VALUE
    # THAT IS EXCEPT THE COLNAME COLUMN BECAUSE WE DON'T WANT TO SCALE OUR VARIABLE
    # WE ARE REALLY INTERESTED IN
    scaler = StandardScaler()
    num_columns = df.select_dtypes(include=[np.number]).columns.drop(colname)
    df[num_columns] = scaler.fit_transform(df[num_columns])
    return df

def prep_data(df: pd.DataFrame, colname: str, cols: List[str] = []) -> pd.DataFrame:
    if cols != []:
        df = drop_unneeded_columns(df, cols)
    df = fill_missing_values(df)
    df = encode_categories(df, sparse=False)
    # df = scale_features(df, colname)
    return df

