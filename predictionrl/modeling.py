import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Tuple, List, Union

__all__ = ['clean_string_column', 'drop_unneeded_columns', 'fill_missing_values',
           'encode_categories', 'scale_features', 'prep_data',
           'create_train_test_sets', 'train_linear_model', 'train_lasso_model',
           'train_linear_model', 'train_tree_model', 'train_forrest_model',
           'evaluate_model', 'feature_importance_from_model', 'model_train_features']

# DATA PROCESSING FUNCTIONS -------------------------------------------------------
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

# -----------------------------------------------------------------------

# TRAINING FUNCTIONS -------------------------------------------------------
def create_train_test_sets(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 432) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_set, test_set

def train_linear_model(train_set: pd.DataFrame, colname: str) -> LinearRegression:
    train_X = train_set.drop(colname, axis=1)
    train_y = train_set[colname]
    lin_reg = LinearRegression()
    lin_reg.fit(train_X, train_y)
    return lin_reg

def train_lasso_model(train_set: pd.DataFrame, colname: str) -> LassoCV:
    train_X = train_set.drop(colname, axis=1)
    train_y = train_set[colname]
    lasso = LassoCV(cv=5, random_state=42).fit(train_X, train_y)
    return lasso

def train_tree_model(train_set: pd.DataFrame, colname: str) -> DecisionTreeRegressor:
    train_X = train_set.drop(colname, axis=1)
    train_y = train_set[colname]
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(train_X, train_y)
    return tree

def train_forrest_model(train_set: pd.DataFrame, colname: str) -> RandomForestRegressor:
    train_X = train_set.drop(colname, axis=1)
    train_y = train_set[colname]
    forrest = RandomForestRegressor(random_state=42)
    forrest.fit(train_X, train_y)
    return forrest

def feature_importance_from_model(model: Union[LinearRegression, LassoCV, RandomForestRegressor, DecisionTreeRegressor], feature_names: List[str]) -> pd.DataFrame:
    # GETTING FEATURE IMPORTANCE FROM MODEL COEFFICIENTS
    # DETERMINE THE CORRECT OBJECT VARIABLE FOR IMPORTANCE BASED ON THE MODEL
    if hasattr(model, 'coef_'):
        # LINEAR AND LASSO
        importance_values = np.abs(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        # TREE AND FORREST
        importance_values = model.feature_importances_
    else:
        raise ValueError("Model does not have recognized importance attribute.")

    importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance_values})
    importance = importance.sort_values(by='Importance', ascending=False)
    return importance


def evaluate_model(model: Union[LinearRegression, LassoCV, DecisionTreeRegressor, RandomForestRegressor],
                   train_set: pd.DataFrame,
                   test_set: pd.DataFrame,
                   colname: str,
                   perf_thresh: float = 0.1) -> None:

    # EXAMINE THE TRAINING SET
    train_X = train_set.drop(colname, axis=1)
    train_y = train_set[colname]
    train_predictions = model.predict(train_X)
    train_rmse = np.sqrt(mean_squared_error(train_y, train_predictions))

    # EXAMINE THE TEST SET
    test_X = test_set.drop(colname, axis=1)
    test_y = test_set[colname]
    test_predictions = model.predict(test_X)
    test_rmse = np.sqrt(mean_squared_error(test_y, test_predictions))

    print('-'*90)
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    # HOW WELL DID IT DO IN TESTING RELATIVE TO TRAINING
    if test_rmse < 0.3:
        print("Prediction performance of the model is Great.")
    elif test_rmse < 0.5:
        print("Prediction performance of the model is OK.")
    elif test_rmse < 0.6:
        print("Prediction performance of the model is Subpar.")
    else:
        print("Prediction Performance of the model is Awful.")

    if train_rmse < test_rmse:
        if (test_rmse - train_rmse) / train_rmse > perf_thresh:
            print("Model may be overfitting.")
        else:
            print("Model has balanced performance on training and test sets.")
    elif train_rmse > test_rmse:
        if (train_rmse - test_rmse) / train_rmse > perf_thresh:
            print("Model may be underfitting.")
        else:
            print("Model has balanced performance on training and test sets.")
    else:
        print("Model has balanced performance on training and test sets.")


def model_train_features(df: pd.DataFrame, colname: str, type: str = 'linear') -> Tuple[Union[LinearRegression, LassoCV], pd.DataFrame, pd.DataFrame]:
    train_set, test_set = create_train_test_sets(df)
    print('-'*90)
    if type == 'lasso':
        print(f'We are training a LassoCV model')
        model = train_lasso_model(train_set, colname)
    elif type == 'tree':
        print(f'We are training a DecisionTreeRegressor model')
        model = train_tree_model(train_set, colname)
    elif type == 'forrest':
        print(f'We are training a RandomForestRegressor model')
        model = train_forrest_model(train_set, colname)
    else:
        print(f'We are training a LinearRegression model')
        model = train_linear_model(train_set, colname)
    return model, train_set, test_set

