import os
import numpy as np
import pandas as pd
from typing import Any, Dict
from schema.data_schema import BinaryClassificationSchema, load_json_data_schema
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from joblib import dump
from config import paths


def impute_numeric(input_data: pd.DataFrame, column: Any, value='median') -> pd.DataFrame:
    if value == 'mean':
        input_data[column].fillna(value=input_data[column].mean(), inplace=True)
    elif value == 'median':
        input_data[column].fillna(value=input_data[column].median(), inplace=True)
    elif value == 'mode':
        input_data[column].fillna(value=input_data[column].mode().iloc[0], inplace=True)
    else:
        input_data[column].fillna(value=value, inplace=True)
    return input_data


def indicate_missing_values(input_data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = input_data.select_dtypes(exclude=["number"]).columns.tolist()
    return input_data.replace("", np.nan)


def impute_categorical(input_data: pd.DataFrame, column: Any) -> pd.DataFrame:
    if percentage_of_missing_values(input_data)[column] > 10:
        input_data[column].fillna(value='Missing', inplace=True)
    else:
        input_data[column].fillna(value=input_data[column].mode().iloc[0], inplace=True)
    return input_data


def drop_all_nan_features(input_data: pd.DataFrame) -> pd.DataFrame:
    return input_data.dropna(axis=1, how='all')


def percentage_of_missing_values(input_data: pd.DataFrame) -> Dict:
    columns_with_missing_values = input_data.columns[input_data.isna().any()]
    return (input_data[columns_with_missing_values].isna().mean().sort_values(ascending=False) * 100).to_dict()


def drop_constant_features(input_data: pd.DataFrame) -> pd.DataFrame:
    constant_columns = input_data.columns[input_data.nunique() == 1]
    return input_data.drop(columns=constant_columns)


def drop_duplicate_features(input_data: pd.DataFrame) -> pd.DataFrame:
    return input_data.T.drop_duplicates().T


def encode(input_data: pd.DataFrame, schema: BinaryClassificationSchema) -> pd.DataFrame:
    encodings = []
    cat_features = schema.categorical_features
    if not cat_features:
        return input_data
    for f in cat_features:
        number_of_allowed_values = len(schema.allowed_categorical_values[f])
        drop_first = number_of_allowed_values == 2
        encoding = pd.get_dummies(input_data[f], prefix=f, drop_first=drop_first)
        input_data.drop(f, axis='columns', inplace=True)
        encodings.append(encoding)
    return pd.concat(input_data + encodings, axis='columns')


def drop_mostly_missing_columns(input_data: pd.DataFrame, thresh=0.6) -> pd.DataFrame:
    threshold = int(thresh * len(input_data))
    return input_data.dropna(axis=1, thresh=threshold)


def normalize(input_data: pd.DataFrame, schema: BinaryClassificationSchema, scaler=None) -> pd.DataFrame:
    numeric_features = schema.numeric_features
    if scaler is None:
        scaler = StandardScaler()
        input_data[numeric_features] = scaler.fit_transform(input_data[numeric_features])
        dump(scaler, os.path.join(paths.TRAIN_DIR, 'scaler.joblib'))
    else:
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    return input_data


def remove_outliers_zscore(input_data: pd.DataFrame, column: str) -> pd.DataFrame:
    threshold = 3
    z_scores = np.abs(zscore(input_data[column]))
    condition = z_scores < threshold
    return input_data[condition]


def remove_outliers_iqr(input_data: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = input_data[column].quantile(0.25)
    q3 = input_data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    condition = (input_data[column] > upper) | (input_data[column] < lower)
    return input_data[~condition]


