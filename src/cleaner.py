import numpy as np
from typing import List, Optional, Dict, Union




file_path = "data/raw_data.csv"


def load_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=',',dtype=str,skip_header=1)
    return data

data = load_csv(file_path)

print(data)
print("##########################################################3")







'''Numeric and Categorical Columns'''
def detect_numeric_columns(data : np.ndarray) -> List[int]:
    numeric_columns = []
    for col in range(data.shape[1]):
        column = data[:,col]
        non_empty = column[column!='']
        try:
            _ = non_empty.astype(float)
            numeric_columns.append(col)
        except ValueError:
            continue
    return numeric_columns


def detect_categorical_columns(data : np.ndarray) ->List[int]:
    return [col for col in range(data.shape[1]) if col not in detect_numeric_columns(data)]







'''Handling missing values'''
def handling_missing_values(data: np.ndarray,numeric_strategy: str = 'mean',categoric_strategy: str = 'mode') -> np.ndarray:
    numerics_col = detect_numeric_columns(data)
    categoric_col = detect_categorical_columns(data)


    for col in numerics_col:
        col_data = np.array([float(x) if x not in ['','nan'] else np.nan for x in data[:,col]])
        if numeric_strategy == 'mean':
            replacement = np.nanmean(col_data)
        else:
            replacement = np.nanmedian(col_data)
        col_data = np.where(np.isnan(col_data),replacement,col_data)
        data[:,col] = col_data.astype(str)
    

    for col in categoric_col:
        col_data = data[:,col]
        non_empty = col_data[col_data != '']
        if(len(non_empty) > 0):
            values, count = np.unique(non_empty,return_counts=True)
        if categoric_strategy == 'mode':
            replacement = values[np.argmax(count)]
        else:
            replacement = 'unknown'
        data[:,col] = np.where(col_data == '', replacement, col_data)
    return data



print(handling_missing_values(data))




