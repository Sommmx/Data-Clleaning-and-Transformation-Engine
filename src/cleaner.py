import numpy as np
from typing import List, Optional, Dict, Union



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



# print(handling_missing_values(data))



'''Handling Outliers'''
def handling_outliers(data: np.ndarray,threshold: float=3):
    numeric_data = detect_numeric_columns(data)

    for col in numeric_data:
        col_data = data[:,col].astype(float)
        mean = np.mean(col_data)
        std = np.std(col_data)
        z = (col_data - mean)/std
        print(z)
        median = np.median(col_data)
        col_data = np.where(np.abs(z) > threshold, median, col_data)
        data[:,col] = col_data.astype(str)
    return data





def normalize_numeric(data: np.ndarray, model: str = 'z-score') -> np.ndarray:
    numeric_data = detect_numeric_columns(data)
    for col in numeric_data:
        col_data = data[:,col].astype(float)
        if model == 'z-score':
            std = np.std(col_data)
            mean = np.mean(col_data)
            col_data = (col_data - mean)/std
        elif model == 'min-max':
            col_data = (col_data - np.min(col_data))/(np.max(col_data) - np.min(col_data))
        data[:,col] = col_data.astype(str)
    return data




'''Encode_categorical'''
def encode_categorical(
        data: np.ndarray,
        save_mapping: bool = False
) -> Union[np.ndarray, Dict[int,Dict[str,int]]]:
    categoric_data = detect_categorical_columns(data)
    mapping = {}
    for col in categoric_data:
        unique_val = np.unique(data[:,col])
        col_mapping = {val:idx for idx,val in enumerate(unique_val)}
        data[:,col] = np.array([col_mapping[val] for val in data[:,col]])
        mapping[col] = col_mapping
    return (data,mapping) if save_mapping else data