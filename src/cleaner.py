import numpy as np
from typing import List, Optional, Dict, Union




file_path = "data/raw_data.csv"


def load_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=',',dtype=str,skip_header=1)
    return data

data = load_csv(file_path)

print(data)







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
    pass