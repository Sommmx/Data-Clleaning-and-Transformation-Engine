import numpy as np
import os

def save_csv(data: np.ndarray, header: np.ndarray, output_path: str):
    """Save NumPy array to CSV with header."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_data = np.vstack([header, data])
    np.savetxt(output_path, full_data, delimiter=',', fmt='%s')

def load_csv(file_path: str) -> np.ndarray:
    """Load CSV skipping header."""
    return np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)

def load_header(file_path: str) -> np.ndarray:
    return np.genfromtxt(file_path, delimiter=',', dtype=str, max_rows=1)
