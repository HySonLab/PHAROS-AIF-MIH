"""Load CT COVID index CSV and return rows for train/val splits."""
import os
import pandas as pd
from typing import List, Dict, Any


def load_task1_index(root_path: str, index_csv: str) -> List[Dict[str, Any]]:
    """Load task1_npy_index.csv. Returns all rows. npy_path is relative to root_path."""
    csv_path = os.path.join(root_path, index_csv)
    df = pd.read_csv(csv_path)
    return df.to_dict('records')


def load_task2_index(root_path: str, index_csv: str) -> List[Dict[str, Any]]:
    """Load task2_npy_index.csv. Returns all rows. npy_path is relative to root_path."""
    csv_path = os.path.join(root_path, index_csv)
    df = pd.read_csv(csv_path)
    return df.to_dict('records')


def filter_by_split(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    """Filter rows by split (train or val)."""
    return [r for r in rows if r.get('split') == split]
