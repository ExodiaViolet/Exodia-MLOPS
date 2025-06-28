"""
This is a boilerplate pipeline 'split_data'
generated using Kedro 0.19.14
"""
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


from sklearn.model_selection import StratifiedShuffleSplit

def split_datasets(df: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_col = parameters["target_column"]
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=200)
    
    for train_idx, test_idx in splitter.split(df, df[target_col]):
        ref_data = df.iloc[train_idx]
        ana_data = df.iloc[test_idx]
    
    return ref_data, ana_data