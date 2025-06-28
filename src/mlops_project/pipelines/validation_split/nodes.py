"""
This is a boilerplate pipeline 'validation_split'
generated using Kedro 0.19.14
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split



def val_split(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and validation sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    assert [col for col in data.columns if data[col].isnull().any()] == []
    y = data[parameters["target_column"]]
    X = data.drop(columns=parameters["target_column"], axis=1)
    X = X.drop(columns=["encounter_id", "patient_nbr"], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=parameters["test_fraction"], random_state=parameters["random_state"])

    return X_train, X_val, y_train, y_val, list(X_train.columns)