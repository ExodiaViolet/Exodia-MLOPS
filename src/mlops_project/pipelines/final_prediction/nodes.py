"""
This is a boilerplate pipeline 'final_prediction'
generated using Kedro 0.19.14
"""
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np

logger = logging.getLogger(__name__)

def final_prediction(
    X: pd.DataFrame,
    model: Any,  # Use Any because model type varies depending on the library
    columns: list[str]  # This should be a list of column names to use for prediction
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Make predictions using a trained classification model and compute evaluation metrics.

    Args:
        X (pd.DataFrame): Input DataFrame containing features and the 'readmitted' label column.
        model (Any): Trained classification model with `.predict()` and optionally `.predict_proba()`.
        columns (list[str]): List of column names to use for prediction.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - Updated DataFrame with predictions.
            - Dictionary containing classification metrics.
    """
    
    # Separate labels
    y_test = X['readmitted']

    # Make a copy to avoid modifying original data
    X_copy = X.copy()

    # Predict
    y_pred = model.predict(X_copy[columns])
    X_copy['y_pred'] = y_pred

    # If your model supports predict_proba (for AUC), use it
    try:
        y_proba = model.predict_proba(X_copy[columns])[:, 1]
    except AttributeError:
        y_proba = None

    # Calculate evaluation metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)

    # Logging
    logger.info('Predictions made.')
    logger.info('#prediction: %s', len(y_pred))
    logger.info('Metrics: %s', metrics)

    return X_copy, metrics