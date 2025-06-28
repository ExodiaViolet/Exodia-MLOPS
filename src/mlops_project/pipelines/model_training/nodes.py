"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.14
"""
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from sklearn.metrics import f1_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def model_train(X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_test: pd.DataFrame,
                parameters: Dict[str, Any], 
                best_columns,
                experiment_name: str = 'Default_Experiment'):
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.
        parameters (Dict[str, Any]): Model parameters including baseline_model_params.
        best_columns: List of selected feature columns.
        experiment_name (str): MLflow experiment name. Defaults to 'Default_Experiment'.

    Returns:
    --
        model: Trained model.
        columns: Feature names used in training.
        results_dict: Dictionary with model metrics.
        plt: Matplotlib plot object with SHAP summary plot.
    """
    best_columns_list = list(best_columns)
    
    mlflow_track = parameters.get('mlflow_track', True)  # Default to True if not specified

    if mlflow_track:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        mlflow.xgboost.autolog(log_model_signatures=True, log_input_examples=True)

    # Open pickle file with regressors or initialize new XGBClassifier
    if parameters.get("load_model", True):
        with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
            classifier = pickle.load(f)
    else:
        classifier = xgb.XGBClassifier(**parameters['baseline_model_params'])

    results_dict = {}

    if mlflow_track:
        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            if parameters.get("use_feature_selection", False):
                logger.info(f"Using feature selection in model train...")
                X_train = X_train[best_columns_list]
                X_test = X_test[best_columns_list]

            y_train = np.ravel(y_train)
            model = classifier.fit(X_train, y_train)

            # Making predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluating model
            acc_train = f1_score(y_train, y_train_pred)
            acc_test = f1_score(y_test, y_test_pred)

            # Saving results in dict
            results_dict['classifier'] = classifier.__class__.__name__
            results_dict['train_score'] = acc_train
            results_dict['test_score'] = acc_test

            # Logging info
            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged train model in run {run_id}")
            logger.info(f"Train f1 is {acc_train}")
            logger.info(f"Test f1 is {acc_test}")

    else:
        # No mlflow tracking
        if parameters.get("use_feature_selection", False):
            logger.info(f"Using feature selection in model train...")
            X_train = X_train[best_columns_list]
            X_test = X_test[best_columns_list]

        y_train = np.ravel(y_train)
        model = classifier.fit(X_train, y_train)

        # Making predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluating model
        acc_train = f1_score(y_train, y_train_pred)
        acc_test = f1_score(y_test, y_test_pred)

        # Saving results in dict
        results_dict['classifier'] = classifier.__class__.__name__
        results_dict['train_score'] = acc_train
        results_dict['test_score'] = acc_test

        logger.info(f"Train f1 is {acc_train}")
        logger.info(f"Test f1 is {acc_test}")

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=best_columns_list)

    shap.summary_plot(shap_values.values, X_train, feature_names=X_train.columns, show=False)

    return model, list(X_train.columns), results_dict, plt
