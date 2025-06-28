"""
This is a boilerplate pipeline 'model_selection'
generated using Kedro 0.19.14
"""
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import yaml
import pickle
import warnings
warnings.filterwarnings("ignore", category=Warning)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

def _get_or_create_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id
     
def model_selection(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_test: pd.DataFrame,
                    champion_dict: Dict[str, Any],
                    champion_model : pickle.Pickler,
                    parameters: Dict[str, Any]):

    """Trains a model on the given data and optionally logs results with MLflow.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.
        champion_dict (dict): Dictionary storing current champion model info.
        champion_model (pickle.Pickler): Current champion model object.
        parameters (dict): Parameters defined in parameters.yml including 'mlflow_track'.

    Returns:
    --
        best_model: The selected best model after training and tuning.
    """
    print(parameters)
    scale_pos_weight = parameters.get('scale_pos_weight')
    models_dict = {
        'LogisticRegression': LogisticRegression(class_weight='balanced'),
        'XGBClassifier': XGBClassifier(eval_metric='logloss',scale_pos_weight=scale_pos_weight)
    }

    initial_results = {}
    mlflow_track = parameters.get('mlflow_track', False)

    if mlflow_track:
        import mlflow
        import mlflow.sklearn
        import yaml

        with open('conf/local/mlflow.yml') as f:
            experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
            experiment_id = _get_or_create_experiment_id(experiment_name)
            logger.info(experiment_id)
    else:
        logger.info("MLflow tracking is disabled for this run.")

    logger.info('Starting first step of model selection : Comparing between model types')

    for model_name, model in models_dict.items():
        if mlflow_track:
            with mlflow.start_run(experiment_id=experiment_id, nested=True):
                mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

                y_train_ravel = np.ravel(y_train)

                if model_name == 'LogisticRegression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train_ravel)
                    y_pred = model.predict(X_test_scaled)
                    score = f1_score(y_test, y_pred)
                else:
                    model.fit(X_train, y_train_ravel)
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test, y_pred)

                initial_results[model_name] = score
                run_id = mlflow.last_active_run().info.run_id
                logger.info(f"Logged model : {model_name} in run {run_id}")
        else:
            y_train_ravel = np.ravel(y_train)

            if model_name == 'LogisticRegression':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train_ravel)
                y_pred = model.predict(X_test_scaled)
                score = f1_score(y_test, y_pred)
            else:
                model.fit(X_train, y_train_ravel)
                y_pred = model.predict(X_test)
                score = f1_score(y_test, y_pred)

            initial_results[model_name] = score
            logger.info(f"Trained model : {model_name} with score: {score}")

    best_model_name = max(initial_results, key=initial_results.get)
    best_model = models_dict[best_model_name]

    logger.info(f"Best model is {best_model_name} with score {initial_results[best_model_name]}")
    logger.info('Starting second step of model selection : Hyperparameter tuning')

    param_grid = parameters['hyperparameters'][best_model_name]

    if mlflow_track:
        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            if best_model_name == 'LogisticRegression':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                gridsearch = GridSearchCV(best_model, param_grid, cv=2, scoring='f1', n_jobs=-1)
                gridsearch.fit(X_train_scaled, y_train_ravel)
                best_model = gridsearch.best_estimator_
                X_test_final = X_test_scaled
            else:
                gridsearch = GridSearchCV(best_model, param_grid, cv=2, scoring='f1', n_jobs=-1)
                gridsearch.fit(X_train, y_train_ravel)
                best_model = gridsearch.best_estimator_
                X_test_final = X_test

            logger.info(f"Hypertuned model score: {gridsearch.best_score_}")
    else:
        if best_model_name == 'LogisticRegression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            gridsearch = GridSearchCV(best_model, param_grid, cv=2, scoring='f1', n_jobs=-1)
            gridsearch.fit(X_train_scaled, y_train_ravel)
            best_model = gridsearch.best_estimator_
            X_test_final = X_test_scaled
        else:
            gridsearch = GridSearchCV(best_model, param_grid, cv=2, scoring='f1', n_jobs=-1)
            gridsearch.fit(X_train, y_train_ravel)
            best_model = gridsearch.best_estimator_
            X_test_final = X_test

        logger.info(f"Hypertuned model score: {gridsearch.best_score_}")

    pred_score = f1_score(y_test, best_model.predict(X_test_final))

    if champion_dict['test_score'] < pred_score:
        logger.info(f"New champion model is {best_model_name} with score: {pred_score} vs {champion_dict['test_score']} ")
        return best_model
    else:
        logger.info(f"Champion model is still {champion_dict['classifier']} with score: {champion_dict['test_score']} vs {pred_score} ")
        return champion_model
