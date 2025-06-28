"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines

from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from mlops_project.pipelines import (
    data_unit_tests as data_tests,
    data_ingestion as ingestion,
    split_data as split_data,
    data_preprocessing as preproc,
    validation_split as val_split,
    model_training as model_training,
    model_selection as model_selection,
    final_prediction as final_prediction
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_ingestion_pipeline = ingestion.create_pipeline()
    data_unit_tests_pipeline = data_tests.create_pipeline()
    data_split_pipeline = split_data.create_pipeline()
    data_preprocessing_pipeline = preproc.create_pipeline()
    val_split_pipeline = val_split.create_pipeline()
    model_training_pipeline = model_training.create_pipeline()
    model_selection_pipeline = model_selection.create_pipeline()
    final_prediction_pipeline = final_prediction.create_pipeline()
    full_pipeline = (
        data_ingestion_pipeline
        + data_unit_tests_pipeline
        + data_split_pipeline
        + data_preprocessing_pipeline
        + val_split_pipeline
        + model_training_pipeline
        + model_selection_pipeline
        + final_prediction_pipeline
    )
    return {

        "data_ingestion": data_ingestion_pipeline,
        "data_unit_tests": data_unit_tests_pipeline,
        "split_data": data_split_pipeline,
        "data_preprocessing": data_preprocessing_pipeline,
        "validation_split": val_split_pipeline,
        "model_training": model_training_pipeline,
        "model_selection": model_selection_pipeline,
        "final_prediction": final_prediction_pipeline,
        "__default__": full_pipeline

    }