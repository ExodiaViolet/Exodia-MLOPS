"""
This is a boilerplate pipeline 'preprocess_train_pipeline'
generated using Kedro 0.19.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  clean_and_process_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= clean_and_process_data,
                inputs= ["ref_data", "ana_data"],
                outputs=["train_processed_data","test_processed_data"],
                name="data_preprocessing",
            ),
        ]
    )