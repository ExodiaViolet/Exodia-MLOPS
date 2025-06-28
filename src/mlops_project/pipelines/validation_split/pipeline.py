"""
This is a boilerplate pipeline 'validation_split'
generated using Kedro 0.19.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  val_split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= val_split,
                inputs=["train_processed_data","parameters"],
                outputs= ["X_train_data","X_val_data","y_train_data","y_val_data","best_columns"],
                name="split",
            ),
        ]
    )