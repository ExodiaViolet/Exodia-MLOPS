"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_train,
                inputs=["X_train_data", "X_val_data", "y_train_data", "y_val_data",
                        "parameters","best_columns"],
                outputs=["production_model", "production_columns", "production_model_metrics", "output_plot"],
                name="model_train",
            ),
        ]
    )