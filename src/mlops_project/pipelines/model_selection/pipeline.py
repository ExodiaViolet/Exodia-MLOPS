"""
This is a boilerplate pipeline 'model_selection'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa


from .nodes import model_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=["X_train_data","X_val_data","y_train_data","y_val_data",
                        "production_model_metrics",
                        "production_model",
                        "parameters"],
                outputs="champion_model",
                name="model_selection",
            ),
        ]
    )
