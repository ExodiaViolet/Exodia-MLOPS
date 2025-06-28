"""
This is a boilerplate pipeline 'final_prediction'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import final_prediction

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=final_prediction,
                inputs=["test_processed_data", "champion_model","production_columns"],
                outputs=["df_with_predict", "predict_describe"],
                name="predict",
            ),
        ]
    )