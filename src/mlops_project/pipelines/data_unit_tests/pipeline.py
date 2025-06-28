"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa


from .nodes import test_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= test_data,
                inputs="diabetic_raw_data",
                outputs= "reporting_tests",
                name="data_unit_tests",
            ),

        ]
    )
