"""
This is a boilerplate pipeline 'drift_simulation'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa


from .nodes import simulate_drift

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= simulate_drift,
                inputs="test_processed_data",
                outputs= "test_drifted_data",
                name="data_unit_tests",
            ),

        ]
    )