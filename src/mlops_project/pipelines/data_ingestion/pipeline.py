"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa



from .nodes import ingestion

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= ingestion,
                inputs=["diabetic_raw_data","parameters"],
                outputs= "data_ingested",
                name="data_ingestion",
            ),

        ]
    )

