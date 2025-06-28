"""
This is a boilerplate pipeline 'split_data'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline 


from .nodes import  split_datasets


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= split_datasets,
                inputs= ["data_ingested", "parameters"],
                outputs=["ref_data","ana_data"],
                name="split_out_of_sample",
            ),
        ]
    )

