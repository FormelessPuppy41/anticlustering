"""
This is a boilerplate pipeline 'kaggle_data'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import load_kaggle_data, create_test_kaggle_data, process_kaggle_data

from ...constants import Parameters as P, Catalog as C


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # node(
        #     func=load_kaggle_data,
        #     inputs=P.KaggleData.KAGGLE_1418,
        #     outputs=C.Data.KAGGLE_1418,
        #     name="load_kaggle_data_node_1418",
        # ),
        # node(
        #     func=load_kaggle_data,
        #     inputs=P.KaggleData.KAGGLE_1920,
        #     outputs=C.Data.KAGGLE_1920,
        #     name="load_kaggle_data_node_1920",
        # ),
        # node(
        #     func=create_test_kaggle_data,
        #     inputs=C.Data.KAGGLE_1418,
        #     outputs=C.Data.KAGGLE_TEST,
        #     name="create_test_kaggle_data_node_1418",
        # ),
        node(
            func=process_kaggle_data,
            inputs=C.Data.KAGGLE_TEST,
            outputs=C.Data.KAGGLE_PROCESSED,
            name="load_kaggle_data_node_1920",
        ),
    ])
