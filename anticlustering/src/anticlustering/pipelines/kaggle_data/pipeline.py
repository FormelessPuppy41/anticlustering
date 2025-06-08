"""
This is a boilerplate pipeline 'kaggle_data'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import (
    load_kaggle_data, 
    create_test_kaggle_data, 
    parse_kaggle_data,
    kaggle_df_to_loan_records,
    loan_records_to_long_df
)

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
        # ), #FIXME: Remove this and use the reduce_n parameter in the process_kaggle_data node instead
        node(
            func=parse_kaggle_data,
            inputs=[
                C.Data.KAGGLE_TEST,
                P.OnlineAnticluster.KAGGLE_COLUMNS,
                P.OnlineAnticluster.REDUCE_N,
                P.RNG_NUMBER
            ],
            outputs=C.Data.KAGGLE_PROCESSED,
            name="load_kaggle_data_node_1920",
        ),
        node(
            func=kaggle_df_to_loan_records,
            inputs=C.Data.KAGGLE_PROCESSED,
            outputs=C.Data.KAGGLE_PROCESSED_LOAN_RECORDS,
            name="kaggle_df_to_loan_records_node",
        ),
        node(
            func=loan_records_to_long_df,
            inputs=[
                C.Data.KAGGLE_PROCESSED_LOAN_RECORDS,
                P.OnlineAnticluster.AS_OF_STR,
                P.OnlineAnticluster.REGULAR_REPAYMENT
            ],
            outputs=C.Data.KAGGLE_PROCESSED_LOAN_RECORDS_LONG,
            name="loan_records_to_long_df_node",
        ),
    ])
