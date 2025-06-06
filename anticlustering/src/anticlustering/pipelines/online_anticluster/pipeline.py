"""
This is a boilerplate pipeline 'online_anticluster'
generated using Kedro 0.19.13
"""

# --------------------------------------------------------------------------- #
#                          Kedro pipeline factory                             #
# --------------------------------------------------------------------------- #


from kedro.pipeline import node, Pipeline, pipeline  # noqa

from ...constants import Parameters as P, Catalog as C
from .nodes import simulate_stream, update_anticlusters

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=simulate_stream,
            inputs=[
                C.Data.KAGGLE_PROCESSED_LOAN_RECORDS,
                P.OnlineAnticluster.STREAM_START_DATE,
                P.OnlineAnticluster.STREAM_END_DATE
            ],
            outputs=C.Data.KAGGLE_STREAM_MONTHLY_EVENTS,
            name="simulate_stream_node"
        ),
        node(
            func=update_anticlusters,
            inputs=[
                C.Data.KAGGLE_PROCESSED_LOAN_RECORDS,
                C.Data.KAGGLE_STREAM_MONTHLY_EVENTS,
                P.OnlineAnticluster.K_GROUPS,
                P.OnlineAnticluster.HARD_BALANCE_COLS,
                P.OnlineAnticluster.SIZE_TOLERANCE,
                P.OnlineAnticluster.REBALANCE_FREQUENCY,
                P.OnlineAnticluster.METRICS_CAT_COLS
            ],
            outputs=[
                C.Data.ANTICLUSTER_ASSIGNMENTS,
                C.Data.ANTICLUSTER_METRICS
            ],
            name="update_anticlusters_node"
        )
    ])

