from kedro.pipeline import Pipeline, node

from ...constants import Parameters as P, Catalog as C
from .nodes import benchmark_all


def create_pipeline(**kwargs):
    """
    One-node pipeline:
        Input  : simulated data dict
        Outputs: timing table (+ optional all_models pickle)
    """
    return Pipeline(
        [
            node(
                func=benchmark_all,
                inputs=[
                    C.SIM_DATA,                      # dict of DataFrames
                    P.Anticluster.K,              # n_clusters
                    P.Anticluster.SOLVERS,
                    P.Anticluster.STORE_MODELS,
                ],
                outputs=[
                    C.TABLE1,                        # timing table
                    C.ALL_MODELS,          # optional models
                ],
                name="benchmark_all_solvers",
            )
        ]
    )
