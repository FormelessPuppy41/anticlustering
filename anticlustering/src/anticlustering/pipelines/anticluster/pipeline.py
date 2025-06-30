from kedro.pipeline import Pipeline, node

from ...constants import Parameters as P, Catalog as C
from .nodes import benchmark_all, benchmark_simulation


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
                    C.Data.SIM_DATA,                      # dict of DataFrames
                    P.Anticluster.K,              # n_clusters
                    P.Anticluster.SOLVERS,
                    P.Anticluster.STORE_MODELS,
                ],
                outputs=[
                    C.Visualisation.TABLE1,                        # timing table
                    C.Visualisation.GRAPH1,                       # convergence graph
                    C.ALL_MODELS,          # optional models
                ],
                name="benchmark_all_solvers",
            ),
            node(
                func=benchmark_simulation,
                inputs=[
                    C.Data.SIMULATION_STUDY_DATA,  # DataFrame with simulation study data
                    P.Anticluster.SIM_STUDY.SOLVERS,
                ],
                outputs=C.Visualisation.TABLE2,  # timing table for simulation study
                name="benchmark_simulation_study",
            ),
        ]
    )

