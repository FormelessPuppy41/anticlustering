# src/anticlustering/visualization/pipeline.py
from kedro.pipeline import Pipeline, node
from ...constants import Catalog as C, Parameters as P

from .nodes import visualise_first_partitions, centroid_convergence

def create_pipeline(**kwargs):
    return Pipeline(
        [   
            node(
                func=centroid_convergence,
                inputs=[
                    C.ALL_MODELS,
                    C.Data.SIM_DATA,
                    P.Visualisation.MAIN_SOLVER,
                ],
                outputs=[
                    C.Visualisation.CONVERGENCE_TABLE,
                    C.Visualisation.CONVERGENCE_FIGURE,
                ],
                name="centroid_convergence",
            ),
            node(
                func=visualise_first_partitions,
                inputs=[
                    C.Data.SIM_DATA,          # same dict used earlier
                    C.ALL_MODELS,        # needs store_models=True upstream
                    P.Visualisation.MAIN_SOLVER,   # e.g. "Exchange"
                    P.Visualisation.NUMBER_OF_N,    # e.g. 10
                    P.Visualisation.MATCH_MODE,      # e.g. "exact"
                ],
                outputs=[
                    C.Visualisation.PARTITION_TABLE,    
                    C.Visualisation.PARTITION_FIGURES,
                ],
                name="visualise_first_partitions",
            )
        ]
    )
