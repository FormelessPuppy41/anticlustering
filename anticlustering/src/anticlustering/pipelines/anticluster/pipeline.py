from kedro.pipeline import Pipeline, node, pipeline
from anticlustering.constants.catalog import Catalog as C
from anticlustering.constants.parameters import Parameters as P

from .nodes import instantiate_solvers, solve_all


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            instantiate_solvers,
            inputs=P.Anticluster.ROOT,          # whole dict
            outputs=C.Anticluster.SOLVER_LIST,
            name="instantiate_anticluster_solvers_node",
        ),
        node(
            solve_all,
            inputs=dict(
                solvers=C.Anticluster.SOLVER_LIST,
                datasets=C.DataSimulation.INTERMEDIATE_GENERATED_DATASETS,   # same as in your sim pipeline
            ),
            outputs=[C.Anticluster.RESULTS, C.Anticluster.TABLE_PICKLE, C.Anticluster.TABLE_CSV],                # e.g. "anticluster_results"
            name="solve_anticluster_all_node",
        ),
    ])
