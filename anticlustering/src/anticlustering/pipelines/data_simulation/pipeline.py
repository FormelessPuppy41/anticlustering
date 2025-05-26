# pipeli​nes/data_simulation/pipeline.py
from kedro.pipeline import Pipeline, node
from ...constants import Parameters as P, Catalog as C
from .nodes import simulate_matrices


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=simulate_matrices,
                inputs=[
                    P.DataSimulation.N_VALUES,
                    P.DataSimulation.NUM_FEATURES,
                    P.DataSimulation.RNG_SEED,
                ],
                outputs=C.SIM_DATA,          # e.g. "all_simulated_data"
                name="simulate_all_matrices",
            )
        ]
    )
