# pipeli​nes/data_simulation/pipeline.py
from kedro.pipeline import Pipeline, node
from ...constants import Parameters as P, Catalog as C
from .nodes import simulate_matrices, generate_simulation_study_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # node(
            #     func=simulate_matrices,
            #     inputs=[
            #         P.DataSimulation.N_VALUES,
            #         P.DataSimulation.NUM_FEATURES,
            #         P.DataSimulation.RNG_SEED,
            #     ],
            #     outputs=C.Data.SIM_DATA,          # e.g. "all_simulated_data"
            #     name="simulate_all_matrices",
            # ),
            node(
                func=generate_simulation_study_data,
                inputs=[
                    P.DataSimulation.SimulationStudy.K_VALUES,
                    P.DataSimulation.SimulationStudy.N_RUNS,
                    P.DataSimulation.SimulationStudy.N_RANGE_MIN,
                    P.DataSimulation.SimulationStudy.N_RANGE_MAX,
                    P.DataSimulation.SimulationStudy.DTR_UNIFORM,
                    P.DataSimulation.SimulationStudy.DTR_NORMAL_STD,
                    P.DataSimulation.SimulationStudy.DTR_NORMAL_WIDE,
                    P.DataSimulation.RNG_SEED
                ],
                outputs=C.Data.SIMULATION_STUDY_DATA,          # e.g. "all_simulated_data"
                name="generate_simulation_study_data",
            ),
        ]
    )
