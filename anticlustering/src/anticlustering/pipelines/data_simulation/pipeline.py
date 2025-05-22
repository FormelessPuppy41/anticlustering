# src/<your_project>/pipelines/data_simulation/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_all_datasets, combine_datasets

from anticlustering.constants.catalog import Catalog as C
from anticlustering.constants.parameters import Parameters as P

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_all_datasets,
            inputs={
                "n_values": P.DataSimulation.N_VALUES,
                "num_features": P.DataSimulation.NUM_FEATURES,
                "distribution": P.DataSimulation.DISTRIBUTION,
                "seed": P.DataSimulation.SEED,
            },
            outputs=C.DataSimulation.INTERMEDIATE_GENERATED_DATASETS,
            name="generate_all_datasets_node",
        ),
        node(
            func=combine_datasets,
            inputs=C.DataSimulation.INTERMEDIATE_GENERATED_DATASETS,
            outputs=C.DataSimulation.SIMULATED_DATASETS,
            name="combine_datasets_node",
        ),
    ])
