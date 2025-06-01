# src/<project>/parameters.py
class Parameters:
    """String constants for YAML parameter paths."""

    class DataSimulation:
        N_VALUES      = "params:simulation.n_values"
        NUM_FEATURES  = "params:simulation.n_features"
        RNG_SEED      = "params:simulation.rng_seed"

    class Anticluster:
        STORE_MODELS   = "params:anticluster.store_models"
        SOLVERS        = "params:anticluster.solvers"           # list of dicts
        K              = "params:anticluster.k"

    class Visualisation:
        MAIN_SOLVER    = "params:visualisation.main_solver"     # list or str.
        ARTEFACT_DIR   = "params:visualisation.artefact_dir"
        NUMBER_OF_N    = "params:visualisation.number_of_Ns"    # e.g. 10
        MATCH_MODE     = "params:visualisation.match_mode"      # "exact", "contains", "regex"
