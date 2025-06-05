# src/<project>/parameters.py
class Parameters:
    """String constants for YAML parameter paths."""
    RNG_NUMBER = "params:rng_number"  # e.g. 42, 1234, etc.

    class DataSimulation:
        N_VALUES      = "params:simulation.n_values"
        NUM_FEATURES  = "params:simulation.n_features"
        RNG_SEED      = "params:simulation.rng_seed"

    class KaggleData:
        KAGGLE_1418   = "params:kaggle_data.kaggle_1418"
        KAGGLE_1920   = "params:kaggle_data.kaggle_1920"

    class Anticluster:
        STORE_MODELS   = "params:anticluster.store_models"
        SOLVERS        = "params:anticluster.solvers"           # list of dicts
        K              = "params:anticluster.k"

    class OnlineAnticluster:
        AS_OF_STR           = "params:online.as_of_str"
        REGULAR_REPAYMENT   = "params:online.regular_repayment"
        REDUCE_N            = "params:online.reduce_n"          # int, e.g. 1000
        SCALE               = "params:online.scale"             # bool, e.g. True or False

    class Visualisation:
        MAIN_SOLVER    = "params:visualisation.main_solver"     # list or str.
        ARTEFACT_DIR   = "params:visualisation.artefact_dir"
        NUMBER_OF_N    = "params:visualisation.number_of_Ns"    # e.g. 10
        MATCH_MODE     = "params:visualisation.match_mode"      # "exact", "contains", "regex"
