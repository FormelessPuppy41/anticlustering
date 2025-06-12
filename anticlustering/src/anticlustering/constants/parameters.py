# src/<project>/parameters.py
class Parameters:
    """String constants for YAML parameter paths."""
    RNG_NUMBER = "params:rng_number"  # e.g. 42, 1234, etc.

    class DataSimulation:
        N_VALUES      = "params:simulation.n_values"
        NUM_FEATURES  = "params:simulation.n_features"
        RNG_SEED      = "params:simulation.rng_seed"

        class SimulationStudy:
            K_VALUES        = "params:simulation_study.k_values"  # e.g. [2, 3, 4, 5]
            N_RUNS          = "params:simulation_study.n_runs"    # e.g. 10

            N_RANGE_MIN     = "params:simulation_study.n_range.min"  # e.g. 10
            N_RANGE_MAX     = "params:simulation_study.n_range.max"  # e.g. 1000

            DTR_UNIFORM     = "params:simulation_study.distributions.uniform"  # bool, e.g. True or False
            DTR_NORMAL_STD  = "params:simulation_study.distributions.normal_std"   # bool, e.g. True or False
            DTR_NORMAL_WIDE = "params:simulation_study.distributions.normal_wide"  # float, e.g. 0.0

    class KaggleData:
        KAGGLE_1418   = "params:kaggle_data.kaggle_1418"
        KAGGLE_1920   = "params:kaggle_data.kaggle_1920"

    class Anticluster:
        STORE_MODELS   = "params:anticluster.store_models"
        SOLVERS        = "params:anticluster.solvers"           # list of dicts
        K              = "params:anticluster.k"

        class SIM_STUDY:
            SOLVERS        = "params:anticluster.solvers_simulation_study"  # list of dicts

    class OnlineAnticluster:
        KAGGLE_COLUMNS      = "params:kaggle_columns"  # e.g. kaggle_columns_1920.yaml

        AS_OF_STR           = "params:online.as_of_str"
        REGULAR_REPAYMENT   = "params:online.regular_repayment"
        REDUCE_N            = "params:online.reduce_n"          # int, e.g. 1000
        SCALE               = "params:online.scale"             # bool, e.g. True or False
        STREAM_START_DATE   = "params:online.stream_start_date"
        STREAM_END_DATE     = "params:online.stream_end_date"   # e.g. "2020-01-01"
        K_GROUPS            = "params:online.k_groups"          # int, e.g. 10
        HARD_BALANCE_COLS   = "params:online.hard_balance_cols"  # list of str, e.g. ["loan_status"]
        SIZE_TOLERANCE      = "params:online.size_tolerance"     # int, e.g. 1  
        REBALANCE_FREQUENCY = "params:online.rebalance_frequency"  # int, e.g. 3 (months)
        METRICS_CAT_COLS    = "params:online.metrics_cat_cols"   # list of str, e.g. ["grade", "purpose"]

    class Visualisation:
        MAIN_SOLVER    = "params:visualisation.main_solver"     # list or str.
        ARTEFACT_DIR   = "params:visualisation.artefact_dir"
        NUMBER_OF_N    = "params:visualisation.number_of_Ns"    # e.g. 10
        MATCH_MODE     = "params:visualisation.match_mode"      # "exact", "contains", "regex"
