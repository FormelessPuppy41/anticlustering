# src/<project>/parameters.py
class Parameters:
    """String constants for YAML parameter paths."""

    class DataSimulation:
        N_VALUES      = "params:simulation.n_values"
        NUM_FEATURES  = "params:simulation.n_features"
        RNG_SEED      = "params:simulation.rng_seed"

    class Anticluster:
        STORE_MODELS   = "params:anticluster.store_models"
        SOLVERS        = "params:anticluster.solvers"   # list of dicts
        K              = "params:anticluster.k"

        class SolverLimits:
            ILP_MAX_N        = "params:anticluster.solver_limits.ilp_max_n"
            PRECLUSTER_MAX_N = "params:anticluster.solver_limits.precluster_max_n"
