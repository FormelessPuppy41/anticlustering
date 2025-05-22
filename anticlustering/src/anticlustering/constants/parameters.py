


class Parameters:
    """
    Class to hold the parameters.
    """
    class DataSimulation:
        """
        DataSimulation class to hold the parameters for data simulation.
        """
        N_VALUES = "params:data_simulation.n_values"
        NUM_FEATURES = "params:data_simulation.num_features"
        DISTRIBUTION = "params:data_simulation.distribution"
        SEED = "params:data_simulation.seed"

    class Anticluster:
        """
        Anticluster class to hold the parameters for anticlustering.
        """
        ROOT = "params:Anticluster"