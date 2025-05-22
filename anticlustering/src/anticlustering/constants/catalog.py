


class Catalog:
    """
    Catalog class to manage the catalog of datasets.
    """
    class DataSimulation:
        """
        DataSimulation class to manage the catalog of datasets for data simulation.
        """
        SIMULATED_DATASETS              = "primary_data_simulated_datasets"
        INTERMEDIATE_GENERATED_DATASETS = "intermediate_generated_datasets"
    
    class Anticluster:
        """
        Anticluster class to manage the catalog of datasets for anticlustering.
        """
        SOLVER_LIST = 'anticluster_solvers'
        RESULTS = "primary_anticluster_results"
        TABLE_PICKLE = "primary_anticluster_table_pickle"
        TABLE_CSV = "primary_anticluster_table_csv"