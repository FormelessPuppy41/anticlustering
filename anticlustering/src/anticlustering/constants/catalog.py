
# src/<project>/catalog.py
class Catalog:
    """String constants for dataset names."""

    SIM_DATA        = "all_simulated_data"
    ALL_MODELS      = "report_all_models"

    class Visualisation:
        """Dataset names for visualisation outputs."""
        TABLE1              = "report_data_table1_replication"
        GRAPH1              = "report_data_graph1_replication"

        PARTITION_TABLE     = "visualisation_tables"  # e.g. "04_reporting/viz"
        PARTITION_FIGURES   = "visualisation_figures"     # e.g. "04_reporting/viz_figures"
        CONVERGENCE_TABLE   = "convergence_table"  # e.g. "04_reporting/convergence_table"
        CONVERGENCE_FIGURE  = "convergence_figure"

