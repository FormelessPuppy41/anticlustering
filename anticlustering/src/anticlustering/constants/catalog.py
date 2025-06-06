
# src/<project>/catalog.py
class Catalog:
    """String constants for dataset names."""

    class Data:
        KAGGLE_1920         = "kaggle_data_1920"
        KAGGLE_1418         = "kaggle_data_1418"

        KAGGLE_TEST                         = "kaggle_data_test"  # e.g. kaggle_data_1418_test
        KAGGLE_PROCESSED                    = "kaggle_data_test_processed"  # e.g. kaggle_data_1920_processed
        KAGGLE_PROCESSED_LOAN_RECORDS       = "kaggle_data_test_processed_loan_records"  # e.g. kaggle_data_1920_processed_loan_records
        KAGGLE_PROCESSED_LOAN_RECORDS_LONG  = "kaggle_data_test_processed_loan_records_long"  # e.g. kaggle_data_1920_processed_loan_records_long

        KAGGLE_STREAM_MONTHLY_EVENTS        = "kaggle_stream_monthly_events"  # e.g. kaggle_data_1920_stream_monthly_events

        ANTICLUSTER_ASSIGNMENTS  = "anticluster_assignments"  # e.g. anticluster_assignments_2020-01-01
        ANTICLUSTER_METRICS      = "anticluster_metrics"      # e.g. anticluster_metrics_2020-01-01

        SIM_DATA            = "all_simulated_data"
    
    
    ALL_MODELS              = "report_all_models"

    class Visualisation:
        """Dataset names for visualisation outputs."""
        TABLE1              = "report_data_table1_replication"
        GRAPH1              = "report_data_graph1_replication"

        PARTITION_TABLE     = "visualisation_tables"  # e.g. "04_reporting/viz"
        PARTITION_FIGURES   = "visualisation_figures"     # e.g. "04_reporting/viz_figures"
        CONVERGENCE_TABLE   = "convergence_table"  # e.g. "04_reporting/convergence_table"
        CONVERGENCE_FIGURE  = "convergence_figure"

