"""
This is a boilerplate pipeline 'online_anticluster'
generated using Kedro 0.19.13
"""

# --------------------------------------------------------------------------- #
#                          Kedro pipeline factory                             #
# --------------------------------------------------------------------------- #


from kedro.pipeline import node, Pipeline, pipeline  # noqa

from ...constants import Parameters as P, Catalog as C
from .nodes import simulate_stream, update_anticlusters, simulate_solvers, simulate_online_data, simulate_online_solvers, aggregate_results_by_bins, sample_solve_aggregate

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=sample_solve_aggregate,
            inputs=[
                C.Data.KAGGLE_PROCESSED_LOAN_RECORDS,
                P.OnlineAnticluster.K_GROUPS,
                P.OnlineAnticluster.KAGGLE_COLUMNS,
                P.OnlineAnticluster.METRICS_CAT_COLS,
                P.OnlineAnticluster.HARD_BALANCE_COLS,
            ],
            outputs=C.Data.ONLINE_ANTICLUSTER_LOANS_RESULTS,
            name="sample_solve_aggregate_node"
        ),
        # node(
        #     func=simulate_stream,
        #     inputs=[
        #         C.Data.KAGGLE_PROCESSED_LOAN_RECORDS,
        #         P.OnlineAnticluster.STREAM_START_DATE,
        #         P.OnlineAnticluster.STREAM_END_DATE
        #     ],
        #     outputs=C.Data.KAGGLE_STREAM_MONTHLY_EVENTS,
        #     name="simulate_stream_node"
        # ),
        # node(
        #     func=update_anticlusters,
        #     inputs=[
        #         C.Data.KAGGLE_PROCESSED_LOAN_RECORDS,
        #         C.Data.KAGGLE_STREAM_MONTHLY_EVENTS,
        #         P.OnlineAnticluster.K_GROUPS,
        #         P.OnlineAnticluster.KAGGLE_COLUMNS,
        #         P.OnlineAnticluster.METRICS_CAT_COLS,
        #         P.OnlineAnticluster.HARD_BALANCE_COLS
        #     ],
        #     outputs=[
        #         C.Data.ANTICLUSTER_ASSIGNMENTS,
        #         C.Data.ANTICLUSTER_METRICS
        #     ],
        #     name="update_anticlusters_node"
        # ), 
        # node(
        #     func=simulate_solvers,
        #     inputs=[
                
        #     ],
        #     outputs=C.Online.ONLINE_SOLVER_METRICS,
        #     name="simulate_solvers_node"
        # ),
        #1) generate simulators
        node(
            func=simulate_online_data,
            inputs=None,
            outputs=C.Online.SIMULATORS,
            name="simulate_online_data_node",
        ),

        #2) run all solvers over those simulators
        node(
            func=simulate_online_solvers,
            inputs=dict(
                sims           = C.Online.SIMULATORS,
            ),
            outputs=C.Online.SOLVER_RAW_RESULTS,
            name="simulate_online_solvers_node",
        ),

        # 3) aggregate into the final “Table 2”‐style output
        node(
            func=aggregate_results_by_bins,
            inputs=C.Online.SOLVER_RAW_RESULTS,
            outputs=C.Online.ONLINE_SOLVER_METRICS,
            name="aggregate_results_by_bins_node",
        ),
        ]
    )

