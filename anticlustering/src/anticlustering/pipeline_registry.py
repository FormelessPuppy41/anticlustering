"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# BaseLine pipelines
from anticlustering.pipelines.data_simulation import create_pipeline as data_simulation_pl
from anticlustering.pipelines.anticluster import create_pipeline as anticluster_pl
from anticlustering.pipelines.visualisation import create_pipeline as visualisation_pl

# Online pipelines
from anticlustering.pipelines.kaggle_data import create_pipeline as kaggle_data_pl
from anticlustering.pipelines.online_anticluster import create_pipeline as online_anticluster_pl


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())

    # ------- Register the individual pipelines ------------------------

    # BaseLine pipeline
    # This pipeline runs the data simulation, the anticlustering algorithm, and visualisation.
    # The outputs are Table1, Table2, and a Graph.
    baseline = data_simulation_pl() + anticluster_pl() + visualisation_pl()
    pipelines["baseline"] = baseline
    
    # Online pipeline
    # This pipeline runs the data preprocessing, the online anticlustering algorithm, and visualisation.
    online = kaggle_data_pl() + online_anticluster_pl() 
    pipelines["online"] = online


    return pipelines





"""
Issues are:

within_group_variance has issues -> somehow groupsizes appear to be 0, which leads to skips and null values in the output.



"""