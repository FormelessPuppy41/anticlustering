"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from anticlustering.pipelines.data_simulation import create_pipeline as data_simulation_pl
from anticlustering.pipelines.anticluster import create_pipeline as anticluster_pl


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
