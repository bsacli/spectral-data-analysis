# """Project pipelines."""

# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines

from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from spectral_data_analysis.pipelines import data_processing as dp
from spectral_data_analysis.pipelines import data_science as ds
from spectral_data_analysis.pipelines import machine_learning as ml


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    machine_learning_pipeline = ml.create_pipeline()

    return {"data_processing":data_processing_pipeline, 
            "data_science":data_science_pipeline, 
            "machine_learning":machine_learning_pipeline,
            "__default__": data_processing_pipeline,
            }