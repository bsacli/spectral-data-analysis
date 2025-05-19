from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    process_repeated_measurements,
    clean_raw_data,
    data_consistency_check,
    isolation_forest_outlier_detection,
    filter_outliers,
    denoise_spectrum_data,
    convert_to_magnitude_phase
   
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_raw_data,
                inputs="raw_data",
                outputs="clean_data",
                name="clean_raw_data",
            ),
            node(
                func=data_consistency_check,
                inputs="clean_data",
                outputs="checked_data",
                name="data_consistency_check",
            ),
            node(
                func=denoise_spectrum_data,
                inputs="checked_data",
                outputs="denoised_data",
                name="noise_reduction",
            ),
            node(
                func=process_repeated_measurements,
                inputs="denoised_data",
                outputs="averaged_data",
                name="average_firmness_with_outlier_flag",
            ),
            node(
                func=isolation_forest_outlier_detection,
                inputs="averaged_data",
                outputs="hybrid_outlier_flagged_data",
                name="apply_isolation_forest_outlier_detection",
            ),
            node(
                func=filter_outliers,
                inputs="hybrid_outlier_flagged_data",
                outputs="filtered_data",
                name="filter_outliers_from_averaged_data",
            ),
            node(
                func=convert_to_magnitude_phase,
                inputs="filtered_data",
                outputs="magphase_data",
                name="convert_real_imag_to_magnitude_phase",
            ),
        ]
    ) # type: ignore
