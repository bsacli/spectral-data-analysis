from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    drop_unwanted_columns,
    train_test_split_multitarget,
    encode,
    scale_features,
    scale_targets,
    apply_pca_on_spectral,
    select_features_by_any_target
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   
            node(
                func=drop_unwanted_columns,
                inputs=["filtered_data", "params:drop_cols"],
                outputs="cleaned_data",
                name="clean_unwanted_columns"
            ),
            # node(
            #     func=select_features_by_any_target,
            #     inputs=["cleaned_data", "params:target_cols", "params:correlation_threshold"],
            #     outputs="selected_data",
            #     name="select_features_by_any_target",
            # ),
            node(
                func=train_test_split_multitarget,
                inputs=["cleaned_data", "params:target_cols"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_train_test_multitarget"
            ),
            # node(
            #     func=encode,
            #     inputs=["X_train", "X_test", "params:categorical_cols"],
            #     outputs=["X_train_encoded", "X_test_encoded"],
            #     name="encode_cultivar"
            # ),
            node(
                func=scale_features,
                inputs=["X_train", "X_test"],
                outputs=["X_train_scaled", "X_test_scaled", "feature_scaler"],
                name="scale_features"
            ),
            node(
                func=scale_targets,
                inputs=["y_train", "y_test"],
                outputs=["y_train_scaled", "y_test_scaled", "target_scaler"],
                name="scale_targets"
            ),
            node(
                func=apply_pca_on_spectral,
                inputs=["X_train_scaled", "X_test_scaled", "params:pca_n_components"],
                outputs=["X_train_pca", "X_test_pca", "pca_model"],
                name="apply_pca"
            ),
        ]
    ) # type: ignore
