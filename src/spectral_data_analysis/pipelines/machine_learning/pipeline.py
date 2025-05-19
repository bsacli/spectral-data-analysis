from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    tune_xgboost_for_each_target,   
    train_individual_models, 
    predict_with_individual_models,
    inverse_transform_predictions,
    evaluate_multitarget_regression,
    clean_y_test_for_evaluation
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=tune_xgboost_for_each_target,
            inputs=["X_train_pca", "y_train_scaled", "params:n_trials"],
            outputs="xgb_best_params_per_target",
            name="optuna_hyperparameter_tuning_per_target"
        ),
        node(
            func=train_individual_models,
            inputs=["X_train_pca", "y_train_scaled", "xgb_best_params_per_target"],
            outputs="xgb_models_dict",
            name="train_individual_xgboost_models"
        ),
        node(
            func=predict_with_individual_models,
            inputs=["xgb_models_dict", "X_test_pca"],
            outputs="y_test_pred_scaled",
            name="predict_with_individual_models"
        ),
        node(
            func=inverse_transform_predictions,
            inputs=["y_test_pred_scaled", "target_scaler"],
            outputs="y_test_pred",
            name="inverse_transform_predictions"
        ),
        node(
            func=clean_y_test_for_evaluation,
            inputs=["y_test", "y_test_pred"],
            outputs=["y_test_clean", "y_test_pred_clean"],
            name="clean_y_test_for_evaluation"
        ),
        node(
            func=evaluate_multitarget_regression,
            inputs=["y_test_clean", "y_test_pred_clean"],
            outputs="regression_metrics",
            name="evaluate_model"
        ),
    ])  # type: ignore
