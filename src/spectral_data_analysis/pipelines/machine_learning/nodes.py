import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Store models and parameters
trained_models = {}
best_params_dict = {}

def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "random_state": 42,
        "tree_method": "hist"
    }

    model = XGBRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    return np.mean(scores)


def tune_xgboost_for_each_target(X_train_scaled: pd.DataFrame, y_train_scaled: pd.DataFrame, n_trials: int) -> dict:
    best_params = {}

    for target in y_train_scaled.columns:
        print(f"\n🔍 Tuning model for target: {target}")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train_scaled, y_train_scaled[target]), n_trials=n_trials)
        best_params[target] = study.best_trial.params

    return best_params

def train_individual_models(X_train_scaled, y_train_scaled, best_params, eval_set=None):
    models = {}
    for target in y_train_scaled.columns:
        model = XGBRegressor(**best_params[target])
        if eval_set:
            model.fit(
                X_train_scaled,
                y_train_scaled[target],
                early_stopping_rounds=10,
                eval_set=[eval_set],
                verbose=False
            )
        else:
            model.fit(X_train_scaled, y_train_scaled[target])
        models[target] = model
    return models

def predict_with_individual_models(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    preds = {}

    for target, model in models.items():
        preds[target] = model.predict(X)

    pred_df = pd.DataFrame(preds, index=X.index)
    print(f"Prediction DataFrame info:\n{pred_df.info()}")
    print(f"Any NaNs in prediction? {pred_df.isnull().sum().sum()}")
    return pred_df


def inverse_transform_predictions(y_pred_scaled: pd.DataFrame, target_scaler) -> pd.DataFrame:
    return pd.DataFrame(
        target_scaler.inverse_transform(y_pred_scaled),
        columns=y_pred_scaled.columns,
        index=y_pred_scaled.index
    )


def evaluate_multitarget_regression(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    metrics = {}
    rmse_per_target = {}
    r2_per_target = {}

    for col in y_true.columns:
        rmse = mean_squared_error(y_true[col], y_pred[col], squared=False)
        r2 = r2_score(y_true[col], y_pred[col])
        rmse_per_target[col] = rmse
        r2_per_target[col] = r2

    metrics["rmse_per_target"] = rmse_per_target
    metrics["r2_per_target"] = r2_per_target
    metrics["rmse_mean"] = np.mean(list(rmse_per_target.values()))
    metrics["r2_mean"] = np.mean(list(r2_per_target.values()))

    print("📊 Evaluation metrics:")
    for target in y_true.columns:
        print(f"Target: {target} - RMSE: {rmse_per_target[target]:.4f}, R2: {r2_per_target[target]:.4f}")
    print(f"Mean RMSE: {metrics['rmse_mean']:.4f}")
    print(f"Mean R2: {metrics['r2_mean']:.4f}")

    return metrics


def clean_y_test_for_evaluation(y_test: pd.DataFrame, y_test_pred: pd.DataFrame):
    y_test_clean = y_test.dropna()
    y_test_pred_clean = y_test_pred.loc[y_test_clean.index]
    return y_test_clean, y_test_pred_clean
