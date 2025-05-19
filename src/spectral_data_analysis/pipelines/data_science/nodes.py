import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def drop_unwanted_columns(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    return df.drop(columns=drop_cols, errors='ignore')  # ignore errors if col not present

def train_test_split_multitarget(
    df: pd.DataFrame,
    target_cols: list,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Splits data into train/test sets for multitarget regression (no stratification).

    Parameters:
    - df: Input DataFrame
    - target_cols: List of target column names
    - test_size: Fraction of data to use as test set
    - random_state: Seed for reproducibility

    Returns:
    - Dict with keys: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def encode(train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_cols: list,):
    
    train_df_encoded = pd.get_dummies(train_df, columns=categorical_cols)
    test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols)

    test_df_encoded = test_df_encoded.reindex(columns=train_df_encoded.columns, fill_value=0)
    
    return train_df_encoded, test_df_encoded

def scale_features(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies standard scaling to the train and test data (mean=0, std=1).

    Returns:
        Tuple of scaled train and test dataframes.
    """
    scaler = StandardScaler()
    
    # Fit only on train, then transform both
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_data),
        columns=train_data.columns,
        index=train_data.index
    )
    
    test_scaled = pd.DataFrame(
        scaler.transform(test_data),
        columns=test_data.columns,
        index=test_data.index
    )
    
    return train_scaled, test_scaled, scaler


def scale_targets(y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale the target variables using StandardScaler.

    Returns:
        Scaled y_train, y_test, and the fitted scaler (for inverse_transform)
    """
    scaler = StandardScaler()
    y_train_scaled = pd.DataFrame(
        scaler.fit_transform(y_train),
        columns=y_train.columns,
        index=y_train.index
    )
    y_test_scaled = pd.DataFrame(
        scaler.transform(y_test),
        columns=y_test.columns,
        index=y_test.index
    )
    return y_train_scaled, y_test_scaled, scaler

# def inverse_scale_targets(y_scaled: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
#     return pd.DataFrame(
#         scaler.inverse_transform(y_scaled),
#         columns=y_scaled.columns,
#         index=y_scaled.index
#     )

def get_spectral_columns(df: pd.DataFrame) -> list:
    return [col for col in df.columns if col.endswith('_real') or col.endswith('_imag')]
    # return [col for col in df.columns if col.endswith('_magnitude') or col.endswith('_phase')]

def apply_pca_on_spectral(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: float = 0.99
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    
    spectral_cols = get_spectral_columns(X_train)
    
    X_train_spec = X_train[spectral_cols]
    X_test_spec = X_test[spectral_cols]

    X_train_meta = X_train.drop(columns=spectral_cols)
    X_test_meta = X_test.drop(columns=spectral_cols)

    pca = PCA(n_components=n_components)
    X_train_pca_arr = pca.fit_transform(X_train_spec)
    X_test_pca_arr = pca.transform(X_test_spec)

    pca_cols = [f'PC{i+1}' for i in range(X_train_pca_arr.shape[1])]
    X_train_pca_df = pd.DataFrame(X_train_pca_arr, columns=pca_cols, index=X_train.index)
    X_test_pca_df = pd.DataFrame(X_test_pca_arr, columns=pca_cols, index=X_test.index)

    X_train_final = pd.concat([X_train_meta, X_train_pca_df], axis=1)
    X_test_final = pd.concat([X_test_meta, X_test_pca_df], axis=1)

    return X_train_final, X_test_final, pca

def drop_unwanted_columns(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    return df.drop(columns=drop_cols, errors="ignore")

def select_features_by_any_target(
    df: pd.DataFrame,
    target_cols: list,
    threshold: float = 0.2
) -> pd.DataFrame:
    """
    Select features whose absolute correlation with any target is >= threshold.
    Returns df with only selected features + target columns.
    """
    X = df.drop(columns=target_cols)
    y = df[target_cols]  # Correct: DataFrame of targets

    selected_features = []
    for feature in X.columns:
        # Calculate correlation of feature with each target column
        corrs = y.apply(lambda target: np.corrcoef(X[feature], target)[0, 1])
        max_corr = corrs.abs().max()
        if max_corr >= threshold:
            selected_features.append(feature)

    print(f"Selected features based on correlation threshold {threshold}:")
    print(selected_features)

    # Return DataFrame with selected features + all targets
    return df[selected_features + target_cols]

