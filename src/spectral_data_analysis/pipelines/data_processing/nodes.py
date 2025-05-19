
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    
    logger.info(f"Data shape before cleaning:{df.shape}")
    # Drop rows with missing critical values
    df = df.dropna(subset=['Experiment', 'TestDate', 'Fruit nr', 'Size', 'Weight', 'Firmness (kg)', 'Brix', 'TA'])
    # Convert TestDate to datetime
    df['TestDate'] = pd.to_datetime(df['TestDate'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['TestDate'])
    logger.info(f"Data shape after cleaning:{df.shape}")
    return df

def data_consistency_check(df: pd.DataFrame) -> pd.DataFrame:
    assert (df['Firmness (kg)'] >= 0).all(), "Negative firmness found in data!"
    assert (df['Weight'] > 0).all(), "Non-positive weight found!"
    return df

def process_group(group: pd.DataFrame, threshold: float = 0.20) -> pd.Series:
    """
    Aggregates a group of measurements corresponding to a single fruit.
    - Averages 'Firmness (kg)', 'Brix', and 'TA'.
    - Averages all spectral columns (real and imaginary).
    - Flags the row as an outlier if the relative difference in firmness exceeds the threshold.

    Parameters:
    - group (pd.DataFrame): Group of rows for a single fruit (usually 2 measurements).
    - threshold (float): Relative difference threshold for outlier detection (default 0.20).

    Returns:
    - pd.Series: Single averaged row with outlier flag and source row tracking.
    """
    row = group.iloc[0].copy()
    source_indices = group.index.tolist()
    source_str = "_".join(str(idx) for idx in source_indices)

    for col in ['Firmness (kg)', 'Brix', 'TA']:
        row[col] = group[col].mean()

    spectral_cols = [col for col in group.columns if '_GHz_real' in col or '_GHz_imag' in col]
    for col in spectral_cols:
        row[col] = group[col].mean()

    if len(group) == 2:
        val1, val2 = group['Firmness (kg)'].values
        avg_val = (val1 + val2) / 2
        diff = abs(val1 - val2)
        rel_diff = diff / avg_val if avg_val != 0 else 0
        row['firmness_outlier_flag'] = rel_diff > threshold
    else:
        row['firmness_outlier_flag'] = True

    row['source_rows'] = f"avg_{source_str}"
    return row

def average_with_outlier_flag(df: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    """
    Group by Experiment, TestDate, Fruit nr, Size, and Weight,
    average measurements if there are two values and their relative difference is within a threshold,
    otherwise flag them as outliers.

    Parameters:
    - df: Input DataFrame with 'Firmness (kg)' measurements
    - threshold: Relative difference threshold to flag outliers (default 0.05)

    Returns:
    - DataFrame with averaged rows, outlier flag, and traceable unique source IDs
    """
    grouped = df.groupby(['Experiment', 'TestDate', 'Fruit nr', 'Size', 'Weight'], group_keys=False)
    result_df = grouped.apply(process_group, threshold=threshold).reset_index(drop=True)

    outlier_count = result_df['firmness_outlier_flag'].sum()
    logger.info(f"Number of flagged outliers based on firmness: {outlier_count}")
    
    return result_df

def process_repeated_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kedro node wrapper function to process firmness measurements.
    """
    return average_with_outlier_flag(df)

def isolation_forest_outlier_detection(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """
    Detects multivariate outliers using Isolation Forest on spectral and scalar features.

    Parameters:
    - df: DataFrame with spectral and scalar columns, must include existing 'outlier_flag'
    - contamination: Expected proportion of outliers
    - random_state: for reproducibility

    Returns:
    - DataFrame with new column 'iforest_outlier_flag' and combined 'final_outlier_flag'
    """

    # Select columns to use for multivariate outlier detection
    spectral_cols = [col for col in df.columns if '_GHz_real' in col or '_GHz_imag' in col]
    scalar_cols = ['sensorT', 'Firmness (kg)', 'Brix', 'TA']
    feature_cols = spectral_cols + scalar_cols

    # Check if all columns exist and drop rows with missing features
    feature_cols = [col for col in feature_cols if col in df.columns]
    data = df[feature_cols].fillna(df[feature_cols].mean())  # simple imputation if needed

    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso_forest.fit_predict(data)

    # IsolationForest predicts -1 for outliers, 1 for inliers
    df['iforest_outlier_flag'] = preds == -1

    # Combine with existing firmness outlier flag
    df['final_outlier_flag'] = df['firmness_outlier_flag'] | df['iforest_outlier_flag']

    n_iforest_outliers = df['iforest_outlier_flag'].sum()
    n_final_outliers = df['final_outlier_flag'].sum()
    total = len(df)

    logger.info(f"Isolation Forest detected {n_iforest_outliers} outliers out of {total} samples ({n_iforest_outliers/total:.2%})")
    logger.info(f"Total outliers after combining flags: {n_final_outliers} ({n_final_outliers/total:.2%})")

    return df

def filter_outliers(df: pd.DataFrame, remove_firmness_outliers: bool = True, remove_iforest_outliers: bool = True) -> pd.DataFrame:
    logger.info(df.shape)
    if remove_firmness_outliers and remove_iforest_outliers:
        # Remove rows flagged by either method
        return df[(df['firmness_outlier_flag'] == False) & (df['iforest_outlier_flag'] == False)].reset_index(drop=True)
    elif remove_iforest_outliers:
        return df[df['iforest_outlier_flag'] == False].reset_index(drop=True)
    elif remove_firmness_outliers:
        return df[df['firmness_outlier_flag'] == False].reset_index(drop=True)
    else:
        return df

def denoise_spectrum_data(df: pd.DataFrame, window_length: int = 7, polyorder: int = 3) -> pd.DataFrame:
    """
    Applies Savitzky-Golay smoothing filter to all real and imaginary microwave spectrum columns.

    Parameters:
    - df: Input DataFrame with spectrum and metadata
    - window_length: Size of the filter window (must be odd)
    - polyorder: Order of the polynomial used to fit the samples

    Returns:
    - df_smoothed: DataFrame with smoothed spectral data, other columns unchanged
    """

    real_cols = [col for col in df.columns if '_GHz_real' in col]
    imag_cols = [col for col in df.columns if '_GHz_imag' in col]
    spectrum_cols = real_cols + imag_cols

    # Apply Savitzky-Golay filter row-wise and reconstruct the DataFrame
    spectrum_filtered = df[spectrum_cols].apply(
        lambda row: pd.Series(savgol_filter(row.values, window_length=window_length, polyorder=polyorder),
                              index=spectrum_cols),
        axis=1
    )

    # Combine with non-spectral metadata columns
    metadata_cols = df.columns.difference(spectrum_cols)
    df_smoothed = pd.concat([spectrum_filtered, df[metadata_cols]], axis=1)
    df_smoothed = df_smoothed[df.columns]

    return df_smoothed

def convert_to_magnitude_phase(df: pd.DataFrame) -> pd.DataFrame:
    # Find matching real and imaginary column pairs
    real_cols = sorted([col for col in df.columns if '_GHz_real' in col])
    imag_cols = sorted([col for col in df.columns if '_GHz_imag' in col])
    
    # Check they match in count and names (except suffix)
    assert len(real_cols) == len(imag_cols), "Mismatch in real and imag columns count"
    
    mag_phase_data = {}
    for real_col, imag_col in zip(real_cols, imag_cols):
        base_name = real_col.replace('_real', '')  # base for new column names
        
        real_vals = df[real_col].values
        imag_vals = df[imag_col].values
        
        magnitude = np.sqrt(real_vals**2 + imag_vals**2)
        phase = np.arctan2(imag_vals, real_vals)  # phase in radians
        
        mag_phase_data[f'{base_name}_magnitude'] = magnitude
        mag_phase_data[f'{base_name}_phase'] = phase
    
    mag_phase_df = pd.DataFrame(mag_phase_data, index=df.index)
    
    # Drop original real and imag columns and concat mag+phase
    df = df.drop(columns=real_cols + imag_cols)
    df = pd.concat([df, mag_phase_df], axis=1)
    
    return df

