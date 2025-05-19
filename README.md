# 🍎 Fruit Quality Prediction using Spectral Data (Kedro Project)

This Kedro project implements a robust machine learning pipeline to process spectral and physicochemical fruit data and predict quality attributes like **Firmness**, **Brix**, and **Titratable Acidity (TA)**. The workflow is organized into three modular pipelines:

- `data_processing`: Prepares and cleans raw data.
- `data_science`: Performs feature engineering and dimensionality reduction.
- `machine_learning`: Tunes, trains, and evaluates regression models.

---

## 📂 Project Structure

├── data/                       # Raw, intermediate, and model output datasets
│   ├── 01_raw/                # Original input data (e.g., CSV from sensor)
│   ├── 02_intermediate/       # Processed data after cleaning, denoising, outlier removal
│   ├── 03_primary/            # Train/test splits, scaled and PCA-transformed data
│   ├── 06_models/             # Saved models, scalers, and PCA objects
│   └── 08_reporting/          # (Optional) Visual outputs, metrics, or plots
│
├── conf/
│   └── base/
│       ├── parameters.yml     # Project parameters (target columns, PCA settings, etc.)
│       └── catalog.yml        # Data catalog: dataset definitions and file paths
│
├── notebooks/                 # Exploratory data analysis (EDA), debugging, visualizations
│
├── src/                       # Source code of the Kedro project
│   └── pipelines/
│       ├── data_processing/   # Data cleaning, denoising, outlier detection
│       ├── data_science/      # Feature engineering, scaling, PCA
│       └── machine_learning/  # Model tuning, training, prediction, and evaluation
│
├── pyproject.toml             # Project dependencies and configuration
└── README.md                  # Project documentation (this file)

---

## 🔁 Pipelines Overview

### 1. 🧼 `data_processing`

Transforms raw sensor data into cleaned, outlier-free, and usable format for modeling.

**Steps:**
- `clean_raw_data`: Basic cleaning and preprocessing.
- `data_consistency_check`: Structural and integrity checks.
- `denoise_spectrum_data`: Smoothens spectral curves.
- `process_repeated_measurements`: Aggregates replicates per fruit.
- `isolation_forest_outlier_detection`: Flags anomalies.
- `filter_outliers`: Removes flagged rows.
- `convert_to_magnitude_phase`: Converts complex values to magnitude and phase.

**Inputs:** `raw_data`  
**Output:** Either filtered_data (real + imaginary format) or magphase_data (magnitude + phase format), depending on use case.

---

### 2. 🧪 `data_science`

Handles feature selection, scaling, and PCA-based dimensionality reduction.

**Steps:**
- `drop_unwanted_columns`: Removes uninformative or redundant columns.
- `train_test_split_multitarget`: Multi-target split with custom target columns.
- `scale_features` & `scale_targets`: Standardization using `StandardScaler`.
- `apply_pca_on_spectral`: Reduces dimensionality using PCA (99% variance retained).

**Inputs:** `magphase_data`, `params`  
**Outputs:** Scaled and PCA-transformed datasets for modeling

---

### 3. 🤖 `machine_learning`

Trains one **XGBoost model per target**, optimizes hyperparameters using **Optuna**, and evaluates performance.

**Steps:**
- `tune_xgboost_for_each_target`: Uses Optuna for hyperparameter tuning (per target).
- `train_individual_models`: Trains separate models for each target variable.
- `predict_with_individual_models`: Generates predictions.
- `inverse_transform_predictions`: Applies inverse scaling to predictions.
- `evaluate_multitarget_regression`: Calculates regression metrics (R², RMSE).
- `clean_y_test_for_evaluation`: Aligns predictions and true labels.

**Outputs:** `xgb_models_dict`, `regression_metrics`

---

## 📊 Data Catalog

Key datasets and models managed via `conf/base/catalog.yml`:

| Name                        | Type                 | Location                                 |
|-----------------------------|----------------------|------------------------------------------|
| raw_data                   | `CSV`                | `data/01_raw/Assignment_DataScientist_*.csv` |
| magphase_data              | `CSV`                | `data/02_intermediate/`                  |
| train/test splits          | `CSV`                | `data/03_primary/`                       |
| Scalers & PCA model        | `PickleDataset`      | `data/06_models/`                        |
| Trained XGBoost models     | `PickleDataset`      | `data/06_models/regression_models.pkl`  |

---

## ⚙️ Configuration

All parameters are set in `conf/base/parameters.yml`.


▶️ How to Run
Install Dependencies:

```
pip install -r src/requirements.txt
```
Run the Full Pipeline:

```
kedro run
```

Visualize DAG:

```
kedro viz
```

✅ Outputs

After a successful run, you'll obtain:

- Cleaned and denoised spectral data
- Filtered real/imaginary or magnitude/phase datasets
- Scaled and PCA-reduced train/test splits
- Trained XGBoost models per target
- Regression metrics (RMSE, R²)
