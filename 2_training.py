#!/usr/bin/env python
"""
Enhanced Kaggle Training Script for Regression

This script:
  1. Loads preprocessed train and test data.
  2. Reduces memory usage.
  3. Converts categorical features for models that require it (LGBM, XGBoost) using BinaryEncoder.
     CatBoost uses the original data and its native handling of categoricals.
  4. Tunes hyperparameters for LightGBM, XGBoost, and CatBoost via Optuna.
  5. Performs K-Fold CV training for each model and collects OOF predictions and CV test predictions.
  6. Optimizes ensemble weights using the OOF predictions.
  7. Generates final test predictions by averaging CV test predictions and applying ensemble weights.
  8. Saves the submission file with an incremented filename for later ensemble averaging.

Author: Your Name
Date: YYYY-MM-DD
"""

import math
import time
import os
import gc
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy.optimize import minimize

# Import category encoders for binary encoding
import category_encoders as ce

# -----------------------
# Global Settings
# -----------------------
seed = 42
np.random.seed(seed)
start_time = time.time()

# -----------------------
# Memory Optimization Function
# -----------------------
def reduce_mem_usage(df):
    """Iterate through all columns of a DataFrame and downcast numeric types to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print(f"Initial memory usage: {start_mem:.2f} MB")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            # Integer types
            if np.issubdtype(col_type, np.integer):
                if c_min >= 0:
                    if c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
            # Floating types
            elif np.issubdtype(col_type, np.floating):
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    gc.collect()
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f"Reduced memory usage: {end_mem:.2f} MB (Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%)")
    return df

# -----------------------
# Tqdm Callback for Optuna
# -----------------------
class TqdmCallback:
    def __init__(self, pbar):
        self.pbar = pbar
        self.best_rmse = float("inf")

    def __call__(self, study, trial):
        if study.best_value < self.best_rmse:
            self.best_rmse = study.best_value
            self.pbar.set_postfix({"Best RMSE": f"{self.best_rmse:.5f}"})
            self.pbar.refresh()
        self.pbar.update(1)

# -----------------------
# Hyperparameter Tuning Functions
# -----------------------
def tune_lgb(X, y, n_trials):
    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'seed': seed
        }
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        scores = []
        for train_idx, valid_idx in cv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            model = lgb.train(param, train_data, num_boost_round=10000, valid_sets=[valid_data],
                              callbacks=[lgb.early_stopping(50)])
            preds = model.predict(X_valid, num_iteration=model.best_iteration)
            scores.append(math.sqrt(mean_squared_error(y_valid, preds)))
        return np.mean(scores)

    pbar = tqdm(total=n_trials, desc="LGBM Trials", dynamic_ncols=True)
    study = optuna.create_study(direction='minimize', study_name="lgbm_regression")
    study.optimize(objective, n_trials=n_trials, callbacks=[TqdmCallback(pbar)])
    pbar.close()
    best_params = study.best_params.copy()
    best_params.update({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
                        'boosting_type': 'gbdt', 'seed': seed})
    print(f"LightGBM best RMSE: {study.best_value:.5f}")
    print("Best LightGBM params:", best_params)
    return best_params


def tune_xgb(X, y, n_trials):
    def objective(trial):
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'seed': seed
        }
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        scores = []
        for train_idx, valid_idx in cv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            model = xgb.XGBRegressor(n_estimators=1000, tree_method='auto',
                                     objective='reg:squarederror', random_state=seed,
                                     callbacks=[xgb.callback.EarlyStopping(rounds=50, maximize=False)],
                                     verbosity=0, **param)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            preds = model.predict(X_valid)
            scores.append(math.sqrt(mean_squared_error(y_valid, preds)))
        return np.mean(scores)

    pbar = tqdm(total=n_trials, desc="XGB Trials", dynamic_ncols=True)
    study = optuna.create_study(direction='minimize', study_name="xgb_regression")
    study.optimize(objective, n_trials=n_trials, callbacks=[TqdmCallback(pbar)])
    pbar.close()
    best_params = study.best_params.copy()
    best_params.update({'seed': seed})
    print(f"XGBoost best RMSE: {study.best_value:.5f}")
    print("Best XGBoost params:", best_params)
    return best_params


def tune_cat(X, y, n_trials):
    def objective(trial):
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
        }
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        scores = []
        # X is the original (unencoded) data with categorical columns.
        for train_idx, valid_idx in cv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            # Get categorical feature names for CatBoost
            cat_features = list(X_train.select_dtypes(include=["object"]).columns)
            model = CatBoostRegressor(iterations=1000, eval_metric='RMSE', random_seed=seed, verbose=False, **param)
            model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_valid, y_valid), early_stopping_rounds=50)
            preds = model.predict(X_valid)
            scores.append(math.sqrt(mean_squared_error(y_valid, preds)))
        return np.mean(scores)

    pbar = tqdm(total=n_trials, desc="CatBoost Trials", dynamic_ncols=True)
    study = optuna.create_study(direction='minimize', study_name="catboost_regression")
    study.optimize(objective, n_trials=n_trials, callbacks=[TqdmCallback(pbar)])
    pbar.close()
    best_params = study.best_params.copy()
    best_params.update({'random_seed': seed})
    print(f"CatBoost best RMSE: {study.best_value:.5f}")
    print("Best CatBoost params:", best_params)
    return best_params


# -----------------------
# Cross-Validation & OOF Predictions (with Test Predictions)
# -----------------------
def train_cv_models(X_conv, X_cat, y, best_lgb_params, best_xgb_params, best_cat_params, test_conv, test_cat, n_splits=5):
    """
    X_conv: DataFrame for LightGBM and XGBoost (with binary-encoded categoricals)
    X_cat:   Original DataFrame for CatBoost (with object columns)
    test_conv: Test set for LGBM/XGB
    test_cat: Test set for CatBoost
    Returns:
      oof_lgb, oof_xgb, oof_cat, lgb_best_iters,
      avg_test_preds_lgb, avg_test_preds_xgb, avg_test_preds_cat
    """
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    rmse_lgb_list, rmse_xgb_list, rmse_cat_list = [], [], []
    lgb_best_iters = []

    # Lists for test predictions per fold
    test_preds_lgb_list = []
    test_preds_xgb_list = []
    test_preds_cat_list = []

    fold_no = 1
    for train_idx, valid_idx in folds.split(y):
        print(f"\n--- Fold {fold_no} ---")
        # For LGBM and XGB, use the converted data.
        X_train_conv, X_valid_conv = X_conv.iloc[train_idx], X_conv.iloc[valid_idx]
        # For CatBoost, use the original data.
        X_train_cat, X_valid_cat = X_cat.iloc[train_idx], X_cat.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # LightGBM
        lgb_train = lgb.Dataset(X_train_conv, label=y_train)
        lgb_valid = lgb.Dataset(X_valid_conv, label=y_valid)
        model_lgb = lgb.train(best_lgb_params, lgb_train, num_boost_round=10000, valid_sets=[lgb_valid],
                              callbacks=[lgb.early_stopping(50)])
        preds_lgb = model_lgb.predict(X_valid_conv, num_iteration=model_lgb.best_iteration)
        oof_lgb[valid_idx] = preds_lgb
        rmse_lgb = math.sqrt(mean_squared_error(y_valid, preds_lgb))
        rmse_lgb_list.append(rmse_lgb)
        lgb_best_iters.append(model_lgb.best_iteration)
        print(f"LightGBM RMSE: {rmse_lgb:.5f} (Iter: {model_lgb.best_iteration})")

        # Generate test predictions for LGBM
        test_preds_lgb = model_lgb.predict(test_conv, num_iteration=model_lgb.best_iteration)
        test_preds_lgb_list.append(test_preds_lgb)

        # XGBoost
        model_xgb = xgb.XGBRegressor(n_estimators=1000, tree_method='auto',
                                     objective='reg:squarederror', random_state=seed,
                                     callbacks=[xgb.callback.EarlyStopping(rounds=50, maximize=False)],
                                     verbosity=0, **best_xgb_params)
        model_xgb.fit(X_train_conv, y_train, eval_set=[(X_valid_conv, y_valid)], verbose=False)
        preds_xgb = model_xgb.predict(X_valid_conv)
        oof_xgb[valid_idx] = preds_xgb
        rmse_xgb = math.sqrt(mean_squared_error(y_valid, preds_xgb))
        rmse_xgb_list.append(rmse_xgb)
        print(f"XGBoost RMSE: {rmse_xgb:.5f}")

        # Generate test predictions for XGBoost
        test_preds_xgb = model_xgb.predict(test_conv)
        test_preds_xgb_list.append(test_preds_xgb)

        # CatBoost (uses native categoricals)
        cat_features = list(X_train_cat.select_dtypes(include=["object"]).columns)
        model_cat = CatBoostRegressor(iterations=1000, eval_metric='RMSE',
                                      verbose=False, **best_cat_params)
        model_cat.fit(X_train_cat, y_train, cat_features=cat_features,
                      eval_set=(X_valid_cat, y_valid), early_stopping_rounds=50)
        preds_cat = model_cat.predict(X_valid_cat)
        oof_cat[valid_idx] = preds_cat
        rmse_cat = math.sqrt(mean_squared_error(y_valid, preds_cat))
        rmse_cat_list.append(rmse_cat)
        print(f"CatBoost RMSE: {rmse_cat:.5f}")

        # Generate test predictions for CatBoost
        test_preds_cat = model_cat.predict(test_cat)
        test_preds_cat_list.append(test_preds_cat)

        fold_no += 1

    print("\nCV Results:")
    print(f"Average LightGBM RMSE: {np.mean(rmse_lgb_list):.5f}")
    print(f"Average XGBoost RMSE: {np.mean(rmse_xgb_list):.5f}")
    print(f"Average CatBoost RMSE: {np.mean(rmse_cat_list):.5f}")

    # Average test predictions over folds
    avg_test_preds_lgb = np.mean(np.array(test_preds_lgb_list), axis=0)
    avg_test_preds_xgb = np.mean(np.array(test_preds_xgb_list), axis=0)
    avg_test_preds_cat = np.mean(np.array(test_preds_cat_list), axis=0)

    return oof_lgb, oof_xgb, oof_cat, lgb_best_iters, avg_test_preds_lgb, avg_test_preds_xgb, avg_test_preds_cat

# -----------------------
# Ensemble Weight Optimization
# -----------------------
def optimize_ensemble(oof_preds, y_true):
    # oof_preds: shape (n_samples, n_models)
    def ensemble_rmse(weights, predictions, true_values):
        ensemble_pred = np.dot(predictions, weights)
        return math.sqrt(mean_squared_error(true_values, ensemble_pred))

    n_models = oof_preds.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n_models
    initial_weights = np.repeat(1.0 / n_models, n_models)
    result = minimize(ensemble_rmse, initial_weights, args=(oof_preds, y_true),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# -----------------------
# Final Model Training & Test Predictions (Not used in final prediction generation)
# -----------------------
def train_final_models(X_conv, X_cat, y, best_lgb_params, best_xgb_params, best_cat_params, lgb_rounds):
    # Train LGBM and XGB on converted data; CatBoost on original data.
    final_lgb = lgb.train(best_lgb_params, lgb.Dataset(X_conv, label=y), num_boost_round=lgb_rounds,
                          callbacks=[lgb.early_stopping(50)])
    final_xgb = xgb.XGBRegressor(n_estimators=1000, tree_method='auto',
                                 objective='reg:squarederror', random_state=seed,
                                 verbosity=0, **best_xgb_params)
    final_xgb.fit(X_conv, y, verbose=False)
    cat_features = list(X_cat.select_dtypes(include=["object"]).columns)
    final_cat = CatBoostRegressor(iterations=1000, eval_metric='RMSE',
                                  verbose=False, **best_cat_params)
    final_cat.fit(X_cat, y, cat_features=cat_features)
    return final_lgb, final_xgb, final_cat

def generate_test_predictions(test_conv, test_cat, final_models, ensemble_weights):
    final_lgb, final_xgb, final_cat = final_models
    preds_lgb = final_lgb.predict(test_conv, num_iteration=getattr(final_lgb, "best_iteration", None))
    preds_xgb = final_xgb.predict(test_conv)
    preds_cat = final_cat.predict(test_cat)
    pred_matrix = np.vstack([preds_lgb, preds_xgb, preds_cat]).T
    ensemble_pred = np.dot(pred_matrix, ensemble_weights)
    return preds_lgb, preds_xgb, preds_cat, ensemble_pred

# -----------------------
# Save Submission File (with incremented numbering)
# -----------------------
def save_submission(competition_name, test_index, preds, target_col="Price"):
    """
    Saves the submission file in the competition's 'submissions' folder with an incremented filename.

    Parameters:
    - competition_name (str): The name of the competition (used for directory structure).
    - test_index (pd.Index): The index of the test set for submission.
    - preds (np.array): Predictions for the test set.
    - target_col (str): Name of the target column in the submission file.
    """
    # Define the submissions folder path
    submissions_folder = os.path.join(competition_name, "submissions")
    os.makedirs(submissions_folder, exist_ok=True)  # Ensure directory exists

    # Find existing numbered submissions
    base_filename = "submission"
    ext = ".csv"
    existing_files = [f for f in os.listdir(submissions_folder) if f.startswith(base_filename) and f.endswith(ext)]

    # Extract existing submission numbers
    existing_numbers = [
        int(f.replace(base_filename, "").replace(ext, ""))
        for f in existing_files if f.replace(base_filename, "").replace(ext, "").isdigit()
    ]

    # Determine next available submission number
    next_num = max(existing_numbers) + 1 if existing_numbers else 1
    filename = os.path.join(submissions_folder, f"{base_filename}{next_num}{ext}")

    # Save the submission file
    submission = pd.DataFrame(preds, index=test_index, columns=[target_col])
    submission.to_csv(filename)

    print(f"Submission saved as: {filename}")

# -----------------------
# Main
# -----------------------
def main():
    # File paths (adjust if needed)
    COMPETITION_NAME = "playground-series-s5e2"
    train_path = os.path.join(COMPETITION_NAME, "data", "train_preprocessed.csv")
    test_path = os.path.join(COMPETITION_NAME, "data", "test_preprocessed.csv")

    print("Loading preprocessed data...")
    train_df = pd.read_csv(train_path, index_col="id") if "id" in pd.read_csv(train_path, nrows=5).columns else pd.read_csv(train_path)
    test_df = pd.read_csv(test_path, index_col="id") if "id" in pd.read_csv(test_path, nrows=5).columns else pd.read_csv(test_path)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Reduce memory usage
    gc.collect()
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)

    # Prepare features and target
    target_col = "Price"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    if target_col in test_df.columns:
        test_df = test_df.drop(columns=[target_col])
    print("Features shape:", X.shape, "Target shape:", y.shape)

    # Create two versions of features:
    # - For LGBM and XGBoost: convert categorical features using BinaryEncoder.
    # - For CatBoost: use the original data (CatBoost handles categoricals natively).
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        print("Converting categorical columns using BinaryEncoder for LGBM/XGB:", categorical_cols)
        encoder = ce.BinaryEncoder(cols=categorical_cols, return_df=True)
        X_conv = encoder.fit_transform(X)
        test_conv = encoder.transform(test_df)
        print("Converted features head:")
        print(X_conv.head())
    else:
        X_conv = X.copy()
        test_conv = test_df.copy()
    # For CatBoost, use the original data.
    X_cat = X.copy()
    test_cat = test_df.copy()

    # --- 1. Hyperparameter Tuning ---
    print("\nTuning hyperparameters...")
    best_lgb_params = tune_lgb(X_conv, y, n_trials=150)
    best_xgb_params = tune_xgb(X_conv, y, n_trials=60)
    best_cat_params = tune_cat(X_cat, y, n_trials=15)

    # --- 2. CV Training & OOF Predictions ---
    print("\nStarting K-Fold CV training...")
    oof_lgb, oof_xgb, oof_cat, lgb_best_iters, avg_test_preds_lgb, avg_test_preds_xgb, avg_test_preds_cat = \
        train_cv_models(X_conv, X_cat, y, best_lgb_params, best_xgb_params, best_cat_params, test_conv, test_cat, n_splits=5)

    # Optimize ensemble weights using OOF predictions
    print("\nOptimizing ensemble weights...")
    oof_matrix = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
    optimal_weights = optimize_ensemble(oof_matrix, y.values)
    ensemble_oof_rmse = math.sqrt(mean_squared_error(y, np.dot(oof_matrix, optimal_weights)))
    print("Optimal ensemble weights:")
    print(f"  LightGBM: {optimal_weights[0]:.4f}, XGBoost: {optimal_weights[1]:.4f}, CatBoost: {optimal_weights[2]:.4f}")
    print(f"Ensemble OOF RMSE: {ensemble_oof_rmse:.5f}")

    # --- 3. Generate Final Test Predictions from CV Folds ---
    # Stack the averaged test predictions from CV for each model
    cv_test_pred_matrix = np.vstack([avg_test_preds_lgb, avg_test_preds_xgb, avg_test_preds_cat]).T
    # Combine them using the optimized ensemble weights
    final_ensemble_pred = np.dot(cv_test_pred_matrix, optimal_weights)
    print("Generated final ensemble test predictions from CV folds.")

    # Save submission file under COMPETITION_NAME / submissions /
    save_submission(COMPETITION_NAME, test_df.index, final_ensemble_pred, target_col=target_col)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time / 60:.2f} minutes")

if __name__ == "__main__":
    main()