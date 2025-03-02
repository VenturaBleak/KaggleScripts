#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Kaggle Training Script for Regression with Saving

This script:
  1. Loads preprocessed train and test data from data/.
  2. Reduces memory usage.
  3. Converts categorical features for LGBM/XGBoost using BinaryEncoder.
     CatBoost uses the original data.
  4. Tunes hyperparameters for LightGBM, XGBoost, and CatBoost via Optuna.
  5. Performs K-Fold CV training for each model.
  6. Saves trained models to models/ and saves OOF and test predictions to predictions/.
  7. Optimizes ensemble weights using OOF predictions and saves a test submission to submissions/.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import math
import time
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
import category_encoders as ce
import joblib

# -----------------------
# Global Settings
# -----------------------
seed = 42
np.random.seed(seed)
start_time = time.time()

# Define folder structure relative to the competition folder.
COMPETITION_NAME = "playground-series-s5e2"
DATA_DIR = os.path.join(COMPETITION_NAME, "data")
MODELS_DIR = os.path.join(COMPETITION_NAME, "models")
PREDICTIONS_DIR = os.path.join(COMPETITION_NAME, "predictions")
SUBMISSIONS_DIR = os.path.join(COMPETITION_NAME, "submissions")

for folder in [MODELS_DIR, PREDICTIONS_DIR, SUBMISSIONS_DIR]:
    os.makedirs(folder, exist_ok=True)

# -----------------------
# Memory Optimization Function
# -----------------------
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print(f"Initial memory usage: {start_mem:.2f} MB")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
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
        for train_idx, valid_idx in cv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
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
# Cross-Validation & OOF Predictions
# -----------------------
def train_cv_models(X_conv, X_cat, y, best_lgb_params, best_xgb_params, best_cat_params, test_conv, test_cat, n_splits=5):
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    rmse_lgb_list, rmse_xgb_list, rmse_cat_list = [], [], []
    lgb_best_iters = []
    test_preds_lgb_list, test_preds_xgb_list, test_preds_cat_list = [], [], []
    fold_no = 1
    for train_idx, valid_idx in folds.split(y):
        print(f"\n--- Fold {fold_no} ---")
        X_train_conv, X_valid_conv = X_conv.iloc[train_idx], X_conv.iloc[valid_idx]
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
        test_preds_lgb = model_lgb.predict(test_conv, num_iteration=model_lgb.best_iteration)
        test_preds_lgb_list.append(test_preds_lgb)
        joblib.dump(model_lgb, os.path.join(MODELS_DIR, f"lgb_model_fold{fold_no}.pkl"))

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
        test_preds_xgb = model_xgb.predict(test_conv)
        test_preds_xgb_list.append(test_preds_xgb)
        joblib.dump(model_xgb, os.path.join(MODELS_DIR, f"xgb_model_fold{fold_no}.pkl"))

        # CatBoost
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
        test_preds_cat = model_cat.predict(test_cat)
        test_preds_cat_list.append(test_preds_cat)
        model_cat.save_model(os.path.join(MODELS_DIR, f"cat_model_fold{fold_no}.cbm"))

        fold_no += 1

    print("\nCV Results:")
    print(f"Average LightGBM RMSE: {np.mean(rmse_lgb_list):.5f}")
    print(f"Average XGBoost RMSE: {np.mean(rmse_xgb_list):.5f}")
    print(f"Average CatBoost RMSE: {np.mean(rmse_cat_list):.5f}")

    avg_test_preds_lgb = np.mean(np.array(test_preds_lgb_list), axis=0)
    avg_test_preds_xgb = np.mean(np.array(test_preds_xgb_list), axis=0)
    avg_test_preds_cat = np.mean(np.array(test_preds_cat_list), axis=0)

    # Save predictions for later ensemble experiments.
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    pd.DataFrame(oof_lgb).to_csv(os.path.join(PREDICTIONS_DIR, "oof_lgb.csv"), index=False)
    pd.DataFrame(oof_xgb).to_csv(os.path.join(PREDICTIONS_DIR, "oof_xgb.csv"), index=False)
    pd.DataFrame(oof_cat).to_csv(os.path.join(PREDICTIONS_DIR, "oof_cat.csv"), index=False)
    pd.DataFrame(avg_test_preds_lgb).to_csv(os.path.join(PREDICTIONS_DIR, "test_preds_lgb.csv"), index=False)
    pd.DataFrame(avg_test_preds_xgb).to_csv(os.path.join(PREDICTIONS_DIR, "test_preds_xgb.csv"), index=False)
    pd.DataFrame(avg_test_preds_cat).to_csv(os.path.join(PREDICTIONS_DIR, "test_preds_cat.csv"), index=False)

    return oof_lgb, oof_xgb, oof_cat, lgb_best_iters, avg_test_preds_lgb, avg_test_preds_xgb, avg_test_preds_cat

# -----------------------
# Ensemble Weight Optimization (Using OOF Predictions)
# -----------------------
def optimize_ensemble(oof_preds, y_true):
    def ensemble_rmse(weights, predictions, true_values):
        ensemble_pred = np.dot(predictions, weights)
        return math.sqrt(np.mean((true_values - ensemble_pred) ** 2))
    n_models = oof_preds.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n_models
    initial_weights = np.full(n_models, 1.0 / n_models)
    result = minimize(ensemble_rmse, initial_weights, args=(oof_preds, y_true),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# -----------------------
# Save Submission File
# -----------------------
def save_submission(competition_name, test_index, preds, target_col="Price"):
    submissions_folder = os.path.join(competition_name, "submissions")
    os.makedirs(submissions_folder, exist_ok=True)
    base_filename = "submission"
    ext = ".csv"
    existing_files = [f for f in os.listdir(submissions_folder) if f.startswith(base_filename) and f.endswith(ext)]
    existing_numbers = [int(f.replace(base_filename, "").replace(ext, "")) for f in existing_files if
                        f.replace(base_filename, "").replace(ext, "").isdigit()]
    next_num = max(existing_numbers) + 1 if existing_numbers else 1
    filename = os.path.join(submissions_folder, f"{base_filename}{next_num}{ext}")
    submission = pd.DataFrame(preds, index=test_index, columns=[target_col])
    submission.to_csv(filename, index=False)
    print(f"Submission saved as: {filename}")

# -----------------------
# Main
# -----------------------
def main():
    COMPETITION_NAME = "playground-series-s5e2"
    train_path = os.path.join(COMPETITION_NAME, "data", "train_preprocessed.csv")
    test_path = os.path.join(COMPETITION_NAME, "data", "test_preprocessed.csv")

    print("Loading preprocessed data...")
    train_df = pd.read_csv(train_path, index_col="id") if "id" in pd.read_csv(train_path, nrows=5).columns else pd.read_csv(train_path)
    test_df = pd.read_csv(test_path, index_col="id") if "id" in pd.read_csv(test_path, nrows=5).columns else pd.read_csv(test_path)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    gc.collect()
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)

    target_col = "Price"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    if target_col in test_df.columns:
        test_df = test_df.drop(columns=[target_col])
    print("Features shape:", X.shape, "Target shape:", y.shape)

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
    X_cat = X.copy()
    test_cat = test_df.copy()

    print("\nTuning hyperparameters...")
    best_lgb_params = tune_lgb(X_conv, y, n_trials=200)  # For quick demo, use n_trials=1; in practice, use more
    best_xgb_params = tune_xgb(X_conv, y, n_trials=60)
    best_cat_params = tune_cat(X_cat, y, n_trials=15)

    print("\nStarting K-Fold CV training...")
    oof_lgb, oof_xgb, oof_cat, lgb_best_iters, avg_test_preds_lgb, avg_test_preds_xgb, avg_test_preds_cat = \
        train_cv_models(X_conv, X_cat, y, best_lgb_params, best_xgb_params, best_cat_params, test_conv, test_cat, n_splits=5)

    print("\nOptimizing ensemble weights using OOF predictions...")
    oof_matrix = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
    optimal_weights = optimize_ensemble(oof_matrix, y.values)
    ensemble_oof_rmse = math.sqrt(mean_squared_error(y, np.dot(oof_matrix, optimal_weights)))
    print("Optimal ensemble weights:")
    print(f"  LightGBM: {optimal_weights[0]:.4f}, XGBoost: {optimal_weights[1]:.4f}, CatBoost: {optimal_weights[2]:.4f}")
    print(f"Ensemble OOF RMSE: {ensemble_oof_rmse:.5f}")

    cv_test_pred_matrix = np.vstack([avg_test_preds_lgb, avg_test_preds_xgb, avg_test_preds_cat]).T
    final_ensemble_pred = np.dot(cv_test_pred_matrix, optimal_weights)
    print("Generated final ensemble test predictions from CV folds.")

    save_submission(COMPETITION_NAME, test_df.index, final_ensemble_pred, target_col=target_col)
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()