#!/usr/bin/env python
# training_script.py

"""
Enhanced Kaggle training script for a regression problem.

This script:
  1. Loads preprocessed data (combined or normal) based on a flag.
  2. Uses Optuna to tune hyperparameters for LightGBM, XGBoost, and CatBoost.
  3. Trains three regressors via K-Fold CV and collects out-of-fold predictions.
  4. Optimizes ensemble weights on the OOF predictions (ensuring non-negativity and summing to 1).
  5. Trains final models on full data and generates test predictions using the optimized ensemble.
  6. Compares individual model performance and the ensemble mix.

Best practices include:
  - Efficient hyperparameter tuning with early stopping.
  - Using diverse models to capture different aspects of the data.
  - Optimizing ensemble weights via constrained minimization.
  - Robust K-Fold CV for reliable OOF predictions.

Author: Your Name
Date: YYYY-MM-DD
"""

import math
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
import sys
from scipy.optimize import minimize

# -----------------------
# 1. Global Settings
# -----------------------
seed = 42
np.random.seed(seed)
start_time = time.time()

# Number of Optuna trials for each model (adjust as needed)
n_trials_lgb = 100  # For LightGBM
n_trials_xgb = 20  # For XGBoost
n_trials_cat = 20  # For CatBoost

warnings.filterwarnings("ignore")  # Suppress warnings for clean output

# ------------------------------------------------------------
# Option: Use combined training data (train + underlying) or normal train data.
USE_COMBINED_TRAIN = False  # Set to True to use combined_train/combined_test

if USE_COMBINED_TRAIN:
    train_data_path = 'playground-series-s5e2/data/combined_train.csv'
    test_data_path  = 'playground-series-s5e2/data/combined_test.csv'
    print("Using COMBINED training data (train + underlying) and combined test data.")
else:
    train_data_path = 'playground-series-s5e2/data/train_preprocessed.csv'
    test_data_path  = 'playground-series-s5e2/data/test_preprocessed.csv'
    print("Using NORMAL training data (train only) and preprocessed test data.")

# -----------------------
# 2. Load Preprocessed Data
# -----------------------
print("\nLoading preprocessed data...")
train_df = pd.read_csv(train_data_path)
if 'id' in train_df.columns:
    train_df.set_index('id', inplace=True)
test_df = pd.read_csv(test_data_path)
if 'id' in test_df.columns:
    test_df.set_index('id', inplace=True)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# -----------------------
# 3. Prepare Features and Target
# -----------------------
target_col = 'Price'
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training data!")

# Separate features and target for training data.
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

# Ensure test data does not contain target.
if target_col in test_df.columns:
    test_df = test_df.drop(columns=[target_col])

print(f"Features shape: {X.shape}, Target shape: {y.shape}")
print("Feature columns:", X.columns.tolist())

# -----------------------
# 4. Hyperparameter Optimization with Optuna
# -----------------------
print("\nStarting hyperparameter optimization...")

# ---- LightGBM Tuning ----
print("\n[LightGBM] Tuning hyperparameters...")
class TqdmCallback:
    """Callback for Optuna to update tqdm progress bar."""
    def __init__(self, pbar):
        self.pbar = pbar
        self.best_rmse = float("inf")
    def __call__(self, study, trial):
        if study.best_value < self.best_rmse:
            self.best_rmse = study.best_value
            self.pbar.set_postfix({"Best RMSE": f"{self.best_rmse:.5f}"})
            self.pbar.refresh()
        self.pbar.update(1)

def lgb_objective(trial):
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
        'max_depth': trial.suggest_int('max_depth', 3, 15)
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    rmse_scores = []
    for train_idx, valid_idx in cv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        gbm = lgb.train(
            param,
            train_data,
            num_boost_round=10000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(100)]
        )
        preds = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
        rmse = math.sqrt(mean_squared_error(y_valid, preds))
        rmse_scores.append(rmse)
    return np.mean(rmse_scores)

pbar_lgb = tqdm(total=n_trials_lgb, desc="LGBM Trials", dynamic_ncols=True)
study_lgb = optuna.create_study(direction='minimize', study_name="lgbm_regression")
study_lgb.optimize(lgb_objective, n_trials=n_trials_lgb, callbacks=[TqdmCallback(pbar_lgb)])
pbar_lgb.close()

print(f"LightGBM best RMSE: {study_lgb.best_value:.5f}")
print("Best LightGBM parameters:", study_lgb.best_params)
best_lgb_params = study_lgb.best_params.copy()
best_lgb_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'seed': seed
})

# ---- XGBoost Tuning ----
print("\n[XGBoost] Tuning hyperparameters...")
def xgb_objective(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    rmse_scores = []
    for train_idx, valid_idx in cv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = xgb.XGBRegressor(
            n_estimators=1000,
            tree_method='auto',
            objective='reg:squarederror',
            random_state=seed,
            callbacks=[xgb.callback.EarlyStopping(rounds=100, maximize=False)],
            verbosity=0,
            **param
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        preds = model.predict(X_valid)
        rmse = math.sqrt(mean_squared_error(y_valid, preds))
        rmse_scores.append(rmse)
    return np.mean(rmse_scores)

pbar_xgb = tqdm(total=n_trials_xgb, desc="XGB Trials", dynamic_ncols=True)
study_xgb = optuna.create_study(direction='minimize', study_name="xgb_regression")
study_xgb.optimize(xgb_objective, n_trials=n_trials_xgb, callbacks=[TqdmCallback(pbar_xgb)])
pbar_xgb.close()

print(f"XGBoost best RMSE: {study_xgb.best_value:.5f}")
print("Best XGBoost parameters:", study_xgb.best_params)
best_xgb_params = study_xgb.best_params.copy()

# ---- CatBoost Tuning ----
print("\n[CatBoost] Tuning hyperparameters...")
def cat_objective(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    rmse_scores = []
    for train_idx, valid_idx in cv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = CatBoostRegressor(
            iterations=1000,
            eval_metric='RMSE',
            random_seed=seed,
            early_stopping_rounds=100,
            verbose=False,
            **param
        )
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        preds = model.predict(X_valid)
        rmse = math.sqrt(mean_squared_error(y_valid, preds))
        rmse_scores.append(rmse)
    return np.mean(rmse_scores)

pbar_cat = tqdm(total=n_trials_cat, desc="CatBoost Trials", dynamic_ncols=True)
study_cat = optuna.create_study(direction='minimize', study_name="catboost_regression")
study_cat.optimize(cat_objective, n_trials=n_trials_cat, callbacks=[TqdmCallback(pbar_cat)])
pbar_cat.close()
print(f"CatBoost best RMSE: {study_cat.best_value:.5f}")
print("Best CatBoost parameters:", study_cat.best_params)
best_cat_params = study_cat.best_params.copy()

# -----------------------
# 5. K-Fold Cross-Validation for Model Training & OOF Predictions
# -----------------------
print("\nStarting K-Fold cross-validation with LightGBM, XGBoost, and CatBoost...")
n_splits = 5
folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# Arrays for out-of-fold (OOF) predictions.
oof_preds_lgb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
oof_preds_cat = np.zeros(len(X))

# Lists to store RMSE values and best iterations (for LightGBM).
rmse_list_lgb = []
rmse_list_xgb = []
rmse_list_cat = []
best_iterations = []

fold_idx = 1
for train_idx, valid_idx in folds.split(X):
    print(f"\n--- Fold {fold_idx} ---")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # --- LightGBM ---
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)
    model_lgb = lgb.train(
        best_lgb_params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(100)]
    )
    preds_lgb = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    oof_preds_lgb[valid_idx] = preds_lgb
    rmse_lgb = math.sqrt(mean_squared_error(y_valid, preds_lgb))
    rmse_list_lgb.append(rmse_lgb)
    best_iterations.append(model_lgb.best_iteration)
    print(f"LightGBM RMSE: {rmse_lgb:.5f} | Best Iterations: {model_lgb.best_iteration}")

    # --- XGBoost ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        tree_method='auto',
        objective='reg:squarederror',
        random_state=seed,
        callbacks=[xgb.callback.EarlyStopping(rounds=100, maximize=False)],
        verbosity=0,
        **best_xgb_params
    )
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    preds_xgb = model_xgb.predict(X_valid)
    oof_preds_xgb[valid_idx] = preds_xgb
    rmse_xgb = math.sqrt(mean_squared_error(y_valid, preds_xgb))
    rmse_list_xgb.append(rmse_xgb)
    print(f"XGBoost RMSE: {rmse_xgb:.5f}")

    # --- CatBoost ---
    model_cat = CatBoostRegressor(
        iterations=1000,
        eval_metric='RMSE',
        random_seed=seed,
        verbose=False,
        **best_cat_params
    )
    model_cat.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    preds_cat = model_cat.predict(X_valid)
    oof_preds_cat[valid_idx] = preds_cat
    rmse_cat = math.sqrt(mean_squared_error(y_valid, preds_cat))
    rmse_list_cat.append(rmse_cat)
    print(f"CatBoost RMSE: {rmse_cat:.5f}")

    fold_idx += 1

print("\nCross-Validation Results:")
print(f"Average LightGBM RMSE: {np.mean(rmse_list_lgb):.5f}")
print(f"Average XGBoost RMSE: {np.mean(rmse_list_xgb):.5f}")
print(f"Average CatBoost RMSE: {np.mean(rmse_list_cat):.5f}")

# -----------------------
# 6. Optimize Ensemble Weights Using OOF Predictions
# -----------------------
print("\nOptimizing ensemble weights using OOF predictions...")
oof_matrix = np.vstack([oof_preds_lgb, oof_preds_xgb, oof_preds_cat]).T

def ensemble_rmse(weights, predictions, true_values):
    ensemble_pred = np.dot(predictions, weights)
    return math.sqrt(mean_squared_error(true_values, ensemble_pred))

n_models = oof_matrix.shape[1]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1)] * n_models
initial_weights = np.repeat(1.0 / n_models, n_models)

result = minimize(ensemble_rmse, initial_weights, args=(oof_matrix, y.values),
                  method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

print("Optimal ensemble weights:")
print(f"LightGBM: {optimal_weights[0]:.4f}, XGBoost: {optimal_weights[1]:.4f}, CatBoost: {optimal_weights[2]:.4f}")
print(f"Ensemble OOF RMSE: {ensemble_rmse(optimal_weights, oof_matrix, y.values):.5f}")

# -----------------------
# 7. Final Model Training on Full Data & Generating Test Predictions
# -----------------------
print("\nTraining final models on full training data...")
final_lgb_rounds = int(np.mean(best_iterations) * 1.1)
print(f"Final LightGBM rounds: {final_lgb_rounds}")

dtrain_full = lgb.Dataset(X, label=y)
final_model_lgb = lgb.train(
    best_lgb_params,
    dtrain_full,
    num_boost_round=final_lgb_rounds
)

final_model_xgb = xgb.XGBRegressor(
    n_estimators=1000,
    tree_method='auto',
    objective='reg:squarederror',
    random_state=seed,
    verbosity=0,
    **best_xgb_params
)
final_model_xgb.fit(X, y, verbose=False)

final_model_cat = CatBoostRegressor(
    iterations=1000,
    eval_metric='RMSE',
    random_seed=seed,
    verbose=False,
    **best_cat_params
)
final_model_cat.fit(X, y)

print("\nGenerating predictions on test data...")
preds_test_lgb = final_model_lgb.predict(test_df, num_iteration=final_model_lgb.best_iteration)
preds_test_xgb = final_model_xgb.predict(test_df)
preds_test_cat = final_model_cat.predict(test_df)

test_pred_matrix = np.vstack([preds_test_lgb, preds_test_xgb, preds_test_cat]).T
final_test_preds = np.dot(test_pred_matrix, optimal_weights)

# -----------------------
# 8. Create Submission File
# -----------------------
submission = pd.DataFrame(final_test_preds, index=test_df.index, columns=[target_col])
submission.to_csv("playground-series-s5e2/data/submission.csv")
print("\nSubmission file saved as 'playground-series-s5e2/data/submission.csv'.")

# -----------------------
# 9. Final Summary & Timing
# -----------------------
total_time = time.time() - start_time
print(f"\nTotal training and prediction time: {total_time / 60:.2f} minutes")