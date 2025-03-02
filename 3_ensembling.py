#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble Script for Competition playground-series-s5e2

This script loads previously saved OOF and test predictions from the training phase
(from the folder "playground-series-s5e2/predictions"), then optimizes ensemble blending by
finding the best weights (using the true targets from the training data) and saves the final
ensemble submission in the "playground-series-s5e2/ensemble" folder.

Folder structure:
  playground-series-s5e2/
      ├── data/
      │      train_preprocessed.csv   <- Training data with true target "Price"
      ├── predictions/
      │      oof_lgb.csv, oof_xgb.csv, oof_cat.csv
      │      test_preds_lgb.csv, test_preds_xgb.csv, test_preds_cat.csv
      └── ensemble/
             ensemble_submission1.csv, etc.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math

# --- Constants ---
TARGET = "Price"
ID = "id"
COMPETITION_NAME = "playground-series-s5e2"
PREDICTIONS_DIR = os.path.join(COMPETITION_NAME, "predictions")
ENSEMBLE_DIR = os.path.join(COMPETITION_NAME, "ensemble")
# Default true target file (from training data)
DEFAULT_TRUE_FILE = os.path.join(COMPETITION_NAME, "data", "train_preprocessed.csv")


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")


def load_prediction(file_path, default_id_start=0):
    """
    Loads a CSV file of predictions.
    If the expected "id" or TARGET columns are missing, it adds an "id" column (starting at default_id_start)
    and renames the single column to TARGET.
    Returns a DataFrame with columns [id, Price] sorted by id.
    """
    df = pd.read_csv(file_path)
    if ID not in df.columns:
        # Create an id column using row numbers starting from default_id_start.
        df.insert(0, ID, range(default_id_start, default_id_start + len(df)))
    if TARGET not in df.columns:
        cols = df.columns.tolist()
        # If there is exactly one column besides the id column, rename it to TARGET.
        if len(cols) == 2:
            df.rename(columns={cols[1]: TARGET}, inplace=True)
        else:
            raise ValueError(f"{file_path} is missing required columns.")
    return df[[ID, TARGET]].sort_values(by=ID).reset_index(drop=True)


def load_all_oof():
    """Loads all OOF prediction files from PREDICTIONS_DIR using default id starting at 0."""
    files = ['oof_lgb.csv', 'oof_xgb.csv', 'oof_cat.csv']
    dfs = []
    for f in files:
        fp = os.path.join(PREDICTIONS_DIR, f)
        dfs.append(load_prediction(fp, default_id_start=0))
    return dfs


def load_all_test_preds():
    """Loads all test prediction files from PREDICTIONS_DIR using default id starting at 300000."""
    files = ['test_preds_lgb.csv', 'test_preds_xgb.csv', 'test_preds_cat.csv']
    dfs = []
    for f in files:
        fp = os.path.join(PREDICTIONS_DIR, f)
        dfs.append(load_prediction(fp, default_id_start=300000))
    return dfs


def optimize_ensemble(oof_matrix, y_true):
    """
    Given OOF predictions (n_samples x n_models) and true targets,
    finds the optimal weights (which sum to 1) that minimize the RMSE.
    """

    def ensemble_rmse(weights, predictions, true_values):
        ensemble_pred = np.dot(predictions, weights)
        return math.sqrt(np.mean((true_values - ensemble_pred) ** 2))

    n_models = oof_matrix.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n_models
    init_weights = np.full(n_models, 1.0 / n_models)
    result = minimize(ensemble_rmse, init_weights, args=(oof_matrix, y_true),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def next_ensemble_filename():
    """Determines the next available ensemble submission filename in the ENSEMBLE_DIR."""
    ensure_folder(ENSEMBLE_DIR)
    files = [f for f in os.listdir(ENSEMBLE_DIR) if f.startswith("ensemble_submission") and f.endswith(".csv")]
    numbers = []
    for f in files:
        num_part = f.replace("ensemble_submission", "").replace(".csv", "")
        if num_part.isdigit():
            numbers.append(int(num_part))
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(ENSEMBLE_DIR, f"ensemble_submission{next_num}.csv")


def main():
    ensure_folder(PREDICTIONS_DIR)
    ensure_folder(ENSEMBLE_DIR)

    # Load and merge OOF predictions from training.
    oof_dfs = load_all_oof()
    merged = oof_dfs[0].copy()
    merged = merged.rename(columns={TARGET: "pred_0"})
    for i, df in enumerate(oof_dfs[1:], start=1):
        merged[f"pred_{i}"] = df[TARGET].values
    oof_matrix = merged[[col for col in merged.columns if col.startswith("pred_")]].values

    # Load the true targets (from training). Use a default path so you don't have to type it.
    true_file = input(
        f"Enter the path to the CSV file with the true OOF target values (default: {DEFAULT_TRUE_FILE}): ").strip()
    if not true_file:
        true_file = DEFAULT_TRUE_FILE
    true_df = pd.read_csv(true_file)
    if ID not in true_df.columns or TARGET not in true_df.columns:
        if TARGET in true_df.columns:
            true_df.insert(0, ID, range(len(true_df)))
        else:
            raise ValueError("True target file is missing required columns.")
    true_df = true_df[[ID, TARGET]].sort_values(by=ID).reset_index(drop=True)
    y_true = true_df[TARGET].values

    # Optimize ensemble weights using OOF predictions and true targets.
    optimal_weights = optimize_ensemble(oof_matrix, y_true)
    print("Optimized ensemble weights:", optimal_weights)

    # Load and merge test predictions (from the test set).
    test_dfs = load_all_test_preds()
    merged_test = test_dfs[0].copy()
    merged_test = merged_test.rename(columns={TARGET: "pred_0"})
    for i, df in enumerate(test_dfs[1:], start=1):
        merged_test[f"pred_{i}"] = df[TARGET].values
    pred_cols = [col for col in merged_test.columns if col.startswith("pred_")]
    test_matrix = merged_test[pred_cols].values

    # Compute final ensemble predictions on the test set.
    final_ensemble_pred = np.dot(test_matrix, optimal_weights)
    ensemble_df = merged_test[[ID]].copy()
    ensemble_df[TARGET] = final_ensemble_pred
    avg_ensemble_pred = ensemble_df[TARGET].mean()
    print(f"Average prediction value of ensemble on test set: {avg_ensemble_pred:.6f}")

    # Save the ensemble submission.
    output_file = next_ensemble_filename()
    ensemble_df.to_csv(output_file, index=False)
    print(f"Final ensemble submission saved as: {output_file}")


if __name__ == "__main__":
    main()