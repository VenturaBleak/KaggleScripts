"""
train_eval.py

Demonstrates a time-based adaptive blending between RandomSampler and TPE,
and trains a model using either the extended competition data (with an underlying meta-feature)
or the base preprocessed competition data.
A flag (USE_EXTENDED_DATA) lets the user choose whether to include the underlying meta-feature.
"""

import time
import random
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler, RandomSampler
from sklearn.metrics import mean_squared_error, log_loss, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_preprocessing import reduce_mem_usage

# --------------------------
# Prompt user once to choose dataset type
# --------------------------
use_ext_input = input("Do you want to use extended competition data with the underlying meta-feature? (y/n): ")
USE_EXTENDED_DATA = use_ext_input.strip().lower().startswith("y")
print(f"USE_EXTENDED_DATA set to {USE_EXTENDED_DATA}")

# --------------------------
# Paths and Constants
# --------------------------
DATA_DIR = os.path.join(os.getcwd(), 'playground-series-s5e2', 'data')
TARGET_COLUMN = 'Price'
ID_COLUMN = 'id'

# --------------------------
# Helper function to load dataset
# --------------------------
def load_dataset(file_base, use_extended):
    extended_path = os.path.join(DATA_DIR, f"{file_base}_extended.csv")
    base_path = os.path.join(DATA_DIR, f"{file_base}_preprocessed.csv")
    if use_extended and os.path.exists(extended_path):
        print(f"Loading extended {file_base} data...")
        return pd.read_csv(extended_path)
    else:
        print(f"Loading base {file_base} data...")
        return pd.read_csv(base_path)

# --------------------------
# Load Competition Data
# --------------------------
X_full_train = load_dataset("X_full_train", USE_EXTENDED_DATA)
X_val = load_dataset("X_val", USE_EXTENDED_DATA)
X_test = load_dataset("X_test", USE_EXTENDED_DATA)
y_full_train = pd.read_csv(os.path.join(DATA_DIR, "y_full_train.csv"))[TARGET_COLUMN]

# Optional memory reduction
X_full_train = reduce_mem_usage(X_full_train)
X_val = reduce_mem_usage(X_val)
X_test = reduce_mem_usage(X_test)

# --------------------------
# Define Helper Classes & Functions for Hyperparameter Tuning
# --------------------------
class TimeProgressCallback:
    """A callback that updates a tqdm progress bar after each trial."""
    def __init__(self, total_time: float):
        self.total_time = total_time
        self.start_time = time.time()
        self.pbar = tqdm(total=round(self.total_time, 1), bar_format='{l_bar}{bar:30}{r_bar}')

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        elapsed = time.time() - self.start_time
        if elapsed < self.total_time:
            self.pbar.n = round(elapsed, 1)
            self.pbar.refresh()
        else:
            self.pbar.n = self.total_time
            self.pbar.refresh()
            self.pbar.close()

class TimeBasedAdaptiveSampler(optuna.samplers.BaseSampler):
    """A sampler that adaptively blends between RandomSampler and TPE based on elapsed time."""
    def __init__(self, total_time: float, exponent: float = 2.0, random_seed: int = 42,
                 random_sampler: optuna.samplers.BaseSampler = None, tpe_sampler: optuna.samplers.BaseSampler = None):
        super().__init__()
        self.total_time = total_time
        self.exponent = exponent
        self.start_time = time.time()
        self.random_sampler = random_sampler or RandomSampler(seed=random_seed)
        self.tpe_sampler = tpe_sampler or TPESampler(seed=random_seed)

    def reseed_rng(self) -> None:
        self.random_sampler.reseed_rng()
        self.tpe_sampler.reseed_rng()

    def _elapsed_fraction(self) -> float:
        elapsed = time.time() - self.start_time
        fraction_done = elapsed / self.total_time if self.total_time > 0 else 1.0
        return min(max(fraction_done, 0.0), 1.0)

    def _p_random(self) -> float:
        frac = self._elapsed_fraction()
        return max(0.0, 1.0 - (frac ** self.exponent))

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        p_rand = self._p_random()
        if random.random() < p_rand:
            return self.random_sampler.sample_independent(study, trial, param_name, param_distribution)
        else:
            return self.tpe_sampler.sample_independent(study, trial, param_name, param_distribution)

class TaskType():
    """Auto-detects classification vs. regression and sets objective, metric, model, etc."""
    def __init__(self, y: pd.Series, fallback_metric='rmse'):
        self.determine_task_type(y, fallback_metric)

    def determine_task_type(self, y: pd.Series, fallback_metric: str):
        print(f"Detecting task type... {y.dtype}")
        if y.dtype in ['float32', 'float64']:
            print("Regression detected")
            self.objective = 'regression'
            if fallback_metric == 'mae':
                self.metric, self.eval_metric = 'l1', lambda yt, yp: np.mean(np.abs(yt - yp))
            else:
                self.metric, self.eval_metric = 'l2', lambda yt, yp: mean_squared_error(yt, yp)
            self.model = lgb.LGBMRegressor
            self.num_class = None
        else:
            unique_labels = np.unique(y)
            if len(unique_labels) == 2:
                print("Binary classification detected")
                self.objective = 'binary'
                self.metric = 'binary_logloss'
                self.eval_metric = log_loss
                self.model = lgb.LGBMClassifier
                self.num_class = None
            else:
                print("Multiclass classification detected")
                self.objective = 'multiclass'
                self.metric = 'multi_logloss'
                self.eval_metric = log_loss
                self.model = lgb.LGBMClassifier
                self.num_class = len(unique_labels)

def objective(trial, X, y, task_type, n_splits=5):
    params = {
        'verbosity': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 5, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 64),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
        'cat_l2': trial.suggest_int('cat_l2', 1, 20),
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }
    params['objective'] = task_type.objective
    params['metric'] = task_type.metric

    if task_type.num_class is not None:
        params['num_class'] = task_type.num_class

    Model = task_type.model
    cv_scores = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, valid_idx in kf.split(X, y):
        X_train_cv = X.iloc[train_idx]
        y_train_cv = y.iloc[train_idx]
        X_val_cv = X.iloc[valid_idx]
        y_val_cv = y.iloc[valid_idx]
        model = Model(**params)
        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            callbacks=[lgb.early_stopping(100)]
        )
        preds = model.predict(X_val_cv)
        score = np.sqrt(mean_squared_error(y_val_cv, preds))
        cv_scores.append(score)
    return np.mean(cv_scores)

def kfold_predict(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, best_params: dict, task_type, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    best_params['objective'] = task_type.objective
    best_params['metric'] = task_type.metric
    if task_type.num_class is not None:
        best_params['num_class'] = task_type.num_class
    Model = task_type.model
    if task_type.objective == 'regression':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    test_preds = np.zeros(X_test.shape[0], dtype=np.float64)
    for train_idx, valid_idx in kf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]
        model = Model(**best_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            callbacks=[lgb.early_stopping(100)]
        )
        if task_type.objective == 'binary':
            test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        elif task_type.objective == 'multiclass':
            test_preds += model.predict_proba(X_test) / n_splits
        else:
            test_preds += model.predict(X_test) / n_splits
    return test_preds

# --------------------------
# Main Script
# --------------------------
if __name__ == '__main__':
    DATA_DIR = os.path.join(os.getcwd(), 'playground-series-s5e2', 'data')

    # Load competition data using the flag (prompted only once at the beginning)
    def load_data(file_base, use_extended):
        extended_path = os.path.join(DATA_DIR, f"{file_base}_extended.csv")
        base_path = os.path.join(DATA_DIR, f"{file_base}_preprocessed.csv")
        if use_extended and os.path.exists(extended_path):
            print(f"Loading extended {file_base} data...")
            return pd.read_csv(extended_path)
        else:
            print(f"Loading base {file_base} data...")
            return pd.read_csv(base_path)

    X_full_train = load_data("X_full_train", USE_EXTENDED_DATA)
    X_val = load_data("X_val", USE_EXTENDED_DATA)
    X_test = load_data("X_test", USE_EXTENDED_DATA)
    y_full_train = pd.read_csv(os.path.join(DATA_DIR, "y_full_train.csv"))[TARGET_COLUMN]

    # Optional memory reduction
    X_full_train = reduce_mem_usage(X_full_train)
    X_val = reduce_mem_usage(X_val)
    X_test = reduce_mem_usage(X_test)

    # Detect task type
    task_type = TaskType(y_full_train, fallback_metric='rmse')

    # Set up matplotlib backend
    matplotlib.use('Agg')

    # Define timeout (in seconds) for the Optuna study
    TOTAL_TIME = 3600 / 2

    # Create time-based adaptive sampler and study
    sampler = TimeBasedAdaptiveSampler(total_time=TOTAL_TIME, exponent=1.0, random_seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    time_progress_callback = TimeProgressCallback(total_time=TOTAL_TIME)
    study.optimize(
        lambda trial: objective(trial, X_full_train, y_full_train, task_type),
        timeout=TOTAL_TIME,
        callbacks=[time_progress_callback]
    )

    best_params = study.best_params
    best_params['verbosity'] = -1
    print("----------------------------------------")
    print("Optuna optimization finished")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best hyperparameters:", best_params)

    # Train final model on full competition training data
    print("----------------------------------------")
    print("Training final model on full competition training data...")
    best_params['objective'] = task_type.objective
    best_params['metric'] = task_type.metric
    if task_type.num_class is not None:
        best_params['num_class'] = task_type.num_class

    final_model = task_type.model(**best_params)
    final_model.fit(X_full_train, y_full_train)

    # Plot feature importance
    lgb.plot_importance(final_model, max_num_features=30)
    plt.title("Feature importance")
    plt.show()

    # KFold predictions on the test data
    print("Making KFold predictions on the test data...")
    test_preds = kfold_predict(X_full_train, y_full_train, X_test, best_params, task_type, n_splits=5)
    if task_type.objective != 'regression':
        test_preds = np.rint(test_preds).astype(int)

    # Load original test file for IDs and create submission
    df_test_orig = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    submission = pd.DataFrame({
        ID_COLUMN: df_test_orig[ID_COLUMN],
        TARGET_COLUMN: test_preds
    })
    submission_file = os.path.join(DATA_DIR, 'submission.csv')
    submission.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")