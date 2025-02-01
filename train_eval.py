"""
train_eval.py

Demonstrates a time-based adaptive blending between RandomSampler and TPE:
- Early in run => higher chance of random exploration
- Later in run => higher chance of TPE exploitation
- Uses a tqdm progress bar to track time progress during optimization.

Author: Master Programmer (Kaggle champion)
"""

import time
import random
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler, RandomSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import reduce_mem_usage
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. A callback to show progress in a tqdm bar
# ----------------------------------------------------------------------
class TimeProgressCallback:
    """
    A callback that updates a tqdm progress bar after each trial,
    showing how many seconds have elapsed out of the total_time.

    Parameters
    ----------
    total_time : float
        The number of seconds we expect the study to run (timeout).
    """
    def __init__(self, total_time: float):
        self.total_time = total_time
        self.start_time = time.time()
        # Create a tqdm bar with 'total' = total_time
        # We'll treat 'n' as "elapsed seconds"
        self.pbar = tqdm(total=round(self.total_time,1), bar_format='{l_bar}{bar:30}{r_bar}')

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        elapsed = time.time() - self.start_time
        if elapsed < self.total_time:
            # Update the bar to reflect how many seconds have passed
            self.pbar.n = round(elapsed, 1)
            self.pbar.refresh()
        else:
            # We've reached or exceeded total_time, so fill the bar and close
            self.pbar.n = self.total_time
            self.pbar.refresh()
            self.pbar.close()

# ----------------------------------------------------------------------
# 2. Define a time-based AdaptiveBlendedSampler
# ----------------------------------------------------------------------
class TimeBasedAdaptiveSampler(optuna.samplers.BaseSampler):
    """
    A custom sampler that adaptively blends between RandomSampler and TPE
    based on elapsed time vs. a total_time budget.

    With probability p_rand = 1 - (fraction_done^exponent), we sample from
    RandomSampler; otherwise from TPE.

    fraction_done = min(elapsed_time / total_time, 1.0)

    Parameters
    ----------
    total_time : float
        The total time (seconds) for which the study is expected to run
        (passed as "timeout=...").
    exponent : float
        Controls how quickly p_rand decays. e.g. 1.0 => linear,
        2.0 => faster decay, 0.5 => slower decay.
    random_seed : int
        Seed for reproducibility.
    random_sampler : BaseSampler
        Sampler used for random exploration.
    tpe_sampler : BaseSampler
        Sampler used for exploitation (TPE).
    """

    def __init__(
        self,
        total_time: float,
        exponent: float = 2.0,
        random_seed: int = 42,
        random_sampler: optuna.samplers.BaseSampler = None,
        tpe_sampler: optuna.samplers.BaseSampler = None
    ):
        super().__init__()
        self.total_time = total_time
        self.exponent = exponent
        self.start_time = time.time()

        self.random_sampler = random_sampler or RandomSampler(seed=random_seed)
        self.tpe_sampler = tpe_sampler or TPESampler(seed=random_seed)

    def reseed_rng(self) -> None:
        """Reseed sub-samplers for reproducibility."""
        self.random_sampler.reseed_rng()
        self.tpe_sampler.reseed_rng()

    def _elapsed_fraction(self) -> float:
        elapsed = time.time() - self.start_time
        fraction_done = elapsed / self.total_time if self.total_time > 0 else 1.0
        return min(max(fraction_done, 0.0), 1.0)

    def _p_random(self) -> float:
        """
        Probability of choosing random sampler:
          p_rand = 1 - fraction_done^exponent
        """
        frac = self._elapsed_fraction()
        return max(0.0, 1.0 - (frac ** self.exponent))

    def infer_relative_search_space(self, study, trial):
        # We'll not do special partial search-space definitions
        # We'll rely on sample_independent for the main logic.
        return {}

    def sample_relative(self, study, trial, search_space):
        # We'll do the real sampling in sample_independent
        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        p_rand = self._p_random()
        if random.random() < p_rand:
            # exploration
            return self.random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        else:
            # exploitation
            return self.tpe_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )


# ----------------------------------------------------------------------
# 3. Objective / K-Fold / TaskType
# ----------------------------------------------------------------------
def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    task_type,
    n_splits: int = 5
) -> float:
    """
    Objective function for Optuna hyperparameter tuning.
    """
    param = {
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

    param['objective'] = task_type.objective
    param['metric'] = task_type.metric

    if task_type.num_class is not None:
        param['num_class'] = task_type.num_class
        labels = np.unique(y)
    else:
        labels = None

    Model = task_type.model
    eval_metric = task_type.eval_metric

    # K-Fold
    if task_type.objective == 'regression':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores = []
    for train_idx, valid_idx in kf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = Model(**param)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            callbacks=[lgb.early_stopping(100)]
        )

        if task_type.objective == 'binary':
            valid_pred = model.predict_proba(X_valid_fold)[:, 1]
        elif task_type.objective == 'multiclass':
            valid_pred = model.predict_proba(X_valid_fold)
        else:
            valid_pred = model.predict(X_valid_fold)

        if task_type.objective == 'multiclass':
            score = eval_metric(y_valid_fold, valid_pred, labels=labels)
        else:
            score = eval_metric(y_valid_fold, valid_pred)

        cv_scores.append(score)

    return np.mean(cv_scores)


def kfold_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    best_params: dict,
    task_type,
    n_splits: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Perform K-fold cross-validation on the full dataset, then
    predict on X_test in each fold, aggregating predictions.
    """
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
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = Model(**best_params)
        model.fit(
            X_train_fold,
            y_train_fold,
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


class TaskType():
    """
    Auto-detects classification vs. regression from y dtype
    and sets objective, metric, model, etc.
    """
    def __init__(self, y: pd.Series, fallback_metric='rmse'):
        self.determine_task_type(y, fallback_metric)

    def determine_task_type(self, y: pd.Series, fallback_metric: str):
        print(f"Detecting task type... {y.dtype}")
        if y.dtype in ['float32', 'float64']:
            print("Regression detected")
            self.objective = 'regression'
            if fallback_metric == 'mae':
                self.metric, self.eval_metric = 'l1', mean_absolute_error
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


# ----------------------------------------------------------------------
# 4. Main Script
# ----------------------------------------------------------------------
if __name__ == '__main__':

    COMPETITION_NAME = 'playground-series-s5e2'
    TARGET_COLUMN = 'Price'
    ID_COLUMN = 'id'
    PROBLEM_TYPE = 'regression'

    # Let's define how long we want to run (in seconds).
    TOTAL_TIME = 3600 / 12 # = 5 minutes

    # Paths
    DATA_DIR = os.path.join(os.getcwd(), COMPETITION_NAME, 'data')

    # Load numeric training data
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_full_train_preprocessed.csv'))
    if ID_COLUMN in X_train.columns:
        X_train.drop(columns=[ID_COLUMN], inplace=True)

    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_full_train.csv'))[TARGET_COLUMN]

    # Optional memory reduction
    X_train = reduce_mem_usage(X_train)

    # Detect task type
    task_type = TaskType(y_train, fallback_metric='rmse')

    # Set up matplotlib
    matplotlib.use('Agg')  # or 'TkAgg', 'Qt5Agg', etc.

    # Create the time-based blended sampler:
    # exponent=1 => linear decay of random probability
    # exponent=2 => random probability decays faster near the end
    sampler = TimeBasedAdaptiveSampler(
        total_time=TOTAL_TIME,
        exponent=1.0,
        random_seed=42
    )

    # Create the study, telling it to stop after TOTAL_TIME seconds
    study = optuna.create_study(direction='minimize', sampler=sampler)

    # Initialize the tqdm-based time progress callback
    time_progress_callback = TimeProgressCallback(total_time=TOTAL_TIME)

    # Run the optimization with both a timeout and the callback
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, task_type),
        timeout=TOTAL_TIME,
        callbacks=[time_progress_callback]
    )

    # Retrieve best params
    best_params = study.best_params
    best_params['verbosity'] = -1
    print("----------------------------------------")
    print("Optuna optimization finished")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best hyperparameters:", best_params)

    # Final training
    print("----------------------------------------")
    print("Training the final model on the full training set...")
    best_params['objective'] = task_type.objective
    best_params['metric'] = task_type.metric
    if task_type.num_class is not None:
        best_params['num_class'] = task_type.num_class

    final_model = task_type.model(**best_params)
    final_model.fit(X_train, y_train)

    # Plot feature importance
    lgb.plot_importance(final_model, max_num_features=30)
    plt.title("Feature importance")
    plt.show()

    # Load preprocessed test data
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test_preprocessed.csv'))
    if ID_COLUMN in X_test.columns:
        X_test.drop(columns=[ID_COLUMN], inplace=True)
    X_test = reduce_mem_usage(X_test)

    # KFold predictions
    print("Making KFold predictions on the test data...")
    test_preds = kfold_predict(X_train, y_train, X_test, best_params, task_type, n_splits=5)

    # Threshold if classification
    if task_type.objective != 'regression':
        test_preds = np.rint(test_preds).astype(int)
        # e.g., map 0 -> "False", 1 -> "True"
        # test_preds = np.where(test_preds == 0, 'False', 'True')

    # Save submission
    submission = pd.DataFrame({
        ID_COLUMN: df_test[ID_COLUMN],
        TARGET_COLUMN: test_preds
    })
    submission_file = os.path.join(DATA_DIR, 'submission.csv')
    submission.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")