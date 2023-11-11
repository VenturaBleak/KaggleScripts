# Import necessary libraries
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt
import seaborn as sns


# Adjusted objective function for different task types
def objective(trial, X, y, task_type, n_splits=5):
    param = {
        'verbosity': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
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
    Model = task_type.model
    eval_metric = task_type.eval_metric

    # Cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task_type.objective != 'regression' else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, valid_index in kf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]
        y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]
        model = Model(**param)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)], callbacks=[lgb.early_stopping(100)])
        valid_pred = model.predict_proba(X_valid_fold)[:, 1] if task_type.objective != 'regression' else model.predict(X_valid_fold)
        cv_score = eval_metric(y_valid_fold, valid_pred)
        cv_scores.append(cv_score)
    return np.mean(cv_scores)

# Function to perform k-fold cross-validation and return test predictions
# Adjusted k-fold prediction function
def kfold_predict(X, y, X_test, best_params, task_type, n_splits=5, seed=42):
    best_params['objective'] = task_type.objective
    best_params['metric'] = task_type.metric
    if task_type.num_class is not None:
        best_params['num_class'] = task_type.num_class
    Model = task_type.model

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) if task_type.objective != 'regression' else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    test_preds = np.zeros(X_test.shape[0])
    for train_index, valid_index in kf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]
        y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]
        model = Model(**best_params)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)], callbacks=[lgb.early_stopping(100)])
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits if task_type.objective != 'regression' else model.predict(X_test) / n_splits
    return test_preds

class TaskType():
    def __init__(self, y):
        self.determine_task_type(y)

    # Function to determine the type of task
    def determine_task_type(self, y):
        if y.dtype == 'float':
            self.objective = 'regression'
            self.metric = 'mean_absolute_error'
            self.eval_metric = mean_absolute_error
            self.model = lgb.LGBMRegressor
            self.num_class = None
        elif y.dtype == 'int':
            if len(np.unique(y)) == 2:
                print("Binary classification detected")
                self.objective = 'binary'
                self.metric = 'roc_auc_score'
                self.eval_metric = roc_auc_score
                self.model = lgb.LGBMClassifier
                self.num_class = None
            else:
                print("Multiclass classification detected")
                self.objective = 'multiclass'
                self.metric = 'log_loss'
                self.eval_metric = log_loss
                self.model = lgb.LGBMClassifier
                self.num_class = len(np.unique(y))

# The main part of the script
if __name__ == '__main__':
    # competition specific, change as needed
    COMPETITION_NAME = 'playground-series-s3e9'
    TARGET_COLUMN = 'Strength'
    ID_COLUMN = 'id'

    OPTIMIZATION_TIME = 3600 / 24 # 3600 seconds = 1 hour

    # Define constants
    DATA_DIR = os.path.join(os.getcwd(), COMPETITION_NAME, 'data')

    # Load training data
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_full_train_preprocessed.csv'))
    X_train = X_train.drop([ID_COLUMN], axis=1)
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_full_train.csv'))[TARGET_COLUMN]

    # Determine task type
    TaskType = TaskType(y_train)

    # Load test data
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test_preprocessed.csv'))
    X_test = X_test.drop([ID_COLUMN], axis=1)

    # reduce memory usage
    from data_preprocessing import reduce_mem_usage
    X_train = reduce_mem_usage(X_train)
    X_test = reduce_mem_usage(X_test)

    # Optuna optimization
    sampler = TPESampler(seed=42)
    if TaskType.objective == 'binary':
        # maximize roc_auc_score for binary classification
        study = optuna.create_study(direction='maximize', sampler=sampler)
    else:
        # minimize log_loss for multiclass; minimize mean_absolute_error for regression
        study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train, task_type=TaskType), timeout=OPTIMIZATION_TIME) # 3600 seconds = 1 hour

    # Output the best hyperparameters
    best_params = study.best_params
    best_params['verbosity'] = -1  # Add verbosity parameter
    print("Best hyperparameters:", best_params)

    # Train the final model on the full training set
    best_params['objective'] = TaskType.objective
    best_params['metric'] = TaskType.metric
    if TaskType.num_class is not None:
        best_params['num_class'] = TaskType.num_class
    final_model = TaskType.model(**best_params)
    final_model.fit(X_train, y_train)

    # Feature importance visualization
    lgb.plot_importance(final_model, max_num_features=30)
    plt.title("Feature importance")
    plt.show()

    # Use best hyperparameters to train the model on the full training dataset and make predictions
    print("Training model on the full training set with best parameters found and making predictions...")
    test_predictions = kfold_predict(X_train, y_train, X_test, best_params, task_type=TaskType,n_splits=5)

    # Prepare submission
    submission = pd.DataFrame({
        ID_COLUMN: df_test[ID_COLUMN],
        TARGET_COLUMN: test_predictions
    })
    submission_file = os.path.join(DATA_DIR, 'submission.csv')
    submission.to_csv(submission_file, index=False)

    print(f"Submission file saved to {submission_file}")