# Import necessary libraries
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score
from optuna.samplers import TPESampler
from sklearn.metrics import confusion_matrix, classification_report


# Define the Optuna objective function
def objective(trial, X_train, y_train, X_val, y_val):
    # Hyperparameters to be optimized by Optuna
    param = {
        "objective": 'binary',
        "metric": 'binary_logloss',
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    # Define the dataset for LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)

    # Training the model
    gbm = lgb.train(param, dtrain, valid_sets=[lgb.Dataset(X_val, label=y_val)])

    # Making predictions on the validation set
    preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    pred_labels = np.rint(preds)

    # Calculating accuracy for the current set of hyperparameters
    accuracy = accuracy_score(y_val, pred_labels)
    return accuracy


# The main part of the script
if __name__ == '__main__':
    # Define constants
    DATA_DIR = os.path.join(os.getcwd(), 'playground-series-s3e24', 'data')
    TARGET_COLUMN = 'smoking'  # Replace with your actual target column name
    ID_COLUMN = 'id'  # Replace with your actual ID column name

    # Load training data
    train_file = os.path.join(DATA_DIR, 'X_train_preprocessed.csv')
    train_target_file = os.path.join(DATA_DIR, 'y_train.csv')
    X_train = pd.read_csv(train_file).drop(ID_COLUMN, axis=1)  # Drop ID column for training
    y_train = pd.read_csv(train_target_file)[TARGET_COLUMN]

    # Load validation data
    val_file = os.path.join(DATA_DIR, 'X_val_preprocessed.csv')
    val_target_file = os.path.join(DATA_DIR, 'y_val.csv')
    X_val = pd.read_csv(val_file).drop(ID_COLUMN, axis=1)  # Drop ID column for validation
    y_val = pd.read_csv(val_target_file)[TARGET_COLUMN]

    # Create a study object and optimize the objective function
    sampler = TPESampler(seed=42)  # Make the optimization reproducible
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=200)

    # Train the model with the best hyperparameters on the full training set
    full_train_file = os.path.join(DATA_DIR, 'X_full_train_preprocessed.csv')
    full_train_target_file = os.path.join(DATA_DIR, 'y_full_train.csv')
    X_full_train = pd.read_csv(full_train_file).drop(ID_COLUMN, axis=1)  # Drop ID column for full training
    y_full_train = pd.read_csv(full_train_target_file)[TARGET_COLUMN]
    best_params = study.best_trial.params
    full_train_set = lgb.Dataset(X_full_train, label=y_full_train)
    full_model = lgb.train(best_params, full_train_set)

    # Evaluate on the validation set (optional, for your insight)
    valid_predictions = full_model.predict(X_val)
    valid_pred_labels = (valid_predictions >= 0.5).astype(int)
    accuracy = accuracy_score(y_val, valid_pred_labels)
    roc_auc = roc_auc_score(y_val, valid_predictions)
    conf_matrix = confusion_matrix(y_val, valid_pred_labels)
    class_report = classification_report(y_val, valid_pred_labels)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f"Confusion matrix:\n{conf_matrix}")
    print(f"Classification report:\n{class_report}")

    # Make predictions on the test set
    test_file = os.path.join(DATA_DIR, 'X_test_preprocessed.csv')
    X_test = pd.read_csv(test_file)
    test_ids = X_test[ID_COLUMN]  # Preserve ID column for submission
    X_test = X_test.drop(ID_COLUMN, axis=1)  # Drop ID column for prediction
    test_predictions = full_model.predict(X_test)

    # Prepare submission
    submission = pd.DataFrame({ID_COLUMN: test_ids, TARGET_COLUMN: test_predictions})
    submission_file = os.path.join(DATA_DIR, 'submission.csv')
    submission.to_csv(submission_file, index=False)

    print(f"Submission file saved to {submission_file}")