# Import necessary libraries
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Optuna objective function
def objective(trial, X_train, y_train, X_val, y_val):
    N_SPLITS = 5
    param_grid = {
    # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
    "max_depth": trial.suggest_int("max_depth", 3, 12),
    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
    "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
    "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
    "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    "bagging_fraction": trial.suggest_float(
        "bagging_fraction", 0.2, 0.95, step=0.1
    ),
    "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
    "feature_fraction": trial.suggest_float(
        "feature_fraction", 0.2, 0.95, step=0.1
    ),
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    cv_scores = np.empty(N_SPLITS)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgbm.LGBMClassifier(objective="binary", **param_grid)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                lgb.LightGBMPruningCallback(trial, "binary_logloss")
            ],  # Add a pruning callback
        )

        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)

# Function to perform k-fold cross-validation on the full training set
def k_fold_cross_validation(X, y, best_params, n_splits=10, seed=42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = []
    fold_aucs = []

    for train_index, valid_index in kf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]
        y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]

        dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
        dvalid = lgb.Dataset(X_valid_fold, label=y_valid_fold)

        model = lgb.train(
            best_params,
            dtrain,
            valid_sets=[dvalid]
        )

        valid_pred = model.predict(X_valid_fold)
        fold_auc = roc_auc_score(y_valid_fold, valid_pred)
        fold_aucs.append(fold_auc)
        models.append(model)

    print(f'Mean ROC AUC for {n_splits}-fold CV: {np.mean(fold_aucs)}')
    return models


# Function to train a logistic regression classifier
def train_and_evaluate_linear_classifier(X_train, y_train, X_val, y_val, features):
    print("---------------------------------")
    print("Training logistic regression model...")
    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')

    # Fit the model on the training set
    model.fit(X_train[features], y_train)

    # Evaluate the model
    val_predictions_proba = model.predict_proba(X_val[features])[:, 1]
    val_predictions = model.predict(X_val[features])
    val_auc = roc_auc_score(y_val, val_predictions_proba)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f'Validation AUC: {val_auc:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    return model, val_predictions_proba


# Function to train and evaluate a LightGBM classifier
def train_and_evaluate_lgbm_classifier(X_train, y_train, X_val, y_val, best_params):
    print("---------------------------------")
    print("Training LightGBM model...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        best_params,
        dtrain,
        valid_sets=[dvalid]
    )

    val_predictions = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = roc_auc_score(y_val, val_predictions)
    val_accuracy = accuracy_score(y_val, np.rint(val_predictions))

    print(f'LightGBM Validation AUC: {val_auc:.4f}')
    print(f'LightGBM Validation Accuracy: {val_accuracy:.4f}')

    return model, val_predictions


# Function to visualize feature importance for the final LightGBM model
def visualize_feature_importance(model, features):
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(importance_type='gain'), features)),
                               columns=['Value', 'Feature'])
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()


# The main part of the script
if __name__ == '__main__':
    COMPETITION_NAME = 'tabular-playground-series-sep-2021'
    TARGET_COLUMN = 'claim'
    ID_COLUMN = 'id'  # Replace with your actual ID column name

    # Define constants
    DATA_DIR = os.path.join(os.getcwd(), COMPETITION_NAME, 'data')

    ######################## Find the best hyperparameters ########################
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
    sampler = TPESampler(seed=1)  # Make the optimization reproducible
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=5)

    # Retrieve the best parameters from the study
    best_params = study.best_params

    # replace verbosity parameter with -1
    best_params['verbosity'] = -1

    ######################## Evaluate against baseline ########################
    # Evaluate against baseline
    # Train and evaluate a logistic regression model as a baseline
    print("Training and evaluating a logistic regression model as a baseline...")
    linear_model, linear_val_predictions = train_and_evaluate_linear_classifier(
        X_train, y_train, X_val, y_val, X_train.columns
    )

    # Train and evaluate a LightGBM model with the best hyperparameters
    print("Training and evaluating a LightGBM model with the best hyperparameters...")
    final_lgbm_model, lgbm_val_predictions = train_and_evaluate_lgbm_classifier(
        X_train, y_train, X_val, y_val, best_params
    )

    ######################## Train the final model ########################
    # Load the full training data (which does not include the validation set)
    full_train_file = os.path.join(DATA_DIR, 'X_full_train_preprocessed.csv')
    full_train_target_file = os.path.join(DATA_DIR, 'y_full_train.csv')
    X_full_train = pd.read_csv(full_train_file).drop(ID_COLUMN, axis=1)
    y_full_train = pd.read_csv(full_train_target_file)[TARGET_COLUMN]

    # Perform k-fold cross-validation on the full training set
    models = k_fold_cross_validation(X_full_train, y_full_train, best_params)

    # Select the best model or ensemble them as needed
    # For simplicity, let's use the first model as our final model
    final_model = models[0]

    # Visualize feature importance for the final LightGBM model
    print("Visualizing feature importance for the final LightGBM model...")
    visualize_feature_importance(final_model, X_full_train.columns)

    ######################## Make predictions on the test set and save submission file ########################
    # Make predictions on the test set
    test_file = os.path.join(DATA_DIR, 'X_test_preprocessed.csv')
    X_test = pd.read_csv(test_file)
    test_ids = X_test[ID_COLUMN]
    X_test = X_test.drop(ID_COLUMN, axis=1)
    test_predictions = final_model.predict(X_test)

    # Prepare submission
    submission = pd.DataFrame({ID_COLUMN: test_ids, TARGET_COLUMN: test_predictions})
    submission_file = os.path.join(DATA_DIR, 'submission.csv')
    submission.to_csv(submission_file, index=False)

    print(f"Submission file saved to {submission_file}")