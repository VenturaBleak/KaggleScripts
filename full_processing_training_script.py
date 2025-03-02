#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Competition: playground-series-s5e3
Folder structure:
  COMPETITION_NAME = "playground-series-s5e3"
  DATA_DIR = os.path.join(COMPETITION_NAME, "data")
  MODELS_DIR = os.path.join(COMPETITION_NAME, "models")
  PREDICTIONS_DIR = os.path.join(COMPETITION_NAME, "predictions")
  SUBMISSIONS_DIR = os.path.join(COMPETITION_NAME, "submissions")

This script:
  - Loads synthetic data (train.csv) and test data, and also loads the original dataset (Rainfall.csv).
  - The original dataset is cleaned and a column "synthetic_indicator" is added (0 for original, 1 for synthetic).
  - A fixed holdout (H) is taken from the synthetic data.
  - Two candidate training sets are formed:
      Candidate A: synthetic-only (R = synthetic_train - holdout)
      Candidate B: merge(R with original)
  - A simple stacking classifier is trained on each candidate (on R only) and evaluated on the same holdout H.
  - The candidate with the higher ROC AUC on H is chosen.
  - Then, for final training, the full candidate is used:
      If B is chosen, merge the entire synthetic_train with original;
      Otherwise, use synthetic_train only.
  - Feature engineering is applied, and then the rest of the pipeline (Optuna tuning, final training, stacking ensemble) is executed.
  - The test dataset is marked as synthetic.
The goal is to optimize ROC AUC on a binary classification task.
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

# New imports for neural network and simple comparison classifier
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier as SimpleStackingClassifier

import warnings

warnings.filterwarnings("ignore", message=r".*use_label_encoder.*")

# Define folder structure
COMPETITION_NAME = "playground-series-s5e3"
DATA_DIR = os.path.join(COMPETITION_NAME, "data")
MODELS_DIR = os.path.join(COMPETITION_NAME, "models")
PREDICTIONS_DIR = os.path.join(COMPETITION_NAME, "predictions")
SUBMISSIONS_DIR = os.path.join(COMPETITION_NAME, "submissions")
for folder in [DATA_DIR, MODELS_DIR, PREDICTIONS_DIR, SUBMISSIONS_DIR]:
    os.makedirs(folder, exist_ok=True)

RANDOM_STATE = 42


# -------------------------
# Data Loading Functions
# -------------------------
def load_data():
    """Load synthetic train.csv and test.csv from DATA_DIR."""
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


# -------------------------
# Preprocessing and Extended Feature Engineering
# -------------------------
def feature_engineering(df):
    # Convert 'day' to datetime (errors coerced)
    df['day'] = pd.to_datetime(df['day'], errors='coerce')

    # Extract temporal features
    df['month'] = df['day'].dt.month
    df['day_of_week'] = df['day'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Temperature features
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['avg_temp'] = (df['maxtemp'] + df['mintemp']) / 2
    df['temp_deviation'] = df['temparature'] - df['avg_temp']

    # Dew point depression
    df['dew_point_depression'] = df['temparature'] - df['dewpoint']

    # Wind direction - sine and cosine transformation
    df['wind_dir_rad'] = np.deg2rad(df['winddirection'])
    df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
    df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
    df.drop(columns=['wind_dir_rad'], inplace=True)

    # Wind chill factor (simplified version)
    df['wind_chill'] = (13.12 + 0.6215 * df['temparature']
                        - 11.37 * (df['windspeed'] ** 0.16)
                        + 0.3965 * df['temparature'] * (df['windspeed'] ** 0.16))

    # Interaction features
    df['humidity_temp'] = df['humidity'] * df['temparature']
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']

    # Rolling statistical features (using a window of 7)
    df['rolling_temp_mean'] = df['avg_temp'].rolling(window=7).mean()
    df['rolling_wind_mean'] = df['windspeed'].rolling(window=7).mean()
    df['rolling_humidity_mean'] = df['humidity'].rolling(window=7).mean()

    # Lag features (lag 1)
    df['temp_lag_1'] = df['avg_temp'].shift(1)
    df['humidity_lag_1'] = df['humidity'].shift(1)
    df['windspeed_lag_1'] = df['windspeed'].shift(1)

    # Interaction features between pressure, windspeed, and sunshine
    df['pressure_temp_interaction'] = df['pressure'] * df['avg_temp']
    df['windspeed_temp_interaction'] = df['windspeed'] * df['avg_temp']
    df['sunshine_cloud_interaction'] = df['sunshine'] * df['cloud']

    # Season feature based on month
    df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
    'Summer' if 6 <= x <= 8 else
    'Autumn' if 9 <= x <= 11 else 'Winter')

    # Create lag and difference features for select columns
    for c in ['pressure', 'maxtemp', 'temparature', 'humidity']:
        for gap in [1]:
            df[f"{c}_shift{gap}"] = df[c].shift(gap)
            df[f"{c}_diff{gap}"] = df[c].diff(gap)

    # Binary encoding for season (drop first to avoid multicollinearity)
    df = pd.get_dummies(df, columns=['season'], drop_first=True)

    # Drop original 'day' column as it has been processed
    df.drop(columns=['day'], inplace=True)

    return df


def manual_preprocess(X_train, X_val):
    """Fill NA with 0 and scale using StandardScaler."""
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    return X_train_scaled, X_val_scaled, sc


def missing_values_summary(df):
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().mean() * 100).round(2)
    return pd.DataFrame({"Missing Values": missing_values, "Missing (%)": missing_percentage})


def get_next_submission_number():
    submission_files = os.listdir(SUBMISSIONS_DIR)
    numbers = []
    for filename in submission_files:
        match = re.match(r"submission(\d+)\.csv", filename)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 1


# -------------------------
# Purged Cross-Validation (if needed in future enhancements)
# -------------------------
def purged_cross_validation(X, y, n_splits=5, purge_length=1):
    from sklearn.model_selection import TimeSeriesSplit
    ts_split = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in ts_split.split(X):
        val_idx = val_idx[val_idx >= train_idx[-purge_length]]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        yield X_train, X_val, y_train, y_val


# -------------------------
# OPTUNA OBJECTIVE FUNCTIONS (unchanged)
# -------------------------
def objective_lgb(trial, X, y, cv_mode="stratified"):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'num_estimators': trial.suggest_int('num_estimators', 100, 5000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 5e-4, 1e-1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]
        X_train_scaled, X_valid_scaled, _ = manual_preprocess(X_train_cv, X_valid_cv)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_scaled, y_train_cv,
                  eval_set=[(X_valid_scaled, y_valid_cv)],
                  eval_metric='auc',
                  callbacks=[LightGBMPruningCallback(trial, "auc")])
        y_pred = model.predict_proba(X_valid_scaled)[:, 1]
        auc_scores.append(roc_auc_score(y_valid_cv, y_pred))
    return np.mean(auc_scores)


def objective_rf(trial, X, y, cv_mode="stratified"):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]
        X_train_scaled, X_valid_scaled, _ = manual_preprocess(X_train_cv, X_valid_cv)
        model = RandomForestClassifier(random_state=RANDOM_STATE, **params)
        model.fit(X_train_scaled, y_train_cv)
        y_pred = model.predict_proba(X_valid_scaled)[:, 1]
        auc_scores.append(roc_auc_score(y_valid_cv, y_pred))
    return np.mean(auc_scores)


def objective_xgb(trial, X, y, cv_mode="stratified"):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]
        X_train_scaled, X_valid_scaled, _ = manual_preprocess(X_train_cv, X_valid_cv)
        best_params = params.copy()
        model = XGBClassifier(use_label_encoder=True, eval_metric='auc', random_state=RANDOM_STATE, **best_params)
        model.fit(X_train_scaled, y_train_cv)
        y_pred = model.predict_proba(X_valid_scaled)[:, 1]
        auc_scores.append(roc_auc_score(y_valid_cv, y_pred))
    return np.mean(auc_scores)


def objective_cat(trial, X, y, cv_mode="stratified"):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
    }
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]
        X_train_scaled, X_valid_scaled, _ = manual_preprocess(X_train_cv, X_valid_cv)
        model = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, **params)
        model.fit(X_train_scaled, y_train_cv)
        y_pred = model.predict_proba(X_valid_scaled)[:, 1]
        auc_scores.append(roc_auc_score(y_valid_cv, y_pred))
    return np.mean(auc_scores)


# -------------------------
# FINAL MODEL TRAINING FUNCTIONS (unchanged)
# -------------------------
def train_final_model_lgb(X, y, best_params):
    sc = StandardScaler()
    X = X.fillna(0)
    X_scaled = sc.fit_transform(X)
    best_params.pop('random_state', None)
    model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_STATE)
    model.fit(X_scaled, y)
    return model, sc


def train_final_model_rf(X, y, best_params):
    sc = StandardScaler()
    X = X.fillna(0)
    X_scaled = sc.fit_transform(X)
    from sklearn.ensemble import RandomForestClassifier
    best_params.pop('random_state', None)
    model = RandomForestClassifier(random_state=RANDOM_STATE, **best_params)
    model.fit(X_scaled, y)
    return model, sc


def train_final_model_xgb(X, y, best_params):
    sc = StandardScaler()
    X = X.fillna(0)
    X_scaled = sc.fit_transform(X)
    from xgboost import XGBClassifier
    best_params.pop('random_state', None)
    model = XGBClassifier(use_label_encoder=True, eval_metric='auc', random_state=RANDOM_STATE, **best_params)
    model.fit(X_scaled, y)
    return model, sc


def train_final_model_cat(X, y, best_params):
    sc = StandardScaler()
    X = X.fillna(0)
    X_scaled = sc.fit_transform(X)
    from catboost import CatBoostClassifier
    best_params.pop('random_state', None)
    model = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, **best_params)
    model.fit(X_scaled, y)
    return model, sc


# -------------------------
# MODEL SAVING FUNCTIONS (unchanged)
# -------------------------
def save_model(model, submission_number):
    filename = f"final_model{submission_number}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def save_submission(ids, predictions, id_col, target_col, submission_number):
    submission = pd.DataFrame({id_col: ids, target_col: predictions})
    filename = f"submission{submission_number}.csv"
    filepath = os.path.join(SUBMISSIONS_DIR, filename)
    submission.to_csv(filepath, index=False)
    print(f"Submission saved to {filepath}")


# -------------------------
# KERAS MODEL BUILD FUNCTION
# -------------------------
def build_model(*, input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(96, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer='rmsprop', loss="binary_crossentropy", metrics=['auc'])
    return model


# -------------------------
# MAIN FUNCTION
# -------------------------
def main():
    import warnings
    warnings.filterwarnings("ignore", message="Parameters: { \"use_label_encoder\" } are not used.")
    warnings.filterwarnings("ignore", message="The reported value is ignored because this `step`")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
    tf.random.set_seed(RANDOM_STATE)  # Set TensorFlow seed for determinism

    # --- Data Loading and Candidate Preparation ---
    synthetic_train, test = load_data()
    synthetic_train["synthetic_indicator"] = 1  # Mark synthetic
    print(f"Synthetic Dataset:\n{missing_values_summary(synthetic_train)}")

    original_path = os.path.join(DATA_DIR, "Rainfall.csv")
    original = pd.read_csv(original_path)

    # --- NEW CODE: Add missing id column, starting with largest ID from synthetic data +1 ---
    if 'id' not in original.columns:
        if 'id' in synthetic_train.columns:
            start_id = synthetic_train['id'].max() + 1
        else:
            start_id = 1
        original.insert(0, 'id', range(start_id, start_id + len(original)))

    # Renumber the days for the original dataset (if needed)
    original['day'] = range(1, len(original) + 1)

    # Clean column names and process rainfall
    original.columns = [col.strip() for col in original.columns]
    rain_map = {'yes': 1, 'no': 0}
    original['rainfall'] = original['rainfall'].map(rain_map)
    original.dropna(inplace=True)
    original["synthetic_indicator"] = 0  # Mark original

    # Reorder columns into the desired order
    desired_order = ["id", "day", "pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
                     "humidity", "cloud", "sunshine", "winddirection", "windspeed", "rainfall", "synthetic_indicator"]
    original = original[desired_order]

    print(f"Original Dataset:\n{missing_values_summary(original)}")

    # Assert that column names match between original and synthetic datasets
    assert set(original.columns) == set(
        synthetic_train.columns), "Column names do not match between original and synthetic datasets."

    # --- Create a fixed holdout from synthetic data for the comparison ---
    X_syn = synthetic_train.drop(columns=["rainfall"])
    y_syn = synthetic_train["rainfall"]
    X_R, X_hold, y_R, y_hold = train_test_split(
        X_syn, y_syn, test_size=0.2, random_state=RANDOM_STATE, stratify=y_syn
    )
    # Candidate A: synthetic-only training (R)
    candidate_A = X_R.copy()
    candidate_A["rainfall"] = y_R
    # Candidate B: merge synthetic remainder (R) with the entire original dataset
    candidate_B = pd.concat([candidate_A, original], axis=0, ignore_index=True)

    # Apply extended feature engineering to the candidates and holdout
    candidate_A = feature_engineering(candidate_A)
    candidate_B = feature_engineering(candidate_B)
    X_hold_fe = feature_engineering(X_hold.copy())

    print(f"Candidate A Missing Summary:\n{missing_values_summary(candidate_A)}")
    print(f"Candidate B Missing Summary:\n{missing_values_summary(candidate_B)}")
    print(f"Holdout Missing Summary:\n{missing_values_summary(X_hold_fe)}")

    simple_stack = SimpleStackingClassifier(
        estimators=[('lr', LogisticRegression(random_state=RANDOM_STATE))],
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        cv=5,
        n_jobs=-1
    )
    # Fit on Candidate A and evaluate on holdout
    simple_stack.fit(candidate_A.drop(columns=["rainfall"]), candidate_A["rainfall"])
    auc_A = roc_auc_score(y_hold, simple_stack.predict_proba(X_hold_fe)[:, 1])

    # Fit on Candidate B and evaluate on holdout
    simple_stack.fit(candidate_B.drop(columns=["rainfall"]), candidate_B["rainfall"])
    auc_B = roc_auc_score(y_hold, simple_stack.predict_proba(X_hold_fe)[:, 1])

    print(f"Holdout ROC AUC (Candidate A, synthetic-only): {auc_A:.4f}")
    print(f"Holdout ROC AUC (Candidate B, full candidate): {auc_B:.4f}")

    # --- Decision: choose the candidate with the higher ROC AUC for final training ---
    if auc_B > auc_A:
        print("Choosing full candidate (synthetic + original) for final training.")
        final_candidate = pd.concat([synthetic_train, original], axis=0, ignore_index=True)
    else:
        print("Choosing synthetic-only candidate for final training.")
        final_candidate = synthetic_train.copy()

    # --- Final Training Data Preparation ---
    final_candidate = final_candidate.fillna(0)
    final_candidate = feature_engineering(final_candidate)
    target = 'rainfall'
    id_col = 'id'
    if id_col in final_candidate.columns:
        X_final = final_candidate.drop(columns=[target, id_col])
    else:
        X_final = final_candidate.drop(columns=[target])
    y_final = final_candidate[target]

    # Process test data: mark as synthetic and apply feature engineering
    test["synthetic_indicator"] = 1
    test = test.fillna(0)
    test = feature_engineering(test)
    print(f"Test Dataset Missing Summary:\n{missing_values_summary(test)}")

    # --- Proceed with Optuna tuning and final model training on the full chosen candidate ---
    cv_mode = "stratified"
    timeout = 3600 / 30  # Adjust timeout as needed
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.HyperbandPruner()

    print("\n####### Tuning LightGBM #######")
    study_lgb = optuna.create_study(direction='maximize', study_name="lgb_auc", sampler=sampler, pruner=pruner)
    study_lgb.optimize(lambda trial: objective_lgb(trial, X_final, y_final, cv_mode=cv_mode),
                       timeout=timeout, show_progress_bar=True)
    best_params_lgb = study_lgb.best_trial.params
    best_params_lgb.update({
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE
    })
    print("Best LightGBM params:", best_params_lgb)

    print("\n####### Tuning RandomForest #######")
    study_rf = optuna.create_study(direction='maximize', study_name="rf_auc", sampler=sampler, pruner=pruner)
    study_rf.optimize(lambda trial: objective_rf(trial, X_final, y_final, cv_mode=cv_mode),
                      timeout=timeout, show_progress_bar=True)
    best_params_rf = study_rf.best_trial.params
    best_params_rf.update({'random_state': RANDOM_STATE})
    print("Best RF params:", best_params_rf)

    print("\n####### Tuning XGBoost #######")
    study_xgb = optuna.create_study(direction='maximize', study_name="xgb_auc", sampler=sampler, pruner=pruner)
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_final, y_final, cv_mode=cv_mode),
                       timeout=timeout, show_progress_bar=True)
    best_params_xgb = study_xgb.best_trial.params
    best_params_xgb.update({'random_state': RANDOM_STATE})
    print("Best XGB params:", best_params_xgb)

    print("\n####### Tuning CatBoost #######")
    study_cat = optuna.create_study(direction='maximize', study_name="cat_auc", sampler=sampler, pruner=pruner)
    study_cat.optimize(lambda trial: objective_cat(trial, X_final, y_final, cv_mode=cv_mode),
                       timeout=timeout, show_progress_bar=True)
    best_params_cat = study_cat.best_trial.params
    best_params_cat.update({'random_state': RANDOM_STATE})
    print("Best CatBoost params:", best_params_cat)

    print("\n####### Training Final Models #######")
    # Train final models on full chosen candidate
    lgb_model, scaler = train_final_model_lgb(X_final, y_final, best_params_lgb)
    rf_model, _ = train_final_model_rf(X_final, y_final, best_params_rf)
    xgb_model, _ = train_final_model_xgb(X_final, y_final, best_params_xgb)
    cat_model, _ = train_final_model_cat(X_final, y_final, best_params_cat)
    X_final_scaled = scaler.transform(X_final)

    ### DEFINE NEURAL NETWORK PIPELINE (includes its own scaling)
    nn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nn', KerasClassifier(
            model=build_model,
            epochs=30,
            batch_size=16,
            verbose=0,
            random_state=RANDOM_STATE,
            input_shape=(X_final_scaled.shape[1],)
        ))
    ])

    final_lr = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)
    from joblib import parallel_backend
    with parallel_backend('threading'):
        stack = StackingClassifier(
            estimators=[
                ('lgb', lgb_model),
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('cat', cat_model),
                ('nn', nn_pipeline)
            ],
            final_estimator=final_lr,
            cv=5,
            n_jobs=-1,
            passthrough=True
        )
        stack.fit(X_final_scaled, y_final)

    submission_number = get_next_submission_number()
    save_model(stack, submission_number)

    if id_col in test.columns:
        X_test_final = test.drop(columns=[id_col])
    else:
        X_test_final = test.copy()
    X_test_final = X_test_final.fillna(0)
    X_test_scaled = scaler.transform(X_test_final)
    predictions = stack.predict_proba(X_test_scaled)[:, 1]
    save_submission(
        test[id_col] if id_col in test.columns else np.arange(len(predictions)),
        predictions,
        id_col,
        target,
        submission_number
    )


if __name__ == "__main__":
    main()