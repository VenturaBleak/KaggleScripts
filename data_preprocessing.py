"""
data_preprocessing.py

Script that:
- Loads raw competition data and underlying data
- Sanitizes column names and reduces memory usage
- Splits competition train data into train/validation
- Applies imputers + binary encoding to categorical features and optionally scales numeric columns
- Processes underlying data by adding an id column (if missing), filling missing values with -99,
  and extracting/saving the underlying target (ground truth)
- Trains an LGBM model (via Optuna) on underlying data and uses it to generate a meta-feature
  that extends all competition datasets (train, val, full train, test)
- Scales the meta-feature separately
- Saves preprocessed outputs to disk
"""

import numpy as np
import pandas as pd
import os
import re
import time
import joblib
from category_encoders.binary import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from feature_engineering import custom_feature_engineering

TARGET_COLUMN = 'Price'
ID_COLUMN = 'id'

##############################
# TQDM Callback for Optuna Study
##############################
class TimeProgressCallback:
    """
    A callback that updates a tqdm progress bar after each trial,
    showing how many seconds have elapsed out of the total_time.
    """
    def __init__(self, total_time: float):
        self.total_time = total_time
        self.start_time = time.time()
        self.pbar = tqdm(total=round(self.total_time, 1), desc="Optuna Trials", ncols=100, unit="sec")

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        elapsed = time.time() - self.start_time
        if elapsed < self.total_time:
            self.pbar.n = round(elapsed, 1)
            self.pbar.refresh()
        else:
            self.pbar.n = self.total_time
            self.pbar.refresh()
            self.pbar.close()

##############################
# Preprocessing Functions
##############################
def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-UTF8 characters and replace spaces with underscores in column names."""
    valid_utf8_char_pattern = re.compile(r'[\w\s-]', flags=re.UNICODE)
    df.columns = [
        ''.join(valid_utf8_char_pattern.findall(col)) if col else col
        for col in df.columns
    ]
    df.columns = df.columns.str.replace(' ', '_')
    return df

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns and convert object columns to category to save memory."""
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage before optimization: {:.2f} MB'.format(start_mem))
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        pct_decrease = 100 * (start_mem - end_mem) / start_mem if start_mem else 0
        print('Decreased by {:.1f}%'.format(pct_decrease))
    return df

def preprocess_data(
    df: pd.DataFrame,
    encoder: BinaryEncoder = None,
    imputer_num: SimpleImputer = None,
    imputer_cat: SimpleImputer = None,
    scaler: StandardScaler = None,
    fit: bool = True,
    id_column: str = None
):
    """
    Main preprocessing function that:
      1) Sanitizes column names
      2) Identifies numeric and categorical columns
      3) Runs custom feature engineering
      4) Reduces memory usage
      5) Imputes missing values
      6) Binary-encodes categorical columns
      7) Scales only the originally numeric columns
      8) Reattaches the id column
    Returns (df, encoder, imputer_num, imputer_cat, scaler).
    """
    # 1) Sanitize columns
    df = sanitize_column_names(df)
    # 2) Identify numeric and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    # Save original numeric columns to scale only these later
    orig_numerical_cols = numerical_cols.copy()
    # 3) Custom feature engineering
    df, numerical_cols, categorical_cols = custom_feature_engineering(df, numerical_cols, categorical_cols)
    # 4) Reduce memory usage
    reduce_mem_usage(df)
    # Separate ID column if present
    if id_column and (id_column in df.columns):
        df_id = df[id_column].copy()
        df.drop(columns=[id_column], inplace=True)
    else:
        df_id = pd.Series(dtype='float64')
    if id_column in numerical_cols:
        numerical_cols.remove(id_column)
    if id_column in categorical_cols:
        categorical_cols.remove(id_column)
    print(f"numerical_cols: {numerical_cols}")
    print(f"categorical_cols: {categorical_cols}")
    # 5) Initialize imputers/encoder/scaler when fitting
    if fit:
        if numerical_cols and imputer_num is None:
            imputer_num = SimpleImputer(strategy='mean')
        if categorical_cols and imputer_cat is None:
            imputer_cat = SimpleImputer(strategy='most_frequent')
        if orig_numerical_cols and scaler is None:
            scaler = StandardScaler()
        if categorical_cols and encoder is None:
            encoder = BinaryEncoder(cols=categorical_cols, drop_invariant=True)
    # 6) Impute categorical columns
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].astype(str)
        if fit and imputer_cat:
            df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
        elif imputer_cat:
            df[categorical_cols] = imputer_cat.transform(df[categorical_cols])
    # 7) Impute numeric columns
    if numerical_cols:
        if fit and imputer_num:
            df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
        elif imputer_num:
            df[numerical_cols] = imputer_num.transform(df[numerical_cols])
    # 8) Binary encode categorical columns
    if categorical_cols and encoder:
        if fit:
            df = encoder.fit_transform(df)
        else:
            df = encoder.transform(df)
    # 9) Scale only the originally numeric columns
    cols_to_scale = [col for col in orig_numerical_cols if col in df.columns]
    if cols_to_scale and scaler is not None:
        if fit:
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    # 10) Reattach ID column if it existed
    if not df_id.empty:
        df.insert(0, id_column, df_id)
    return df, encoder, imputer_num, imputer_cat, scaler

##############################
# Optional custom feature engineering functions
##############################
def add_name_features(df, numerical_cols, categorical_cols):
    """
    Example: Split the 'Name' column into 'first_name' and 'last_name'.
    """
    NAME_COLUMN = 'Name'
    if NAME_COLUMN in df.columns:
        names = df[NAME_COLUMN].str.rsplit(n=1, expand=True)
        df['first_name'] = names[0]
        df['last_name'] = names[1]
        df = df.drop(NAME_COLUMN, axis=1)
        if NAME_COLUMN in categorical_cols:
            categorical_cols.remove(NAME_COLUMN)
            categorical_cols.extend(['first_name', 'last_name'])
    return df, numerical_cols, categorical_cols

def custom_feature_engineering(df, numerical_cols, categorical_cols):
    # Uncomment the following line to apply name splitting if desired:
    # df, numerical_cols, categorical_cols = add_name_features(df, numerical_cols, categorical_cols)
    return df, numerical_cols, categorical_cols

##############################
# Main Preprocessing Workflow (TQDM used only for the underlying model tuning)
##############################
if __name__ == '__main__':
    print("Starting data preprocessing...")

    COMPETITION_NAME = 'playground-series-s5e2'
    PROBLEM_TYPE = 'regression'
    DATA_DIR = os.path.join(os.getcwd(), COMPETITION_NAME, 'data')
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

    # --- Process Competition Training Data ---
    df_train = pd.read_csv(TRAIN_PATH)
    X = df_train.drop(columns=[TARGET_COLUMN])
    y = df_train[TARGET_COLUMN].astype(np.float32)
    print("Splitting competition data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Preprocessing competition training data...")
    X_train, encoder, imputer_num, imputer_cat, scaler = preprocess_data(
        df=X_train, encoder=None, imputer_num=None, imputer_cat=None, scaler=None, fit=True, id_column=ID_COLUMN
    )
    X_train.to_csv(os.path.join(DATA_DIR, 'X_train_preprocessed.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
    joblib.dump(encoder, os.path.join(DATA_DIR, 'binary_encoder.joblib'))
    joblib.dump(imputer_num, os.path.join(DATA_DIR, 'imputer_num.joblib'))
    joblib.dump(imputer_cat, os.path.join(DATA_DIR, 'imputer_cat.joblib'))
    joblib.dump(scaler, os.path.join(DATA_DIR, 'scaler.joblib'))

    print("Preprocessing competition validation data...")
    X_val, _, _, _, _ = preprocess_data(
        df=X_val, encoder=encoder, imputer_num=imputer_num, imputer_cat=imputer_cat, scaler=scaler, fit=False, id_column=ID_COLUMN
    )
    X_val.to_csv(os.path.join(DATA_DIR, 'X_val_preprocessed.csv'), index=False)
    y_val.to_csv(os.path.join(DATA_DIR, 'y_val.csv'), index=False)

    print("Preprocessing full competition training data...")
    df_full_train = pd.read_csv(TRAIN_PATH)
    X_full_train = df_full_train.drop(columns=[TARGET_COLUMN])
    y_full_train = df_full_train[TARGET_COLUMN].astype(np.float32)
    X_full_train, _, _, _, _ = preprocess_data(
        df=X_full_train, encoder=encoder, imputer_num=imputer_num, imputer_cat=imputer_cat, scaler=scaler, fit=False, id_column=ID_COLUMN
    )
    X_full_train.to_csv(os.path.join(DATA_DIR, 'X_full_train_preprocessed.csv'), index=False)
    y_full_train.to_csv(os.path.join(DATA_DIR, 'y_full_train.csv'), index=False)

    print("Preprocessing competition test data...")
    df_test = pd.read_csv(TEST_PATH)
    X_test, _, _, _, _ = preprocess_data(
        df=df_test, encoder=encoder, imputer_num=imputer_num, imputer_cat=imputer_cat, scaler=scaler, fit=False, id_column=ID_COLUMN
    )
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test_preprocessed.csv'), index=False)

    # --- Process Underlying Data ---
    underlying_file = os.path.join(DATA_DIR, 'underlying.csv')
    underlying_processed_file = os.path.join(DATA_DIR, 'underlying_preprocessed.csv')
    y_underlying_file = os.path.join(DATA_DIR, 'y_underlying.csv')
    if os.path.exists(underlying_processed_file):
        print("Underlying data has already been preprocessed. Skipping underlying data processing.")
        df_underlying_processed = pd.read_csv(underlying_processed_file)
        if os.path.exists(y_underlying_file):
            y_underlying = pd.read_csv(y_underlying_file)[TARGET_COLUMN]
        else:
            y_underlying = None
    elif os.path.exists(underlying_file):
        print("Processing underlying data...")
        df_underlying = pd.read_csv(underlying_file)
        if TARGET_COLUMN in df_underlying.columns:
            df_underlying = df_underlying.dropna(subset=[TARGET_COLUMN])
            y_underlying = df_underlying[TARGET_COLUMN].copy()
            df_underlying.drop(columns=[TARGET_COLUMN], inplace=True)
            pd.DataFrame({TARGET_COLUMN: y_underlying}).to_csv(y_underlying_file, index=False)
            print("Saved underlying target to", y_underlying_file)
        else:
            y_underlying = None
        if 'id' not in df_underlying.columns:
            df_underlying.insert(0, 'id', range(1, len(df_underlying) + 1))
            print("Added 'id' column to underlying data.")
        df_underlying.fillna(-99, inplace=True)
        print("Filled missing values in underlying data with -99.")
        df_underlying_processed, _, _, _, _ = preprocess_data(
            df=df_underlying, encoder=encoder, imputer_num=imputer_num, imputer_cat=imputer_cat, scaler=scaler, fit=False, id_column=ID_COLUMN
        )
        df_underlying_processed.to_csv(underlying_processed_file, index=False)
        print(f"Underlying data has been preprocessed and saved to {underlying_processed_file}")
    else:
        print("No underlying data file found.")
        y_underlying = None

    # --- Underlying Model Training & Meta-Feature Generation ---
    UNDERLYING_MODEL_TIMEOUT = 3600 / 6
    if y_underlying is not None:
        print("Training underlying model to generate meta-features...")
        X_underlying = pd.read_csv(underlying_processed_file)
        X_underlying_train, X_underlying_val, y_underlying_train, y_underlying_val = train_test_split(
            X_underlying, y_underlying, test_size=0.2, random_state=42
        )
        # Create TQDM-based callback for the Optuna study
        time_progress_callback = TimeProgressCallback(total_time=UNDERLYING_MODEL_TIMEOUT)

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            }
            cv = 3
            errors = []
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X_underlying_train):
                X_train_cv = X_underlying_train.iloc[train_idx]
                y_train_cv = y_underlying_train.iloc[train_idx]
                X_val_cv = X_underlying_train.iloc[val_idx]
                y_val_cv = y_underlying_train.iloc[val_idx]
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    callbacks=[lgb.early_stopping(50)]
                )
                preds = model.predict(X_val_cv)
                error = np.sqrt(mean_squared_error(y_val_cv, preds))
                errors.append(error)
            return np.mean(errors)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, timeout=UNDERLYING_MODEL_TIMEOUT, callbacks=[time_progress_callback])
        best_params = study.best_params
        print("Best parameters for underlying model:", best_params)

        underlying_model = lgb.LGBMRegressor(**best_params)
        underlying_model.fit(X_underlying, y_underlying)

        # Function to extend a dataset with the meta-feature
        def extend_with_meta(file_base):
            base_path = os.path.join(DATA_DIR, f"{file_base}_preprocessed.csv")
            df = pd.read_csv(base_path)
            preds = underlying_model.predict(df)
            meta_scaler = StandardScaler()
            preds_scaled = meta_scaler.fit_transform(preds.reshape(-1, 1)).flatten()
            df['underlying_pred'] = preds_scaled
            extended_path = os.path.join(DATA_DIR, f"{file_base}_extended.csv")
            df.to_csv(extended_path, index=False)
            print(f"Extended {file_base} data saved to {extended_path}")
            return df

        extend_with_meta("X_train")
        extend_with_meta("X_val")
        extend_with_meta("X_full_train")
        extend_with_meta("X_test")
    else:
        print("Underlying target not available; skipping underlying model training and meta-feature generation.")

    print("Data preprocessing completed.")