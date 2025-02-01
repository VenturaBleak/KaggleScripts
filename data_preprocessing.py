"""
data_preprocessing.py

Script that:
- Loads raw data
- Sanitizes column names
- Reduces memory usage
- Splits into train/validation
- Applies imputers + binary encoding to categorical features
- Optionally scales numeric columns
- Saves preprocessed outputs to disk
"""

import numpy as np
import pandas as pd
import os
from category_encoders.binary import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import re

from feature_engineering import custom_feature_engineering


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove non-UTF8 characters and replace spaces with underscores in column names.
    """
    valid_utf8_char_pattern = re.compile(r'[\w\s-]', flags=re.UNICODE)
    df.columns = [
        ''.join(valid_utf8_char_pattern.findall(col)) if col else col
        for col in df.columns
    ]
    df.columns = df.columns.str.replace(' ', '_')
    return df


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns and convert 'object' columns to 'category' to save memory.
    Be cautious with category dtype if you rely on scikit-learn transformations that expect strings.
    """
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
                # float columns
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
        else:
            # Convert object → category for memory savings
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
    Main data preprocessing function. It:
      1) Sanitizes column names
      2) Identifies numeric/categorical cols
      3) Runs custom feature engineering
      4) Reduces memory usage (object->category)
      5) Imputes numeric/categorical columns
      6) Binary-encodes categorical columns -> numeric
      7) Optionally scales numeric columns
      8) Reattaches any ID column

    Returns (df, encoder, imputer_num, imputer_cat, scaler).
    After this, the DataFrame should have no 'object' columns, since they've been binary-encoded.
    """

    # 1) Sanitize columns
    df = sanitize_column_names(df)

    # 2) Identify numeric vs object cols
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # 3) Feature engineering
    df, numerical_cols, categorical_cols = custom_feature_engineering(
        df, numerical_cols, categorical_cols
    )

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

    # 5) Initialize imputers/encoder/scaler if fitting
    if fit:
        if numerical_cols and imputer_num is None:
            imputer_num = SimpleImputer(strategy='mean')
        if categorical_cols and imputer_cat is None:
            imputer_cat = SimpleImputer(strategy='most_frequent')
        if numerical_cols and scaler is None:
            scaler = StandardScaler()
        # Initialize BinaryEncoder if we have categorical columns
        if categorical_cols and encoder is None:
            encoder = BinaryEncoder(cols=categorical_cols, drop_invariant=True)

    # --- Impute categorical columns ---
    if categorical_cols:
        # Convert category->string so SimpleImputer doesn't fail
        df[categorical_cols] = df[categorical_cols].astype(str)

        if fit and imputer_cat:
            df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
        elif imputer_cat:
            df[categorical_cols] = imputer_cat.transform(df[categorical_cols])

    # --- Impute numeric columns ---
    if numerical_cols:
        if fit and imputer_num:
            df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
        elif imputer_num:
            df[numerical_cols] = imputer_num.transform(df[numerical_cols])

    # 6) Binary Encode the (now-imputed) categorical columns
    if categorical_cols and encoder:
        if fit:
            df = encoder.fit_transform(df)
        else:
            df = encoder.transform(df)

    # After binary encoding, we no longer have old categorical columns.
    # If you want, you could update numerical_cols to reflect the new columns:
    # but typically we can treat them all as numeric now.

    # 7) Scale all numeric columns if you wish
    # Re-detect numeric columns after encoding
    new_numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if new_numerical_cols and scaler is not None:
        if fit:
            df[new_numerical_cols] = scaler.fit_transform(df[new_numerical_cols])
        else:
            df[new_numerical_cols] = scaler.transform(df[new_numerical_cols])

    # 8) Reattach ID column if it existed
    if not df_id.empty:
        df.insert(0, id_column, df_id)

    return df, encoder, imputer_num, imputer_cat, scaler


if __name__ == '__main__':
    """
    Main block for data preprocessing:
      - Load raw data
      - Split train/val
      - Preprocess (w/ binary encoding)
      - Save results
    """
    print("Starting data preprocessing...")

    COMPETITION_NAME = 'playground-series-s5e2'
    TARGET_COLUMN = 'Price'
    ID_COLUMN = 'id'
    PROBLEM_TYPE = 'regression'

    data_dir = os.path.join(os.getcwd(), COMPETITION_NAME, 'data')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    # Load raw train
    df_train = pd.read_csv(train_path)
    X = df_train.drop(columns=[TARGET_COLUMN])
    y = df_train[TARGET_COLUMN]

    if PROBLEM_TYPE == 'classification':
        y = y.astype(int)
    else:
        y = y.astype(np.float32)

    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1) Preprocess training data
    print("---------------------------------")
    print("Preprocessing training data...")
    X_train, encoder, imputer_num, imputer_cat, scaler = preprocess_data(
        df=X_train,
        encoder=None,
        imputer_num=None,
        imputer_cat=None,
        scaler=None,
        fit=True,
        id_column=ID_COLUMN
    )
    X_train.to_csv(os.path.join(data_dir, 'X_train_preprocessed.csv'), index=False)
    y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)

    # Save fitted objects
    joblib.dump(encoder, os.path.join(data_dir, 'binary_encoder.joblib'))
    joblib.dump(imputer_num, os.path.join(data_dir, 'imputer_num.joblib'))
    joblib.dump(imputer_cat, os.path.join(data_dir, 'imputer_cat.joblib'))
    joblib.dump(scaler, os.path.join(data_dir, 'scaler.joblib'))

    # 2) Preprocess validation data
    print("---------------------------------")
    print("Preprocessing validation data...")
    X_val, _, _, _, _ = preprocess_data(
        df=X_val,
        encoder=encoder,
        imputer_num=imputer_num,
        imputer_cat=imputer_cat,
        scaler=scaler,
        fit=False,
        id_column=ID_COLUMN
    )
    X_val.to_csv(os.path.join(data_dir, 'X_val_preprocessed.csv'), index=False)
    y_val.to_csv(os.path.join(data_dir, 'y_val.csv'), index=False)

    # 3) Preprocess full training data
    print("---------------------------------")
    print("Preprocessing full training data...")
    df_full_train = pd.read_csv(train_path)
    X_full_train = df_full_train.drop(columns=[TARGET_COLUMN])
    y_full_train = df_full_train[TARGET_COLUMN]
    if PROBLEM_TYPE == 'classification':
        y_full_train = y_full_train.astype(int)
    else:
        y_full_train = y_full_train.astype(np.float32)

    X_full_train, _, _, _, _ = preprocess_data(
        df=X_full_train,
        encoder=encoder,
        imputer_num=imputer_num,
        imputer_cat=imputer_cat,
        scaler=scaler,
        fit=False,
        id_column=ID_COLUMN
    )
    X_full_train.to_csv(os.path.join(data_dir, 'X_full_train_preprocessed.csv'), index=False)
    y_full_train.to_csv(os.path.join(data_dir, 'y_full_train.csv'), index=False)

    # 4) Preprocess test data
    print("---------------------------------")
    print("Preprocessing test data...")
    df_test = pd.read_csv(test_path)
    X_test, _, _, _, _ = preprocess_data(
        df=df_test,
        encoder=encoder,
        imputer_num=imputer_num,
        imputer_cat=imputer_cat,
        scaler=scaler,
        fit=False,
        id_column=ID_COLUMN
    )
    X_test.to_csv(os.path.join(data_dir, 'X_test_preprocessed.csv'), index=False)

    print("Data preprocessing completed.")