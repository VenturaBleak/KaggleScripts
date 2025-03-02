#!/usr/bin/env python3
"""
Script for Universal Data Preprocessing

This script performs the following:
1. Loads train and test datasets.
2. Fills missing values:
   - Numeric columns: fill with -1.
   - Categorical columns: fill with 'missing'.
3. Detects categorical columns.
4. Optionally encodes categorical features (user prompt).
5. Optionally scales numeric features (user prompt).
6. Saves the preprocessed train and test datasets.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler


# -----------------------------
# Helper Functions
# -----------------------------
def fill_missing(df):
    """
    Fill missing values:
      - Numeric columns: fill with -1.
      - Categorical columns: fill with the string 'missing'.
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(-1)
        else:
            df[col] = df[col].fillna("missing")
    return df


def encode_categorical(df, categorical_cols):
    """
    Encode categorical columns using binary encoding from category_encoders.
    """
    from category_encoders import BinaryEncoder
    encoder = BinaryEncoder(cols=categorical_cols, drop_invariant=True)
    df = encoder.fit_transform(df)
    return df

def scale_numeric(train_df, test_df, numeric_cols):
    """
    Scale numeric columns using StandardScaler.
    Scaler is fit on train data and then applied to both train and test.
    """
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_df, test_df


# -----------------------------
# Main Processing
# -----------------------------
def main():
    warnings.filterwarnings("ignore")

    # Define file paths (adjust these as needed)
    COMPETITION_NAME = "playground-series-s5e2"
    TRAIN_PATH = os.path.join(COMPETITION_NAME, "data", "train.csv")
    TEST_PATH = os.path.join(COMPETITION_NAME, "data", "test.csv")

    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Initial Shapes:")
    print("  Train:", train_df.shape)
    print("  Test :", test_df.shape)

    # Fill missing values
    train_df = fill_missing(train_df)
    test_df = fill_missing(test_df)

    # Identify categorical columns (we assume these are the object dtype columns)
    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    print("Detected categorical columns:", categorical_cols)

    # Ask the user whether to encode categorical features
    do_encode = input("Do you want to encode categorical features? (y/n): ").strip().lower()
    if do_encode == 'y':
        print("Encoding categorical features...")
        train_df = encode_categorical(train_df, categorical_cols)
        test_df = encode_categorical(test_df, categorical_cols)
    else:
        print("Skipping encoding of categorical features.")

    # Identify numeric columns (for scaling, if desired)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    do_scale = input("Do you want to scale numeric features? (y/n): ").strip().lower()
    if do_scale == 'y':
        print("Scaling numeric features...")
        train_df, test_df = scale_numeric(train_df, test_df, numeric_cols)
    else:
        print("Skipping scaling of numeric features.")

    # Save the preprocessed datasets
    train_out = os.path.join(COMPETITION_NAME, "data", "train_preprocessed.csv")
    test_out = os.path.join(COMPETITION_NAME, "data", "test_preprocessed.csv")

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("Preprocessing complete.")
    print("Preprocessed train data saved to:", train_out)
    print("Preprocessed test data saved to :", test_out)


if __name__ == "__main__":
    main()