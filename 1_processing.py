#!/usr/bin/env python3
"""
Script for Playground Series S5E2: Regression Competition

This script performs the following:
1. Loads train, test, and underlying datasets.
2. Adds an ID column to underlying if needed.
3. Adds a "source" column:
   - For train: "train"
   - For underlying: "underlying" (if Price exists)
   - For test: forced to "train"
4. Combines all rows (train, underlying, test) for consistent encoding.
5. Fills missing values and binary-encodes all categorical columns.
6. Splits the encoded data into two sets:
   a) Preprocessed individual datasets (with the encoded “source” features dropped)
      - train_preprocessed.csv, underlying_preprocessed.csv, test_preprocessed.csv
   b) Combined datasets (keeping the encoded “source” features)
      - combined_train.csv (train + underlying) and combined_test.csv (test)
7. Prints the name, head, columns, and shape of each saved dataset for verification.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import numpy as np
import pandas as pd
import category_encoders as ce


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
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(-1)
        else:
            df[col] = df[col].fillna("missing")
    return df


def encode_categorical_binary(df, categorical_cols):
    """
    Binary-encode all specified categorical columns using category_encoders.
    """
    encoder = ce.BinaryEncoder(cols=categorical_cols, return_df=True)
    df = encoder.fit_transform(df)
    return df


# -----------------------------
# Main Processing
# -----------------------------
def main():
    # Define file paths
    TRAIN_PATH = "playground-series-s5e2/data/train.csv"
    TEST_PATH = "playground-series-s5e2/data/test.csv"
    UNDERLYING_PATH = "playground-series-s5e2/data/underlying.csv"

    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    underlying_df = pd.read_csv(UNDERLYING_PATH)

    # Print initial shapes
    print("Initial Shapes:")
    print("  Train:", train_df.shape)
    print("  Test:", test_df.shape)
    print("  Underlying:", underlying_df.shape)
    print("------")

    # Add ID column to underlying if necessary
    if 'id' not in underlying_df.columns:
        underlying_df['id'] = np.arange(len(underlying_df)) + len(train_df) + len(test_df)
        cols = ['id'] + [col for col in underlying_df.columns if col != 'id']
        underlying_df = underlying_df[cols]
    if 'Unnamed: 0' in underlying_df.columns:
        underlying_df = underlying_df.rename(columns={'Unnamed: 0': 'id'})

    # Add "source" column:
    train_df["source"] = "train"
    test_df["source"] = "train"  # force test to be treated as "train"
    # For underlying, label as "underlying" if it contains the target, else label it "underlying" anyway.
    underlying_df["source"] = "underlying"

    # Record lengths for later splitting
    n_train = len(train_df)
    n_underlying = len(underlying_df)
    n_test = len(test_df)

    # -----------------------------
    # Create Combined Data for Consistent Encoding
    # -----------------------------
    # For individual (preprocessed) datasets, we want to ensure consistent columns.
    # We combine all rows (train, underlying, test) to encode them together.
    all_data = pd.concat([train_df, underlying_df, test_df], ignore_index=True)
    all_data = fill_missing(all_data)

    # Identify categorical columns (all object columns)
    categorical_cols = all_data.select_dtypes(include=["object"]).columns.tolist()

    # Binary-encode all categorical columns (including "source")
    all_data_encoded = encode_categorical_binary(all_data, categorical_cols)

    # -----------------------------
    # Split Encoded Data Back into Individual Datasets (Group 1)
    # -----------------------------
    processed_train = all_data_encoded.iloc[:n_train].reset_index(drop=True)
    processed_underlying = all_data_encoded.iloc[n_train:n_train + n_underlying].reset_index(drop=True)
    processed_test = all_data_encoded.iloc[n_train + n_underlying:].reset_index(drop=True)

    # For the preprocessed individual datasets, remove the encoded source features.
    # (They are named with a prefix "source_" by the BinaryEncoder.)
    source_cols = [col for col in processed_train.columns if col.startswith("source_")]
    processed_train_no_source = processed_train.drop(columns=source_cols)
    processed_underlying_no_source = processed_underlying.drop(columns=source_cols)
    processed_test_no_source = processed_test.drop(columns=source_cols)

    # -----------------------------
    # Create Combined Datasets (Group 2)
    # -----------------------------
    # For the combined datasets, we retain the source features.
    # Combined train is the union of train and underlying.
    combined_train = pd.concat([processed_train, processed_underlying], ignore_index=True)
    combined_test = processed_test.copy()

    # -----------------------------
    # Save the Datasets
    # -----------------------------
    # Group 1: Preprocessed individual datasets (without source features)
    processed_train_no_source.to_csv("playground-series-s5e2/data/train_preprocessed.csv", index=False)
    processed_underlying_no_source.to_csv("playground-series-s5e2/data/underlying_preprocessed.csv", index=False)
    processed_test_no_source.to_csv("playground-series-s5e2/data/test_preprocessed.csv", index=False)

    # Group 2: Combined datasets (with source features retained)
    combined_train.to_csv("playground-series-s5e2/data/combined_train.csv", index=False)
    combined_test.to_csv("playground-series-s5e2/data/combined_test.csv", index=False)

    # -----------------------------
    # Print Dataset Details for Verification
    # -----------------------------
    def print_dataset_info(name, df):
        print(f"=== {name} ===")
        print("Head:")
        print(df.head())
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("------\n")

    print("Group 1: Preprocessed Individual Datasets (source features removed)")
    print_dataset_info("Train Preprocessed (train_preprocessed.csv)", processed_train_no_source)
    print_dataset_info("Underlying Preprocessed (underlying_preprocessed.csv)", processed_underlying_no_source)
    print_dataset_info("Test Preprocessed (test_preprocessed.csv)", processed_test_no_source)

    print("Group 2: Combined Datasets (source features retained)")
    print_dataset_info("Combined Train (combined_train.csv)", combined_train)
    print_dataset_info("Combined Test (combined_test.csv)", combined_test)


if __name__ == '__main__':
    main()