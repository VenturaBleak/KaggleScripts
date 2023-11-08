import numpy as np
import pandas as pd
import os
from category_encoders.binary import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib


# Memory reduction function
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage before optimization: {:.2f} MB'.format(start_mem))
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# Function to preprocess the data
# Function to preprocess the data
def preprocess_data(df, numerical_cols, categorical_cols, encoder=None, imputer_num=None, imputer_cat=None, scaler=None, fit=True, id_column=None):
    # Reduce memory usage in place
    reduce_mem_usage(df)

    # Separate ID column if present
    df_id = df[id_column] if id_column in df else pd.DataFrame()

    # Remove ID column for processing
    if id_column in numerical_cols: numerical_cols.remove(id_column)
    if id_column in categorical_cols: categorical_cols.remove(id_column)

    # Initialize imputers and scaler if fitting
    if fit:
        if numerical_cols:
            imputer_num = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
        if categorical_cols:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            encoder = BinaryEncoder(cols=categorical_cols)

    # Process categorical columns if they exist
    if categorical_cols:
        # Impute and encode categorical columns
        if fit:
            df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
            df = encoder.fit_transform(df)
        else:
            df[categorical_cols] = imputer_cat.transform(df[categorical_cols])
            df = encoder.transform(df)

    # Process numerical columns if they exist
    if numerical_cols:
        # Impute and scale numerical columns
        if fit:
            df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = imputer_num.transform(df[numerical_cols])
            df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Combine ID column back if it was separated
    if not df_id.empty:
        df = pd.concat([df_id, df], axis=1)

    return df, encoder, imputer_num, imputer_cat, scaler

# Main section
if __name__ == '__main__':
    print("Starting data preprocessing...")

    # competition specific, change as needed
    COMPETITION_NAME = 'playground-series-s3e24'
    TARGET_COLUMN = 'smoking'
    ID_COLUMN = 'id'

    ######################## Define data paths ########################
    data_dir = os.path.join(os.getcwd(), COMPETITION_NAME, 'data')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    ######################## Load and split the training data ########################
    df_train = pd.read_csv(train_path)
    X = df_train.drop(TARGET_COLUMN, axis=1)
    y = df_train[TARGET_COLUMN]
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    ######################## Define columns ########################
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    ######################## Preprocess the training data ########################
    print("---------------------------------")
    print("Preprocessing training data...")
    X_train, encoder, imputer_num, imputer_cat, scaler = preprocess_data(
        X_train, numerical_cols, categorical_cols, fit=True, id_column=ID_COLUMN
    )

    ######################## Save the preprocessed training data ########################
    print("Saving preprocessed training data...")
    X_train.to_csv(os.path.join(data_dir, 'X_train_preprocessed.csv'), index=False)
    y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
    joblib.dump(imputer_num, os.path.join(data_dir, 'imputer_num.joblib'))
    joblib.dump(imputer_cat, os.path.join(data_dir, 'imputer_cat.joblib'))
    joblib.dump(scaler, os.path.join(data_dir, 'scaler.joblib'))
    if categorical_cols:  # Save the encoder only if there are categorical columns to encode
        joblib.dump(encoder, os.path.join(data_dir, 'encoder.joblib'))

    ######################## Preprocess the validation data ########################
    print("---------------------------------")
    print("Preprocessing validation data...")
    X_val, _, _, _, _ = preprocess_data(
        X_val, numerical_cols, categorical_cols, encoder, imputer_num, imputer_cat, scaler, fit=False, id_column=ID_COLUMN
    )
    X_val.to_csv(os.path.join(data_dir, 'X_val_preprocessed.csv'), index=False)
    y_val.to_csv(os.path.join(data_dir, 'y_val.csv'), index=False)

    ######################## Preprocess the full training data ########################
    print("---------------------------------")
    print("Preprocessing full training data...")
    df_full_train = pd.read_csv(train_path)
    X_full_train = df_full_train.drop(TARGET_COLUMN, axis=1)
    y_full_train = df_full_train[TARGET_COLUMN]
    X_full_train, _, _, _, _ = preprocess_data(
        X_full_train, numerical_cols, categorical_cols, encoder, imputer_num, imputer_cat, scaler, fit=False, id_column=ID_COLUMN
    )
    X_full_train.to_csv(os.path.join(data_dir, 'X_full_train_preprocessed.csv'), index=False)
    y_full_train.to_csv(os.path.join(data_dir, 'y_full_train.csv'), index=False)

    ######################## Preprocess the test data ########################
    print("---------------------------------")
    print("Preprocessing test data...")
    df_test = pd.read_csv(test_path)
    X_test, _, _, _, _ = preprocess_data(
        df_test, numerical_cols, categorical_cols, encoder, imputer_num, imputer_cat, scaler, fit=False, id_column=ID_COLUMN
    )
    X_test.to_csv(os.path.join(data_dir, 'X_test_preprocessed.csv'), index=False)

    print("Data preprocessing completed.")