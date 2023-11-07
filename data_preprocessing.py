import numpy as np
import pandas as pd
import os
from category_encoders.binary import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
def preprocess_data(df_reduced, encoder=None, imputer_num=None, imputer_cat=None, scaler=None,
                    fit=True):
    # Reduce memory usage
    df_reduced = reduce_mem_usage(df_reduced)

    # Define categorical and numerical columns
    # These should be updated to reflect your dataset's columns
    # get numerical columns
    numerical_cols = [cname for cname in df_reduced.columns if
                      df_reduced[cname].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]

    # get categorical columns
    categorical_cols = [cname for cname in df_reduced.columns if df_reduced[cname].dtype == "object"]

    # Separate into categorical and numerical columns
    X_cat = df_reduced[categorical_cols].astype(str)
    X_num = df_reduced[numerical_cols]

    if X_cat.empty or X_num.empty:
        raise ValueError("Categorical or Numerical dataframe is empty after splitting.")

    # Fit or transform the data with the encoder and imputers
    if fit:
        # Impute missing values in numerical columns
        imputer_num = SimpleImputer(strategy='mean')
        X_num = imputer_num.fit_transform(X_num)

        # Impute missing values in categorical columns (after encoding)
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_cat = imputer_cat.fit_transform(X_cat)

        # Fit and transform categorical columns with Binary Encoder
        encoder = BinaryEncoder(cols=categorical_cols)
        X_cat = encoder.fit_transform(X_cat)

        # Standardize numerical columns
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
    else:
        X_num = imputer_num.transform(X_num)
        X_cat = imputer_cat.transform(X_cat)
        X_cat = encoder.transform(X_cat)
        X_num = scaler.transform(X_num)

    # Reconstruct the dataframe
    X_processed = np.concatenate([X_num, X_cat], axis=1)

    return X_processed, encoder, imputer_num, imputer_cat, scaler


# Main section
if __name__ == '__main__':

    competition_name = 'playground-series-s3e24'

    # Define data paths
    data_dir = os.path.join(os.getcwd(), competition_name, 'data')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    # Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Define target column
    target_column = 'smoking'  # Replace with actual target column name

    print(f"Dytpes: {df_train.dtypes}")

    # Split data into features and target
    X_train = df_train.drop(target_column, axis=1,
                            inplace=False)
    y_train = df_train[target_column]

    # Preprocess training data
    X_train_processed, encoder, imputer_num, imputer_cat, scaler = preprocess_data(X_train)

    # Save preprocessed training data, target and the encoders for later use
    pd.DataFrame(X_train_processed).to_csv(os.path.join(data_dir, 'X_train_preprocessed.csv'), index=False)
    y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
    joblib.dump(encoder, os.path.join(data_dir, 'encoder.joblib'))
    joblib.dump(imputer_num, os.path.join(data_dir, 'imputer_num.joblib'))
    joblib.dump(imputer_cat, os.path.join(data_dir, 'imputer_cat.joblib'))
    joblib.dump(scaler, os.path.join(data_dir, 'scaler.joblib'))

    # Preprocess test data with the same encoder, imputers and scaler
    # Note that the target column is not present in the test set, hence we drop only the ID column
    X_test_processed, _, _, _, _ = preprocess_data(df_test, encoder, imputer_num,
                                                   imputer_cat, scaler, fit=False)
    pd.DataFrame(X_test_processed).to_csv(os.path.join(data_dir, 'X_test_preprocessed.csv'), index=False)

    print("Data preprocessing completed and files saved!")