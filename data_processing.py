import numpy as np
import pandas as pd
import os

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

# if name == 'main':
if __name__ == '__main__':
    # data path
    data_dir = os.path.join(os.getcwd(), 'playground-series-s3e24', 'data')

    # Load data
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    # Reduce memory usage
    df = reduce_mem_usage(df)

    # specify target column
    target_column = 'smoking'  # replace 'target_column' with the name of your target column

    # Split data into features and target
    X = df.drop(target_column, axis=1)  # replace 'target_column' with the name of your target column
    y = df[target_column]  # replace 'target_column' with the name of your target column

    # get numerical columns
    numerical_cols = [cname for cname in X.columns if
                      X[cname].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]

    # get categorical columns
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

    # binary encode categorical columns
    from category_encoders.binary import BinaryEncoder
    encoder = BinaryEncoder(cols=categorical_cols)
    X = encoder.fit_transform(X)

    # check number of missing values in each column, display if any
    missing_val_count_by_column = (X.isnull().sum())
    print(f'missing_val_count_by_column:\n{missing_val_count_by_column[missing_val_count_by_column > 0]}')

    # use a sophisticated imputer to fill in missing values,
    # ToDo

    # split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

    ########################################
    # Training
    ########################################
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # specify objective function
    objective_fct = 'binary'  # replace 'binary' with the name of your objective function
    metric = None
    direction = None
    if objective_fct == 'binary':
        metric = 'binary_logloss'
        direction = 'maximize'
    elif objective_fct == 'multiclass':
        metric = 'multi_logloss'
        direction = 'maximize'
    elif objective_fct == 'regression':
        metric = 'rmse'
        direction = 'minimize'
    else:
        raise ValueError('Unknown objective function')

    def objective(trial):
        # Splitting the data in each trial to prevent overfitting to the validation set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Define the dataset for LightGBM
        dtrain = lgb.Dataset(X_train, label=y_train)

        # Hyperparameters to be optimized
        param = {
            "objective": objective_fct,
            "metric": metric,
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

        # Training the model
        gbm = lgb.train(param, dtrain)

        # Making predictions
        preds = gbm.predict(X_test)
        pred_labels = np.rint(preds)

        # Evaluating the model
        accuracy = accuracy_score(y_test, pred_labels)
        return accuracy


    # Creating a study object and optimize the objective function
    from optuna.samplers import TPESampler
    sampler = TPESampler(seed=1) # Make the optimization reproducible
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: {}".format(len(study.trials)))

    # Best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Train the model with the best hyperparameters
    model = lgb.LGBMClassifier(**study.best_params)
    model.fit(X_train, y_train)

    ########################################
    # Evaluation
    ########################################
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
    # Predictions
    predictions = model.predict(X_valid)
    pred_labels = (predictions >= 0.5).astype(int)

    # Evaluation metrics
    accuracy = accuracy_score(y_valid, pred_labels)
    roc_auc = roc_auc_score(y_valid, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(confusion_matrix(y_valid, pred_labels))
    print(classification_report(y_valid, pred_labels))

    ########################################
    # Submission
    ########################################
    # load the test data
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # do all the same preprocessing steps as above

    # train the model on the full training data (train + validation)

    # make predictions on the test data
    predictions = model.predict(test_df)
    pred_labels = (predictions >= 0.5).astype(int)

    # save predictions to csv in the following format: id, target -> to_csv(os.path.join(data_dir, 'submission.csv'), index=False)