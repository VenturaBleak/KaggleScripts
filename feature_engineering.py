def add_name_features(df, numerical_cols, categorical_cols):
    # Split the 'name' column into 'first_name' and 'last_name'
    NAME_COLUMN = 'Name'
    names = df[NAME_COLUMN].str.rsplit(n=1, expand=True)
    df['first_name'] = names[0]
    df['last_name'] = names[1]

    # drop name column
    df = df.drop(NAME_COLUMN, axis=1)

    if NAME_COLUMN in categorical_cols:
        # modify numerical and categorical columns lists
        categorical_cols.append('first_name')
        categorical_cols.append('last_name')
        categorical_cols.remove(NAME_COLUMN)

    return df, numerical_cols, categorical_cols

def custom_feature_engineering(df, numerical_cols, categorical_cols):
    # Here you can add more custom feature engineering steps if needed
    # df, numerical_cols, categorical_cols = add_name_features(df, numerical_cols, categorical_cols)
    return df, numerical_cols, categorical_cols