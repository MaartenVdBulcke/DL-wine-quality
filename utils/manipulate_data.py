from sklearn.model_selection import train_test_split


def make_target_binary(df, column):
    target_mean = df[column].mean()
    df.loc[df[column] < target_mean, column] = 0
    df.loc[df.quality > target_mean, column] = 1
    return df


def split_dataset_in_train_test(df, target, test_size):
    X = df.drop(target, axis=1)
    y = df[target]
    print(y.value_counts())
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
