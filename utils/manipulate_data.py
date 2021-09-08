from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_target_binary(df, column):
    target_mean = df[column].mean()
    df.loc[df[column] < target_mean, column] = 0
    df.loc[df.quality > target_mean, column] = 1
    return df


def split_features_target(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

