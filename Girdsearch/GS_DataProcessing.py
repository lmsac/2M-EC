# Version:1.0 StartHTML:0000000128 EndHTML:0000008295 StartFragment:0000000128 EndFragment:0000008295 SourceURL:about:blank
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression, RidgeClassifier


def data_input(target_group, filepath) -> tuple:
    value_df = pd.read_csv(filepath)
    target_col = []
    labels = []
    for col in value_df.columns:
        if value_df[col][0] in target_group.keys():
            target_col.append(col)
            labels.append(target_group[value_df[col][0]])
    features = np.array(value_df[target_col][1:]).astype(np.float64).T
    labels = np.array(labels).astype(np.int64).ravel()
    feature_names = np.array(value_df["Samples"][1:]).astype(np.str_)
    return features, labels, feature_names


def missing_value_imputation(matrix, method):
    if method == "min_col":
        ratio = 5  # min/imputation
        for col in range(matrix.shape[1]):
            min_col = min(filter(lambda x: x > 0, matrix[:, col]))
            matrix[np.where(matrix[:, col] == 0), col] = min_col / ratio
        return matrix


def get_feature_matrix(filepath, analyte_name, target_group) -> tuple:
    # data input
    features, labels, feature_names = data_input(target_group, filepath)
    ori_feature_nums = len(feature_names)
    # missing value processing
    all_missing_features = np.where(~features.any(axis=0))[0]
    features = np.delete(features, all_missing_features, axis=1)
    feature_names = np.delete(feature_names, all_missing_features)
    features = missing_value_imputation(features, "min_col")
    print(f"Analyte name: {analyte_name}, samples found: {features.shape[0]}")
    print(
        f"{len(all_missing_features)} of {ori_feature_nums} features are removed because of the missing value among all samples\n"
    )
    return features, labels, feature_names


def stratified_split(features, labels, test_size, random_state=1) -> tuple:
    split = StratifiedShuffleSplit(n_splits=1,
                                   test_size=test_size,
                                   random_state=random_state)
    for train_index, test_index in split.split(features, labels):
        X_train, Y_train = features[train_index], labels[train_index]
        X_test, Y_test = features[test_index], labels[test_index]
    return X_train, X_test, Y_train, Y_test


# def my_RFE_model(name):
#     if name == "logistic":
#         return LogisticRegression(n_jobs=1)
#     elif name == "ridge":
#         return RidgeClassifier()
