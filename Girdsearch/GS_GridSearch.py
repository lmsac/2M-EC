import time

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (RFECV, SelectFromModel)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (LogisticRegression, LinearRegression,
                                  SGDClassifier, RidgeClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (GridSearchCV, LeaveOneOut,
                                     StratifiedShuffleSplit)
from sklearn.metrics import (accuracy_score, roc_auc_score)


# Regression Models
def LR_Regression_test():
    return make_pipeline(StandardScaler(), LinearRegression())


def PCA_LR_Regression_test():
    return make_pipeline(StandardScaler(), PCA(20), LinearRegression())


def RF_Regression_test():
    return make_pipeline(StandardScaler(), RandomForestClassifier())


# Categorical Models
def PCA_LR_Model_test():
    return make_pipeline(StandardScaler(), PCA(),
                         LogisticRegression(max_iter=1000))


def SVM_Model_test():
    return make_pipeline(StandardScaler(), SGDClassifier())


def L1_Norm_SVM_Model_test():
    return make_pipeline(StandardScaler(),
                         SGDClassifier(alpha=0.05, penalty="l1"))


# L1-Norm feature elimination with RandomForestClassifier.
def L1_Norm_RF_Model_test():
    return make_pipeline(
        StandardScaler(),
        SelectFromModel(SGDClassifier(alpha=0.05, penalty="l1")),
        RandomForestClassifier())


# L1-Norm feature elimination with MultiLayerPerceptron.
def L1_Norm_MLP_Model_test():
    return make_pipeline(
        StandardScaler(),
        SelectFromModel(SGDClassifier(alpha=0.05, penalty="l1")),
        MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64, 32),
                      max_iter=1000))


# Recursive Feature Elimination with cross-validation.
def RFE_LR_Model_test():
    return make_pipeline(StandardScaler(),
                         RFECV(LogisticRegression(max_iter=1000), step=0.2))


# Recursive Feature Elimination with cross-validation.
def RFE_RF_Model_test():
    return make_pipeline(StandardScaler(),
                         RFECV(RandomForestClassifier(), step=0.2))


# Recursive Feature Elimination with cross-validation.
def RFE_RG_Model_test():
    return make_pipeline(StandardScaler(), RFECV(RidgeClassifier(), step=0.2))


def model_param_grids() -> dict:
    param_grids_dict = {
        # LR_Regression_test: {},
        # PCA_LR_Regression_test: {
        #     "pca__n_components": [10, 20, 30, 40, 50]
        # },
        # RF_Regression_test: {
        #     'randomforestclassifier__n_estimators': range(10, 101, 10),
        #     'randomforestclassifier__max_depth': range(3, 10, 1),
        #     'randomforestclassifier__min_samples_split': range(10, 101, 10)
        # },
        PCA_LR_Model_test: {
            'pca__n_components': range(5, 41, 5),
            'logisticregression__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        },
        SVM_Model_test: {
            'sgdclassifier__penalty': ['l1', 'l2'],
            'sgdclassifier__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        },
        # L1_Norm_SVM_Model_test: {},
        # L1_Norm_RF_Model_test: {},
        # L1_Norm_MLP_Model_test: {},
        RFE_LR_Model_test: {
            'rfecv__estimator__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        },
        # RFE_RF_Model_test: {
        #     "rfecv__estimator__n_estimators": range(10, 101, 10),
        #     'rfecv__estimator__max_depth': range(3, 10, 1),
        #     'rfecv__estimator__min_samples_split': range(10, 101, 10, 100)
        # },
        RFE_RG_Model_test: {
            'rfecv__estimator__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }
    }
    return param_grids_dict


def GridSearch(features, labels, param_grids_dict) -> dict:
    print("----------GridSearching----------")
    split = StratifiedShuffleSplit(n_splits=1,
                                   test_size=0.2,
                                   random_state=1)
    for train_index, test_index in split.split(features, labels):
        X_train, Y_train = features[train_index], labels[train_index]
        X_test, Y_test = features[test_index], labels[test_index]
    now_cnt = 0
    model_cnt = len(param_grids_dict)
    best_model_dict = {}
    for model_name, param_grid in param_grids_dict.items():
        t1 = time.time()
        now_cnt += 1
        model = model_name()
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   cv=LeaveOneOut(),
                                   return_train_score=True,
                                   n_jobs=10)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        train_accuracy = accuracy_score(Y_train, best_model.predict(X_train))
        train_auc = roc_auc_score(Y_train, best_model.predict(X_train))
        test_accuracy = accuracy_score(Y_test, best_model.predict(X_test))
        test_auc = roc_auc_score(Y_test, best_model.predict(X_test))
        best_model_dict[model_name] = {}
        best_model_dict[model_name]["best_params"] = grid_search.best_params_
        best_model_dict[model_name]["train_accuracy"] = train_accuracy
        best_model_dict[model_name]["train_auc"] = train_auc
        best_model_dict[model_name]["test_accuracy"] = test_accuracy
        best_model_dict[model_name]["test_auc"] = test_auc
        t2 = time.time()
        print(
            f"{now_cnt} of {model_cnt} finished, model: {model_name.__name__}, time: {(t2-t1):.2f}s"
        )
        print(
            f"Train accuracy: {train_accuracy:.4f}, Train auc: {train_auc:.4f}"
        )
        print(
            f"Test accuracy: {test_accuracy:.4f}, Test auc: {test_auc:.4f}"
        )
        print("Best params: ", grid_search.best_params_)
    return best_model_dict
