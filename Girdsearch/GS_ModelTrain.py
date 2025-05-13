# Version:1.0 StartHTML:0000000128 EndHTML:0000064756 StartFragment:0000000128 EndFragment:0000064756 SourceURL:about:blank
import time
from collections import Counter
from warnings import simplefilter
from tqdm.auto import tqdm
from rich.progress import Progress
import sys
import os

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler, MinMaxScaler)
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel)
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (LogisticRegression, LinearRegression,
                                  SGDClassifier, RidgeClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (cross_validate, cross_val_predict)
from sklearn.model_selection import LeaveOneOut

from ML_TopFeature import get_top_features

# metabolites
# ignore all future warnings
simplefilter(action='ignore')


def feature_selector():
    # estimator = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
    # estimator = MLPClassifier(solver='sgd', activation='relu', max_iter=20000, hidden_layer_sizes=(100, 100, 100, 100, 100), random_state=1)
    # estimator = KNeighborsClassifier(5)
    # estimator = svm.SVC(kernel='linear', Mode_C=100, class_weight='balanced')
    # estimator = svm.SVC(kernel='rbf', Mode_C=100, gamma=0.5)
    # estimator = svm.SVC(kernel='sigmoid', Mode_C=1000000, gamma=0.0001, coef0=-1.5)
    # estimator = svm.SVC(kernel='poly', degree=2, coef0=10)
    # estimator = XGBClassifier()
    # estimator = RandomForestClassifier()
    estimator = RidgeClassifier()
    # estimator = SGDClassifier ()
    rfe = RFE(estimator=estimator, n_features_to_select=30)
    return rfe


# 模型定义
def LR_Logic():
    return make_pipeline(
        StandardScaler(), feature_selector(),
        LogisticRegression(penalty='l1',
                           solver='liblinear',
                           class_weight='balanced'))


def LR_MLP():
    return make_pipeline(
        StandardScaler(), feature_selector(),
        MLPClassifier(solver='sgd', #'sgd','adam','lbfgs'
                      alpha=0.0001,  #l2[0.0001, 0.001, 0.01]
                      # activation='relu',
                      max_iter=1000,   #200, 400, 600
                      learning_rate='constant', #['adaptive', 'invscaling', 'constant']
                      hidden_layer_sizes=(500, 500, 500, 500), #[(100,), (50, 50), (100, 100)]
                      random_state=1))


def LR_KNei():
    return make_pipeline(StandardScaler(), feature_selector(),
                         KNeighborsClassifier(5))


def LR_SVM_Lin():
    return make_pipeline(
        StandardScaler(), feature_selector(),
        svm.SVC(kernel='linear', C=100, class_weight='balanced'))


def LR_SVM_RBF():
    return make_pipeline(StandardScaler(), feature_selector(),
                         svm.SVC(kernel='rbf', C=100, gamma=0.5))


def LR_SVM_Sig():
    return make_pipeline(
        StandardScaler(), feature_selector(),
        svm.SVC(kernel='sigmoid', C=1000000, gamma=0.0001, coef0=-1.5))


def LR_SVM_Poly():
    return make_pipeline(StandardScaler(), feature_selector(),
                         svm.SVC(kernel='poly', degree=2, coef0=10))


def LR_GBDT():
    return make_pipeline(StandardScaler(), feature_selector(),
                         GradientBoostingClassifier(random_state=10))


def LR_XGB():
    return make_pipeline(StandardScaler(), feature_selector(), XGBClassifier())


def LR_RF():
    return make_pipeline(StandardScaler(), feature_selector(),
                         RandomForestClassifier())


def LR_Regression():
    return make_pipeline(StandardScaler(), LinearRegression())


def PCA_LR_Regression():
    return make_pipeline(StandardScaler(), PCA(20), LinearRegression())


def RF_Regression():
    return make_pipeline(StandardScaler(), RandomForestClassifier())


# Categorical Models
def PCA_LR_Model():
    return make_pipeline(StandardScaler(), PCA(7),
                         LogisticRegression(max_iter=1000))


def SVM_Model():
    return make_pipeline(StandardScaler(), SGDClassifier())


def L1_Norm_SVM_Model():
    return make_pipeline(StandardScaler(),
                         SGDClassifier(alpha=0.05, penalty="l1"))


# L1-Norm feature elimination with RandomForestClassifier.
def L1_Norm_RF_Model():
    return make_pipeline(
        StandardScaler(),
        SelectFromModel(SGDClassifier(alpha=0.05, penalty="l1")),
        RandomForestClassifier())


# L1-Norm feature elimination with LR. yigai
def L1_Norm_LR_Model():
    return make_pipeline(
        StandardScaler(),
        SelectFromModel(
            LogisticRegression(penalty='l1',
                               solver='liblinear',
                               class_weight='balanced')),
        LogisticRegression(penalty='l1',
                           solver='liblinear',
                           class_weight='balanced'))


# L1-Norm feature elimination with MultiLayerPerceptron. MLP修改
def L1_Norm_MLP_Model():
    return make_pipeline(
        StandardScaler(),
        SelectFromModel(SGDClassifier(alpha=0.05, penalty="l1")),
        MLPClassifier(solver='sgd',
                      activation='relu',
                      max_iter=20000,
                      hidden_layer_sizes=(100, 100, 100, 100, 100),
                      random_state=1))


# Recursive Feature Elimination with cross-validation. LR修改
def RFE_LR_Model():
    return make_pipeline(
        StandardScaler(),
        RFECV(LogisticRegression(penalty='l1',
                                 solver='liblinear',
                                 class_weight='balanced'),
              step=0.2))


# Recursive Feature Elimination with cross-validation.
def RFE_RF_Model():
    return make_pipeline(StandardScaler(),
                         RFECV(RandomForestClassifier(), step=0.2))


# Recursive Feature Elimination with cross-validation.
def RFE_RG_Model():
    return make_pipeline(StandardScaler(), RFECV(RidgeClassifier(), step=0.2))


def activated_model() -> tuple:
    return (
         LR_Logic,
         LR_MLP,
         LR_KNei,
         LR_SVM_Lin,
         LR_SVM_RBF,
         LR_SVM_Sig,
         LR_SVM_Poly,
         LR_GBDT,
         LR_XGB,
         LR_RF,
         RF_Regression,
        # PCA_LR_Model,
         SVM_Model,
         L1_Norm_SVM_Model,
         L1_Norm_RF_Model,
         L1_Norm_LR_Model,
        # L1_Norm_MLP_Model,
         RFE_LR_Model,
        # RFE_RF_Model,
         RFE_RG_Model,
    )


def cross_validation_LOO(model, features, labels) -> dict:
    scores_dict = cross_validate(
        estimator=model,
        X=features,
        y=labels,
        cv=LeaveOneOut(),
        scoring="balanced_accuracy",
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        return_train_score=True,
        return_estimator=True,
    )
    return scores_dict


def cross_validation_5CV(model, features, labels) -> dict:
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    scores_dict = cross_validate(
        estimator=model,  # 直接使用模型实例，不是 model()
        X=features,
        y=labels,
        cv=5,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
    )
    return scores_dict

def precision_and_recall_5CV(scores_dict, model, features, labels) -> dict:
    # labels_pred = cross_val_predict(estimator=model,
    #                                 X=features,
    #                                 y=labels,
    #                                 cv=5)
    # conf_matrix = confusion_matrix(labels, labels_pred, sample_weight=None)
    return {
        # "TP": conf_matrix[0, 0],
        # "FN": conf_matrix[1, 0],
        # "TN": conf_matrix[1, 1],
        # "FP": conf_matrix[0, 1],
        "accuracy": scores_dict["test_accuracy"].mean(),
        "precision": scores_dict["test_precision_macro"].mean(),
        "Sens": scores_dict["test_recall_macro"].mean(),
        "F1": scores_dict["test_f1_macro"].mean()
    }


def precision_and_recall(actual_labels, is_test_labels_correct) -> dict:
    import math
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for actual_label, correct in zip(actual_labels, is_test_labels_correct):
        if actual_label and correct:
            TP += 1
        if actual_label and not correct:
            FN += 1
        if not actual_label and correct:
            TN += 1
        if not actual_label and not correct:
            FP += 1
    try:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    except:
        accuracy = math.nan
    try:
        precision = TP / (TP + FP)
    except:
        precision = math.nan
    try:
        recall = TP / (TP + FN)
    except:
        recall = math.nan
    try:
        Spec = TN / (TN + FP)
    except:
        Spec = math.nan
    try:
        F1 = TP / (TP + ((FP + FN) / 2))
    except:
        F1 = math.nan
    try:
        PPV = TP / (TP + FP)
    except:
        PPV = math.nan
    try:
        NPV = TN / (TN + FN)
    except:
        NPV = math.nan
    return {
        "TP": TP,
        "FN": FN,
        "TN": TN,
        "FP": FP,
        "accuracy": accuracy,
        "precision": precision,
        "Sens": recall,
        "Spec": Spec,
        "F1": F1,
        "PPV": PPV,
        "NPV": NPV
    }


def ModelTrain(features, labels, feature_names, analyte_now, progress) -> dict:
    cross_method = "CV"  # choose "LOO" or "CV"
    return_dict = {}
    sample_nums = features.shape[0]
    try:
        activated_models = activated_model()
        task = progress.add_task(f"[green]Model", total=len(activated_model()))
    except:
        activated_models = tqdm(
            activated_model(),
            bar_format=
            '{desc} ({n_fmt}/{total_fmt}) |{bar}|{percentage:>3.0f}% {elapsed}',
            ascii=True,
            colour="green",
            ncols=80,
            desc=
            f"Process {progress[0]+1:>4d} of {progress[1][1]:>4d}, pid:{str(os.getpid()):>5s}",
            position=progress[0] % progress[1][0],
            file=sys.stdout,
            leave=False)
    for model_name in activated_models:
        model = model_name()
        if cross_method == "LOO":
            train_score_dict = cross_validation_LOO(model, features, labels)
            train_metrics_dict = precision_and_recall(
                actual_labels=labels,
                is_test_labels_correct=train_score_dict['test_score'])
        elif cross_method == "CV":
            train_score_dict = cross_validation_5CV(model, features, labels)
            train_metrics_dict = precision_and_recall_5CV(train_score_dict, model, features, labels)
        else:
            print("Undefined cross validation method!!!")
            sys.exit()
        return_dict[model_name.__name__] = {
            key: value
            for key, value in train_metrics_dict.items()
        }

        # get top features
        simple_model = [
            RF_Regression, PCA_LR_Model, SVM_Model, L1_Norm_SVM_Model
        ]
        SFM_model = [L1_Norm_RF_Model, L1_Norm_LR_Model]
        RFE_model = [RFE_LR_Model, RFE_RF_Model, RFE_RG_Model]
        FS_model = [
            LR_Logic, LR_MLP, LR_KNei, LR_SVM_Lin, LR_SVM_RBF, LR_SVM_Sig, LR_SVM_Poly,
            LR_GBDT, LR_XGB, LR_RF
        ]
        MLP_model = [L1_Norm_MLP_Model]

        if model_name in simple_model or model_name in SFM_model:
            n_top = 20
            feature_list = []
            all_weight_dict = Counter()
            for estimator in train_score_dict["estimator"]:
                if model_name in SFM_model:
                    chosen_index = estimator[1].get_support()
                    feature_names_now = feature_names[chosen_index]
                elif model_name in simple_model:
                    feature_names_now = feature_names
                fold_weight_dict = get_top_features(
                    model=estimator[-1],
                    feature_names=feature_names_now,
                    n_top=n_top)
                feature_list += list(fold_weight_dict.keys())
                all_weight_dict.update(fold_weight_dict)
            feature_freq = Counter(feature_list)
            mean_weight_dict = {
                feature_name: {
                    "frequency": freq / sample_nums,
                    "mean_weight": all_weight_dict[feature_name] / freq
                }
                for feature_name, freq in feature_freq.items()
            }
            i = 1
            for key, value in sorted(mean_weight_dict.items(),
                                     key=lambda x:
                                     (x[1]["frequency"], x[1]["mean_weight"]),
                                     reverse=True):
                return_dict[model_name.__name__][f"top_{i}_feature"] = {
                    "name": key,
                    "frequency": value["frequency"],
                    "mean_weight": value["mean_weight"]
                }
                i += 1
            # return_dict[model_name.__name__]["top_features"] = sorted(
            #     mean_weight_dict.items(),
            #     key=lambda x: (x[1]["frequency"], x[1]["mean_weight"]),
            #     reverse=True)

        # elif model_name in RFE_model:
        #     n_top = 20
        #     feature_list = []
        #     feature_names_now = feature_names
        #     for estimator in train_score_dict["estimator"]:
        #         # https://stackoverflow.com/questions/51181170/selecting-a-specific-number-of-features-via-sklearns-rfecv-recursive-feature-e
        #         feature_ranks = estimator[1].ranking_
        #         sorted_ranks_with_index = sorted(enumerate(feature_ranks),
        #                                          key=lambda x: x[1])
        #         top_n_index = [
        #             index for index, rank in sorted_ranks_with_index[:n_top]
        #         ]
        #         feature_list += list(feature_names_now[top_n_index])
        #     feature_freq = Counter(feature_list)
        #     mean_weight_dict = {
        #         feature_name: {
        #             "frequency": freq / sample_nums
        #         }
        #         for feature_name, freq in feature_freq.items()
        #     }
        #     i = 1
        #     for key, value in sorted(mean_weight_dict.items(),
        #                              key=lambda x: x[1]["frequency"],
        #                              reverse=True):
        #         return_dict[model_name.__name__][f"top_{i}_feature"] = {
        #             "name": key,
        #             "frequency": value["frequency"]
        #         }
        #         i += 1
        elif model_name in MLP_model or model_name in FS_model or model_name in RFE_model:
            feature_list = []
            for estimator in train_score_dict["estimator"]:
                if model_name in MLP_model:
                    chosen_index = estimator[1].get_support()
                elif model_name in FS_model or model_name in RFE_model:
                    chosen_index = estimator[1].support_
                feature_names_now = feature_names[chosen_index]
                feature_list += list(feature_names_now)
            feature_freq = Counter(feature_list)
            mean_weight_dict = {
                feature_name: {
                    "frequency": freq / sample_nums
                }
                for feature_name, freq in feature_freq.items()
            }
            i = 1
            for key, value in sorted(mean_weight_dict.items(),
                                     key=lambda x: x[1]["frequency"],
                                     reverse=True):
                return_dict[model_name.__name__][f"top_{i}_feature"] = {
                    "name": key,
                    "frequency": value["frequency"]
                }
                i += 1
        try:
            progress.update(task, advance=1)
            print(f"Analyte(s): {analyte_now}, model: {model_name.__name__}")
            print(train_score_dict)
        except:
            pass
    try:
        progress.update(task, visible=False)
    except:
        pass
    return return_dict


# def GridSearch(features, labels, param_grids_dict) -> dict:
#     print("----------GridSearching----------")
#     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
#     for train_index, test_index in split.split(features, labels):
#         X_train, Y_train = features[train_index], labels[train_index]
#         X_test, Y_test = features[test_index], labels[test_index]
#     now_cnt = 0
#     model_cnt = len(param_grids_dict)
#     best_model_dict = {}
#     for model_name, param_grid in param_grids_dict.items():
#         t1 = time.time()
#         now_cnt += 1
#         model = model_name()
#         grid_search = GridSearchCV(estimator=model,
#                                    param_grid=param_grid,
#                                    cv=LeaveOneOut(),
#                                    return_train_score=True,
#                                    n_jobs=10)
#         grid_search.fit(X_train, Y_train)
#         best_model = grid_search.best_estimator_
#         train_accuracy = accuracy_score(Y_train, best_model.predict(X_train))
#         train_auc = roc_auc_score(Y_train, best_model.predict(X_train))
#         test_accuracy = accuracy_score(Y_test, best_model.predict(X_test))
#         test_auc = roc_auc_score(Y_test, best_model.predict(X_test))
#         best_model_dict[model_name] = {}
#         best_model_dict[model_name]["best_params"] = grid_search.best_params_
#         best_model_dict[model_name]["train_accuracy"] = train_accuracy
#         best_model_dict[model_name]["train_auc"] = train_auc
#         best_model_dict[model_name]["test_accuracy"] = test_accuracy
#         best_model_dict[model_name]["test_auc"] = test_auc
#         t2 = time.time()
#         print(
#             f"{now_cnt} of {model_cnt} finished, model: {model_name.__name__}, time: {(t2-t1):.2f}s"
#         )
#         print(
#             f"Train accuracy: {train_accuracy:.4f}, Train auc: {train_auc:.4f}"
#         )
#         print(f"Test accuracy: {test_accuracy:.4f}, Test auc: {test_auc:.4f}")
#         print("Best params: ", grid_search.best_params_)
#     return best_model_dict
