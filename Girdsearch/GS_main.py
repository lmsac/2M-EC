import numpy as np
import os
import sys
import itertools
import pandas as pd
from multiprocessing import (Pool, cpu_count)
from functools import partial
from tqdm.auto import tqdm
from rich.progress import (Progress, BarColumn, SpinnerColumn,
                           TimeElapsedColumn)

from ML_DataProcessing import (get_feature_matrix, stratified_split)
from ML_GridSearch import GridSearch, model_param_grids
from ML_ModelTrain import ModelTrain
from ML_OuterTest import OuterTest
from ML_PlotFuncs import roc_plot


class LabelError(Exception):

    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo


def my_Progress():
    return Progress(
        "[progress.description]{task.description} ({task.completed}/{task.total})",
        SpinnerColumn(finished_text="[green]✔"), BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn())


def run_param(X_train,
              Y_train,
              merged_feature_names,
              param,
              progress=None) -> list:
    train_results = []
    try:
        progress, analyte_now, index_now = param
    except:
        analyte_now, index_now = param
    X_train_now = np.take(X_train, index_now, axis=1)
    feature_names_now = merged_feature_names[index_now]
    return_dict = ModelTrain(features=X_train_now,
                             labels=Y_train,
                             feature_names=feature_names_now,
                             analyte_now=analyte_now,
                             progress=progress)
    for model_name, infos in return_dict.items():
        train_results.append({})
        train_results[-1]["analytes"] = analyte_now
        train_results[-1]["model_name"] = model_name
        for key in infos.keys():
            train_results[-1][key] = infos[key]
    return train_results


if __name__ == "__main__":
    # global parameters
    MULTIPROCESS = True
    CPU_NUMS = 7
    is_train = True #cross交叉验证
    is_test = False #外部测试验证
    is_roc_plot = False
    # 文件夹位置
    os.chdir(
        r'/Users/apple/Desktop/FD/FD/子宫内膜癌EC/实验进展/maldi数据-2/EC-MALDI2/Data analysis/Model/UCPCI2'
    )
    # os.chdir(r'D:\LDD')

    # 标签命名
    target_group = {"CTRL": 0, "EC": 1}
    # 分析物命名，结果展现
    analyte_names = [
        "UCI",
        "UM",
        # "PP",
        # "PM",
        # "C",
    ]
    # 导入分析物的独立表格名称
    raw_csv = [
        # 'CI.csv',
        '201UCI.csv',
        '201UM.csv',

    ]
    # raw_csv = '20240118_ZJXJ_HPV_Combine_Proteins_Microonly_norm_MetaboAnalyst_uploading_CST1.csv'

    # data matrix input and merge
    print("\n----------Data Loading----------")
    cnt = 0
    analyte_index_dict = {}
    initialize = True
    for filepath, analyte_name in zip(raw_csv, analyte_names):
        features, labels, feature_names = get_feature_matrix(
            filepath=filepath,
            analyte_name=analyte_name,
            target_group=target_group)
        if initialize:
            merged_features = np.empty(shape=(features.shape[0],
                                              0)).astype(np.float64)
            first_labels = labels
            merged_feature_names = np.empty(shape=(features.shape[0],
                                                   0)).astype(np.str_)
            initialize = False
        elif ~(labels == first_labels).all():
            raise LabelError(
                'Labels in different analytes are different, please check the sample order.'
            )
        merged_features = np.append(merged_features, features, axis=1)
        merged_feature_names = np.append(merged_feature_names, feature_names)
        analyte_index_dict[analyte_name] = np.array(
            [i for i in range(cnt, cnt + features.shape[1])]).astype(np.int64)
        cnt += features.shape[1]

    # train and test split
    print("\n----------Train and Test Splitting----------")
    test_size = 0.00  # 训练集验证集的比例
    X_train, X_test, Y_train, Y_test = stratified_split(
        features=merged_features, labels=first_labels, test_size=test_size)
    print(f"{test_size:.0%} of the data will be used as TEST SET")

    # gridsearching for best params
    # best_params_dict = GridSearch(features=X_data,
    #                               labels=labels,
    #                               param_grids_dict=model_param_grids())

    # get combinations of analytes
    print("\n----------Analytes combining----------")
    analyte_coms = []
    index_coms = []
    least_n_features = 2  # 分析物的特征少于n不纳入
    for i in range(len(analyte_index_dict)):
        coms = list(itertools.combinations(analyte_index_dict.keys(), i + 1))
        analyte_coms += coms
        for com in coms:
            index = np.array([]).astype(np.int64)
            for analyte in com:
                index = np.append(index, analyte_index_dict[analyte])
            index_coms.append(index)
    low_features_analyte_index = list(
        map(lambda x: x.shape[0] < least_n_features, index_coms))
    low_features_analyte = list(
        itertools.compress(analyte_coms, low_features_analyte_index))
    enough_features_analyte_index = list(
        map(lambda x: x.shape[0] >= least_n_features, index_coms))
    analyte_coms = list(
        itertools.compress(analyte_coms, enough_features_analyte_index))
    index_coms = list(
        itertools.compress(index_coms, enough_features_analyte_index))
    if low_features_analyte != []:
        print(
            f"The following analyte combinations have features less than {least_n_features},\nwhich will not be used for model training."
        )
        for analyte in low_features_analyte:
            print(analyte)

    # model train
    if is_train:
        print("\n----------Model Training----------")
        all_train_results = []
        if (MULTIPROCESS and CPU_NUMS > 1):
            PARAMS = list(
                zip(
                    enumerate([(CPU_NUMS, len(analyte_coms))
                               for _ in range(len(analyte_coms))]),
                    analyte_coms, index_coms))
            with Pool(CPU_NUMS) as p:
                train_results = list(
                    p.imap(
                        partial(run_param, X_train, Y_train,
                                merged_feature_names), PARAMS))
            all_train_results = [
                item for sublist in train_results for item in sublist
            ]
        else:
            PARAMS = list(zip(analyte_coms, index_coms))
            with my_Progress() as progress:
                task = progress.add_task("[green]Analyte", total=len(PARAMS))
                for param in PARAMS:
                    all_train_results += run_param(
                        X_train=X_train,
                        Y_train=Y_train,
                        merged_feature_names=merged_feature_names,
                        param=param,
                        progress=progress)
                    progress.update(task, advance=1)
        train_df = pd.DataFrame(all_train_results)
        train_df.to_csv("./results_UCI.csv")

    # outer test
    if is_test:
        print("\n----------Outer Testing----------")
        all_test_results = []
        with my_Progress() as progress:
            task = progress.add_task("[green]Analyte", total=len(analyte_coms))
            for test_analyte_now, test_index_now in zip(
                    analyte_coms, index_coms):
                test_results = []
                X_fit_now = np.take(X_train, test_index_now, axis=1)
                X_test_now = np.take(X_test, test_index_now, axis=1)
                test_return_dict = OuterTest(features_fit=X_fit_now,
                                             labels_fit=Y_train,
                                             features_test=X_test_now,
                                             labels_test=Y_test,
                                             progress=progress)
                for model_name, infos in test_return_dict.items():
                    test_results.append({})
                    test_results[-1]["analytes"] = test_analyte_now
                    test_results[-1]["model_name"] = model_name
                    for key in infos.keys():
                        test_results[-1][key] = infos[key]
                all_test_results += test_results
                progress.update(task, advance=1)
        test_df = pd.DataFrame(all_test_results)
        test_df.to_csv("./results_Outest_RidgeClassifier.csv")
        # draw roc plot
        if is_roc_plot:
            for group_name, group_df in test_df.groupby(['analytes']):
                analyte_str = '&'.join(list(group_df["analytes"])[0])
                model_list = list(group_df["model_name"])
                fpr_tpr_df = group_df[group_df.columns[
                    group_df.columns.str.contains('fpr_|tpr_')]]
                fpr_list = []
                tpr_list = []
                for index, row in fpr_tpr_df.iterrows():
                    per_fpr_list = []
                    per_tpr_list = []
                    per_model_fpr_tpr: pd.Series = fpr_tpr_df.loc[index]
                    per_model_fpr_tpr.dropna(axis='index',
                                             how='all',
                                             inplace=True)
                    for key, value in per_model_fpr_tpr.items():
                        if "fpr_" in key:
                            per_fpr_list.append(value)
                        elif "tpr_" in key:
                            per_tpr_list.append(value)
                    fpr_list.append(per_fpr_list)
                    tpr_list.append(per_tpr_list)
                roc_plot(fpr_list, tpr_list, model_list, analyte_str)

    # # standardisation
    # print('----------Standardize----------')
    # X_scaler = StandardScaler()
    # X_std = X_scaler.fit_transform(X_data)

    # # recursive feature elimination
    # print('----------Feature Elimination----------')
    # t1 = time.time()
    # model = my_RFE_model(RFE_model)
    # rfe = RFE(model, n_features_to_select=RFE_features)
    # X_selected = rfe.fit_transform(X_std, labels)
    # t2 = time.time()
    # print("Time: %.2fs" % (t2 - t1))
    # print(rfe.get_support(indices=True))
