import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import joblib
import networkx as nx
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import smote_variants as sv

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('97CI.csv')
df

# cl_numeric_features = [
#     'CI_age', 'CI_height', 'CI_weight', 'CI_BMI', 'CI_triglycerides',
#     'CI_total cholesterol', 'CI_HDL', 'CI_LDL', 'CI_free fatty acids',
#     'CI_CA125', 'CI_HE4', 'CI_endometrial thickness'
# ]
# cl_categorical_features = [
#     'CI_menopause', 'CI_HRT', 'CI_diabetes', 'CI_hypertension',
#     'CI_endometrial heterogeneity', 'CI_uterine cavity occupation',
#     'CI_uterine cavity occupying lesion with rich blood flow', 'CI_uterine cavity fluid'


df_1 = df[['CI_age','CI_endometrial thickness','CI_HE4']]

#'CI_HE4', 'CI_age', 'CI_CA125', 'CI_weight','CI_height','CI_BMI','CI_triglycerides','CI_total cholesterol',
           # 'CI_free fatty acids','CI_endometrial thickness','CI_HDL','CI_LDL'
df_1

df_1.isnull().sum()

df_2 = df[['CI_menopause', 'CI_HRT','CI_endometrial heterogeneity', 'CI_uterine cavity occupation','CI_uterine cavity occupying lesion with rich blood flow', 'CI_uterine cavity fluid']]
df_2

df_2.isnull().sum()

# 'CI_menopause', 'CI_HRT', 'CI_diabetes', 'CI_hypertension','CI_endometrial heterogeneity',
#            'CI_uterine cavity occupation','CI_uterine cavity occupying lesion with rich blood flow', 'CI_uterine cavity fluid'

# 对每列的缺失值用该列的平均值进行填补
df_1_filled = df_1.fillna(df_1.mean())
df_1_filled

df_2_filled = df_2.fillna(df_2.mode().iloc[0])
df_2_filled

# 合并 df_1_filled 和 df_2_filled 为 df3
df3 = pd.concat([df_1_filled, df_2_filled], axis=1)

# 进行 MinMaxScaler 处理
scaler = StandardScaler()
df3_scaled = pd.DataFrame(scaler.fit_transform(df3), columns=df3.columns)

# 保存 scaler 对象到文件
joblib.dump(scaler, 'scaler_standard_C.pkl')

# ------------------------------------------------------------------------------------
data = pd.read_csv('97CM.csv')
data

# 要提取的列
columns_to_extract = [
    'CM4160.0','CM727.0','CM889.0','CM7441.0','CM995.0','CM7440.0','CM7439.0','CM734.0',
    'CM1857.0','CM6407.0','CM2920.0','CM729.0','CM628.0'
]

# ------------------------------------------------------------------------------------
# 提取指定列
extracted_df = data[columns_to_extract]
extracted_df

df_concat = pd.concat([df3_scaled, extracted_df], axis=1)     #paramo combine
df_concat['target'] = data['target']
df_concat


#———————————————————————————————划分数据集——————————————————————————————

# 划分特征和目标变量
X = df_concat.drop(['target'], axis=1)
y = df_concat['target']

# # 划分训练集和测试集
# # 注意：random_state=42 是为了确保结果的可复现性，并且针对该数据集进行了特定处理。读者在使用自己的数据时，可以自由修改此参数。
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
#                                                     stratify=df['target'])

# 加载保存的索引
train_idx = np.load("train_idx.npy")
test_idx = np.load("test_idx.npy")

# 使用索引划分数据集
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

#——————————————————————————————————采样——————————————
# 查看存在那些采样算法
print(sv.get_all_oversamplers())

print("应用算法前数据维度:")
print("样本数量:", X.shape[0])
print("特征数量:", X.shape[1])
print("类别分布:\n", pd.Series(y).value_counts())
# 如果 X 是 pandas DataFrame，转换为 numpy 数组
if isinstance(X, pd.DataFrame):
    X = X.values

# 如果 y 是 pandas Series，转换为 numpy 数组
if isinstance(y, pd.Series):
    y = y.values

# 使用 Borderline_SMOTE1 或 Borderline_SMOTE2
smote = sv.Borderline_SMOTE1()  # 或者使用 sv.Borderline_SMOTE2()
# smote = SMOTE()

# 对数据进行过采样
X_resampled, y_resampled = smote.sample(X, y)

# 打印采样后的数据维度
print("应用 SMOTE 后的数据维度:")
print("样本数量:", X_resampled.shape[0])
print("特征数量:", X_resampled.shape[1])
print("类别分布:\n", pd.Series(y_resampled).value_counts())

X_resampled_df = pd.DataFrame(X_resampled, columns=df_concat.drop(['target'], axis=1).columns)     #采样
X_resampled_df

X = X_resampled_df
y = y_resampled
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,    #paramo
                                                    random_state=42, stratify=y_resampled)

#———————————————————————————————rf树模型——————————————————————————————
# 创建随机森林分类器实例
rf_classifier = RandomForestClassifier(
    random_state=42,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion='gini'
)

# 定义参数网格，用于网格搜索
param_grid = {
    'n_estimators': [100, 200, 300],  # 森林中树的数量
    'max_depth': [None, 10, 20, 30],  # 每棵树的最大深度
}

# 使用GridSearchCV进行网格搜索和k折交叉验证
grid_search_rf = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid,
    scoring='accuracy',   # 评价指标，可以改为其他如'roc_auc', 'f1'等
    cv=5,                 # 5折交叉验证
    n_jobs=-1,            # 并行计算
    verbose=1             # 输出详细进度信息
)

# 训练模型
grid_search_rf.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search_rf.best_params_)
print("Best accuracy score: ", grid_search_rf.best_score_)

# 使用最优参数训练的模型
best_rf_classifier = grid_search_rf.best_estimator_

# 在测试集上进行预测
pred_rf = best_rf_classifier.predict(X_test)

# 输出模型的完整评价指标
print(classification_report(y_test, pred_rf))

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, pred_rf)

# 绘制热力图
plt.figure(figsize=(10, 7), dpi=1200)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':15}, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig("随机森林混淆矩阵图.pdf", format='pdf', bbox_inches='tight')
plt.show()

# 预测概率
y_score_1 = best_rf_classifier.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr_logistic_1, tpr_logistic_1, _ = roc_curve(y_test, y_score_1)
roc_auc_logistic_1 = auc(fpr_logistic_1, tpr_logistic_1)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr_logistic_1, tpr_logistic_1, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_logistic_1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("随机森林ROC曲线图.pdf", format='pdf', bbox_inches='tight')
plt.show()

#———————————————————————————————xgBoost——————————————————————————————
# XGBoost模型参数
params_xgb = {
    'learning_rate': 0.02,            # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'booster': 'gbtree',              # 提升方法，这里使用梯度提升树（Gradient Boosting Tree）
    'objective': 'binary:logistic',   # 损失函数，这里使用逻辑回归，用于二分类任务
    'max_leaves': 127,                # 每棵树的叶子节点数量，控制模型复杂度。较大值可以提高模型复杂度但可能导致过拟合
    'verbosity': 1,                   # 控制 XGBoost 输出信息的详细程度，0表示无输出，1表示输出进度信息
    'seed': 42,                       # 随机种子，用于重现模型的结果
    'nthread': -1,                    # 并行运算的线程数量，-1表示使用所有可用的CPU核心
    'colsample_bytree': 0.6,          # 每棵树随机选择的特征比例，用于增加模型的泛化能力
    'subsample': 0.7,                 # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
    'eval_metric': 'logloss'          # 评价指标，这里使用对数损失（logloss）
}


# 初始化XGBoost分类模型
model_xgb = xgb.XGBClassifier(**params_xgb)


# 定义参数网格，用于网格搜索
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # 树的数量
    'max_depth': [3, 4, 5, 6, 7],               # 树的深度
    'learning_rate': [0.01, 0.02, 0.05, 0.1],   # 学习率
}

# 使用GridSearchCV进行网格搜索和k折交叉验证
grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring='neg_log_loss',  # 评价指标为负对数损失
    cv=5,                    # 5折交叉验证
    n_jobs=-1,               # 并行计算
    verbose=1                # 输出详细进度信息
)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)
print("Best Log Loss score: ", -grid_search.best_score_)

# 使用最优参数训练模型
best_model_xgboost = grid_search.best_estimator_

# 对测试集进行预测
pred_xgboost = best_model_xgboost.predict(X_test)
# 输出模型报告， 查看评价指标
print(classification_report(y_test, pred_xgboost))

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, pred_xgboost)

# 绘制热力图
plt.figure(figsize=(10, 7), dpi=1200)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':15}, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig("XGBoost混淆矩阵图.pdf", format='pdf', bbox_inches='tight')
plt.show()

# 预测概率
y_score_2 = best_model_xgboost.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr_logistic_2, tpr_logistic_2, _ = roc_curve(y_test, y_score_2)
roc_auc_logistic_2 = auc(fpr_logistic_2, tpr_logistic_2)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr_logistic_2, tpr_logistic_2, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_logistic_2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("XGBoostROC曲线图.pdf", format='pdf', bbox_inches='tight')
plt.show()

#———————————————————————————————CatBoost——————————————————————————————
# CatBoost模型参数
params_catboost = {
    'learning_rate': 0.02,            # 学习率，控制每一步的步长，用于防止过拟合
    'depth': 6,                       # 树的深度，控制模型复杂度
    'loss_function': 'Logloss',       # 损失函数，这里使用对数损失，用于二分类任务
    'verbose': 100,                   # 控制 CatBoost 输出信息的详细程度，设置为 100 表示每 100 次迭代输出一次信息
    'random_seed': 42,                # 随机种子，用于重现模型的结果
    'thread_count': -1,               # 并行运算的线程数量，-1表示使用所有可用的CPU核心
    'subsample': 0.7,                 # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
    'l2_leaf_reg': 3.0                # L2正则化项的系数，用于防止过拟合
}

# 初始化CatBoost分类模型
model_catboost = CatBoostClassifier(**params_catboost)

# 定义参数网格，用于网格搜索
param_grid_catboost = {
    'iterations': [100, 200, 300, 400, 500],  # 迭代次数，相当于树的数量
    'depth': [3, 4, 5, 6, 7],                 # 树的深度
    'learning_rate': [0.01, 0.02, 0.05, 0.1], # 学习率
}

# 使用GridSearchCV进行网格搜索和k折交叉验证
grid_search_catboost = GridSearchCV(
    estimator=model_catboost,
    param_grid=param_grid_catboost,
    scoring='neg_log_loss',  # 评价指标为负对数损失
    cv=5,                    # 5折交叉验证
    n_jobs=-1,               # 并行计算
    verbose=1                # 输出详细进度信息
)

# 训练模型
grid_search_catboost.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search_catboost.best_params_)
print("Best Log Loss score: ", -grid_search_catboost.best_score_)

# 使用最优参数训练模型
best_model_catboost = grid_search_catboost.best_estimator_

# 对测试集进行预测
pred_catboost = best_model_catboost.predict(X_test)
# 输出模型报告， 查看评价指标
print(classification_report(y_test, pred_catboost))

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, pred_catboost)

# 绘制热力图
plt.figure(figsize=(10, 7), dpi=1200)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':15}, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig("CatBoost混淆矩阵图.pdf", format='pdf', bbox_inches='tight')
plt.show()

# 预测概率
y_score_3 = best_model_catboost.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr_logistic_3, tpr_logistic_3, _ = roc_curve(y_test, y_score_3)
roc_auc_logistic_3 = auc(fpr_logistic_3, tpr_logistic_3)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr_logistic_3, tpr_logistic_3, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_logistic_3)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("catboostROC曲线图.pdf", format='pdf', bbox_inches='tight')
plt.show()

#———————————————————————————————LightGBM——————————————————————————————
params_lgb = {
    'learning_rate': 0.02,            # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'boosting_type': 'gbdt',          # 提升方法，这里使用梯度提升决策树（Gradient Boosting Decision Tree）
    'objective': 'binary',            # 损失函数，这里使用二分类
    'num_leaves': 127,                # 每棵树的叶子节点数量，控制模型复杂度。较大值可以提高模型复杂度但可能导致过拟合
    'verbosity': 1,                   # 控制 LightGBM 输出信息的详细程度，0表示无输出，1表示输出进度信息
    'random_state': 42,               # 随机种子，用于重现模型的结果
    'n_jobs': -1,                     # 并行运算的线程数量，-1表示使用所有可用的CPU核心
    'colsample_bytree': 0.6,          # 每棵树随机选择的特征比例，用于增加模型的泛化能力
    'subsample': 0.7,                 # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
    'metric': 'binary_logloss'        # 评价指标，这里使用对数损失（binary_logloss）
}

# 初始化LightGBM分类模型
model_lgb = lgb.LGBMClassifier(**params_lgb)

# 定义参数网格，用于网格搜索
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # 树的数量
    'max_depth': [3, 4, 5, 6, 7],               # 树的深度
    'learning_rate': [0.01, 0.02, 0.05, 0.1],   # 学习率
}

# 使用GridSearchCV进行网格搜索和k折交叉验证
grid_search = GridSearchCV(
    estimator=model_lgb,
    param_grid=param_grid,
    scoring='neg_log_loss',  # 评价指标为负对数损失
    cv=5,                    # 5折交叉验证
    n_jobs=-1,               # 并行计算
    verbose=1                # 输出详细进度信息
)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)
print("Best Log Loss score: ", -grid_search.best_score_)

# 使用最优参数训练模型
best_model_lightgbm = grid_search.best_estimator_

# 对测试集进行预测
pred_lightgbm = best_model_lightgbm.predict(X_test)
# 输出模型报告， 查看评价指标
print(classification_report(y_test, pred_lightgbm))

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, pred_lightgbm)

# 绘制热力图
plt.figure(figsize=(10, 7), dpi=1200)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':15}, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig("lightgbm混淆矩阵图.pdf", format='pdf', bbox_inches='tight')
plt.show()

# 预测概率
y_score_4 = best_model_lightgbm.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr_logistic_4, tpr_logistic_4, _ = roc_curve(y_test, y_score_4)
roc_auc_logistic_4 = auc(fpr_logistic_4, tpr_logistic_4)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr_logistic_4, tpr_logistic_4, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_logistic_4)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("lightgbmROC曲线图.pdf", format='pdf', bbox_inches='tight')
plt.show()

#———————————————————————————————总ROC曲线——————————————————————————————
plt.figure()
plt.plot(fpr_logistic_1, tpr_logistic_1, color='darkorange', lw=2, label='RF ROC curve (area = %0.2f)' % roc_auc_logistic_1)
plt.plot(fpr_logistic_2, tpr_logistic_2, color='blue', lw=2, label='XGBoost ROC curve (area = %0.2f)' % roc_auc_logistic_2)
plt.plot(fpr_logistic_3, tpr_logistic_3, color='green', lw=2, label='CatBoost ROC curve (area = %0.2f)' % roc_auc_logistic_3)
plt.plot(fpr_logistic_4, tpr_logistic_4, color='red', lw=2, label='LightGBM ROC curve (area = %0.2f)' % roc_auc_logistic_4)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("总ROC曲线图.pdf", format='pdf', bbox_inches='tight')
plt.show()


#———————————————————————————————保存模型——————————————————————————————
# 保存模型
joblib.dump(best_rf_classifier , 'rf.pkl')
joblib.dump(best_model_xgboost , 'xgboost.pkl')
joblib.dump(best_model_catboost , 'catboost.pkl')
joblib.dump(best_model_lightgbm , 'lightgbm.pkl')

# 获取RandomForest特征重要性并排序
rf_feature_importances = best_rf_classifier.feature_importances_
rf_sorted_indices = np.argsort(rf_feature_importances)[::-1]
rf_sorted_features = X_train.columns[rf_sorted_indices]
rf_sorted_importances = rf_feature_importances[rf_sorted_indices]

# 获取XGBoost特征重要性并排序
xgb_feature_importances = best_model_xgboost.feature_importances_
xgb_sorted_indices = np.argsort(xgb_feature_importances)[::-1]
xgb_sorted_features = X_train.columns[xgb_sorted_indices]
xgb_sorted_importances = xgb_feature_importances[xgb_sorted_indices]

# 获取LightGBM特征重要性并排序
lgb_feature_importances = best_model_lightgbm.feature_importances_
lgb_sorted_indices = np.argsort(lgb_feature_importances)[::-1]
lgb_sorted_features = X_train.columns[lgb_sorted_indices]
lgb_sorted_importances = lgb_feature_importances[lgb_sorted_indices]

# 获取CatBoost特征重要性并排序
catboost_feature_importances = best_model_catboost.get_feature_importance()
catboost_sorted_indices = np.argsort(catboost_feature_importances)[::-1]
catboost_sorted_features = X_train.columns[catboost_sorted_indices]
catboost_sorted_importances = catboost_feature_importances[catboost_sorted_indices]

# 创建一个DataFrame来保存所有模型的特征重要性
feature_importance_df = pd.DataFrame({
    "RandomForest_Feature": rf_sorted_features,
    "RandomForest_Importance": rf_sorted_importances,
    "XGBoost_Feature": xgb_sorted_features,
    "XGBoost_Importance": xgb_sorted_importances,
    "LightGBM_Feature": lgb_sorted_features,
    "LightGBM_Importance": lgb_sorted_importances,
    "CatBoost_Feature": catboost_sorted_features,
    "CatBoost_Importance": catboost_sorted_importances
})
feature_importance_df


#———————————————————————————————绘制共有特征重要性排序——————————————————————————————
feature = feature_importance_df[['RandomForest_Feature', 'XGBoost_Feature', 'LightGBM_Feature', 'CatBoost_Feature']]

plt.figure(figsize=(12, 8), dpi=1200)

# 创建一个有向图
G = nx.DiGraph()
unique_features = pd.unique(feature.values.ravel('K'))

# 使用一个色系的渐变色
colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(unique_features)))
feature_color_map = {feature_name: colors[i % len(colors)] for i, feature_name in enumerate(unique_features)}
pos = {}
for i, model in enumerate(feature.columns):
    for j, feature_name in enumerate(feature[model]):
        # 添加节点
        G.add_node(f'{model}_{feature_name}', label=feature_name)
        pos[f'{model}_{feature_name}'] = (i, -j)
        if i > 0:
            previous_model = feature.columns[i - 1]
            for prev_j, prev_feature_name in enumerate(feature[previous_model]):
                if feature_name == prev_feature_name:
                    G.add_edge(f'{previous_model}_{prev_feature_name}', f'{model}_{feature_name}', color='black')  # 固定边的颜色为黑色

# 获取节点标签
node_labels = nx.get_node_attributes(G, 'label')

# 设置较小的节点大小和字体大小
node_size = 1000  # 将节点大小减小
font_size = 8     # 将字体大小也减小

# 通过颜色映射绘制节点
node_colors = [feature_color_map[node_labels[node]] for node in G.nodes()]
nx.draw(G, pos, labels=node_labels, with_labels=True, node_color=node_colors, node_size=node_size, font_size=font_size)

# 绘制边
edges = G.edges()
nx.draw_networkx_edges(G, pos, edge_color='black', arrowstyle='-|>', arrowsize=15, width=2)

# 在图上标注模型名称
for i, model in enumerate(feature.columns):
    plt.text(i, -len(feature), model, horizontalalignment='center', fontsize=12, fontweight='bold')

plt.gca().set_axis_off()

# 添加标题
plt.title('Feature Ranking Comparison Across Models with Gradual Color Scheme and Styled Arrows')

# 保存图像
plt.savefig("Feature Ranking Comparison Across Models with Gradual Color Scheme and Styled Arrows.pdf", bbox_inches='tight')

# 显示图像
plt.show()

feature_importance_df.to_excel("特征排名.xlsx", index=False)