import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import smote_variants as sv
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
import joblib
from sklearn.model_selection import train_test_split

##paramo

# 忽略警告
warnings.filterwarnings("ignore")

# 加载数据
df = pd.read_csv('97CM.csv')

# 划分特征和目标变量
X = df.drop(['Samples', 'target'], axis=1)
y = df['target']

# 划分训练集和测试集索引并保存
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=61, stratify=df['target']
)
np.save("train_idx.npy", train_idx)
np.save("test_idx.npy", test_idx)

# 使用索引划分数据集
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

#——————————————————————————————————随机森林，查看特征贡献度top50——————————————

# 创建随机森林分类器实例
rf_classifier = RandomForestClassifier(
    random_state=61,
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
    cv=5,                 # 5折交叉验证                                  #paramo
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

# 获取特征的重要性
feature_importances = best_rf_classifier.feature_importances_

# 将特征和其重要性一起排序
sorted_indices = np.argsort(feature_importances)[::-1]  # 逆序排列，重要性从高到低
sorted_features = X_train.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# 打印排序后的特征及其重要性
for feature_name, importance in zip(sorted_features, sorted_importances):
    print(f"Feature: {feature_name}, Importance: {importance:.4f}")

# 选择前50个重要的特征
top_n = 100                  #paramo
top_features = sorted_features[:top_n]
top_importances = sorted_importances[:top_n]

# 打印前50个特征及其重要性
print("\n前50个特征的重要性：")
for feature_name, importance in zip(top_features, top_importances):
    print(f"Feature: {feature_name}, Importance: {importance:.4f}")

# 绘制按重要性排序的前50个特征贡献柱状图
plt.figure(figsize=(10, 8), dpi=1200)
plt.barh(top_features, top_importances, color='steelblue')
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.title('Top 50 Feature Importance', fontsize=16)
plt.gca().invert_yaxis()
plt.savefig("Top 50 Feature Importance.pdf", bbox_inches='tight')

# 显示图表
plt.show()


#——————————————————————————————————选取top50特征，蒙特卡洛模拟最佳特征数量——————————————
# 从原始数据 df 中提取前50个特征，并按重要性顺序进行列排序
df_top50 = df[top_features].copy()

# 输出前50个特征的DataFrame
df_top50.head()

# 划分特征和目标变量
X = df_top50
y = df['target']

# 设置随机种子
np.random.seed(42)
n_features = X.shape[1]
mc_no = 20  # 蒙特卡洛模拟的次数
cv_scores = np.zeros(n_features)  # 记录交叉验证分数

# 获取最佳模型的所有参数
best_params_rf = best_rf_classifier.get_params()

# 过滤出你感兴趣的参数，并组合默认参数和最佳参数
params_rf = {
    'n_estimators': best_params_rf['n_estimators'],  # 从网格搜索最佳模型获取
    'max_depth': best_params_rf['max_depth'],  # 从网格搜索最佳模型获取
    'min_samples_split': best_params_rf['min_samples_split'],  # 之前定义的默认值
    'min_samples_leaf': best_params_rf['min_samples_leaf'],  # 之前定义的默认值
    'criterion': best_params_rf['criterion'],  # 之前定义的默认值
    'random_state': best_params_rf['random_state'],  # 之前定义的随机种子
}

# 输出最佳参数
print("Best Random Forest parameters: ", params_rf)

# 使用这些参数重新创建模型
model_rf = RandomForestClassifier(**params_rf)

# 蒙特卡洛模拟
for j in np.arange(mc_no):
    # 每次模拟都重新划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=j)     #paramo

    # 逐步增加特征数量并进行交叉验证
    for i in range(1, n_features + 1):
        X_train_subset = X_train.iloc[:, :i]
        scores = cross_val_score(model_rf, X_train_subset, y_train, cv=5, scoring='accuracy', n_jobs=-1)       #paramo
        cv_scores[i - 1] += scores.mean()

# 计算平均交叉验证分数
cv_scores /= mc_no

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, n_features + 1), cv_scores)
plt.xlabel('Number of features selected')
plt.ylabel('Cross validation score (correct classifications)')
plt.title('Feature Selection Impact on Model Performance (with Cross Validation)')
plt.grid(True)
plt.tight_layout()
plt.savefig("Feature Selection Impact on Model Performance.pdf", bbox_inches='tight')
plt.show()

# 找到最优的特征数
optimal_feature_count = np.argmax(cv_scores) + 1  # 获取最佳特征数（加1是因为索引从0开始）
optimal_features = X.columns[:optimal_feature_count]  # 获取最佳特征对应的列名

# 输出最优的特征数和特征名称
print("Optimal number of features:", optimal_feature_count)
print("Optimal features:", optimal_features.tolist())  # 输出最佳特征的名称列表
print("Best CV score:", cv_scores[optimal_feature_count - 1])  # 输出最佳交叉验证分数

optimal_feature_result = pd.DataFrame(optimal_features)
optimal_feature_result.to_csv('features_result.csv', index=False)

# 根据最佳特征列名提取原始数据中的对应列
df_optimal_features = df[optimal_features]
df_optimal_features

params_rf

X = df_optimal_features
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,     #paramo
                                                    random_state=42, stratify=df['target'])
# 使用之前提取的最佳参数构建随机森林模型
rf_classifier = RandomForestClassifier(**params_rf)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
pred_rf = rf_classifier.predict(X_test)

# 输出模型的完整评价指标
print(classification_report(y_test, pred_rf))


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


#——————————————————————————————————rf树模型——————————————
X_resampled_df = pd.DataFrame(X_resampled, columns=df_optimal_features.columns)
X_resampled_df

X = X_resampled_df
y = y_resampled
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,    #paramo
                                                    random_state=42, stratify=y_resampled)
# 使用之前提取的最佳参数构建随机森林模型
rf_classifier_smote = RandomForestClassifier(**params_rf)

# 训练模型
rf_classifier_smote.fit(X_train, y_train)

# 在测试集上进行预测
pred_rf = rf_classifier_smote.predict(X_test)

# 输出模型的完整评价指标
print(classification_report(y_test, pred_rf))

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, pred_rf)

# 绘制热力图
plt.figure(figsize=(10, 7), dpi=1200)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':15}, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig("随机森林混淆矩阵图.pdf", bbox_inches='tight')
plt.show()

# 预测概率
y_score_1 = rf_classifier_smote.predict_proba(X_test)[:, 1]

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
plt.savefig("随机森林ROC曲线图.pdf", bbox_inches='tight')
plt.show()

#——————————————————————————————————xgboost树模型——————————————
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
    cv=5,                    # 5折交叉验证                          #paramo
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
plt.savefig("XGBoost混淆矩阵图.pdf", bbox_inches='tight')
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
plt.savefig("XGBoostROC曲线图.pdf", bbox_inches='tight')
plt.show()

#——————————————————————————————————CatBoost树模型——————————————
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
    cv=5,                    # 5折交叉验证                          #paramo
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
plt.savefig("CatBoost混淆矩阵图.pdf", bbox_inches='tight')
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
plt.savefig("catboostROC曲线图.pdf", bbox_inches='tight')
plt.show()

#——————————————————————————————————Lightgbm树模型——————————————
# LightGBM模型参数
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
    cv=5,                    # 5折交叉验证                       #paramo
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
plt.savefig("lightgbm混淆矩阵图.pdf", bbox_inches='tight')
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
plt.savefig("lightgbmROC曲线图.pdf", bbox_inches='tight')
plt.show()

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
plt.savefig("总ROC曲线图.pdf", bbox_inches='tight')
plt.show()

# 保存模型
joblib.dump(rf_classifier_smote , 'rf.pkl')
joblib.dump(best_model_xgboost , 'xgboost.pkl')
joblib.dump(best_model_catboost , 'catboost.pkl')
joblib.dump(best_model_lightgbm , 'lightgbm.pkl')


