import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import joblib
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from lime.lime_tabular import LimeTabularExplainer
import shap
from matplotlib.backends.backend_pdf import PdfPages
# import lime
# import lime.lime_tabular

scalers = {
    'C': joblib.load('scaler_standard_C.pkl'),
}

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('95CI.csv')

df_1 = df[['CI_age','CI_endometrial thickness','CI_HE4']]
df_2 = df[['CI_menopause', 'CI_HRT','CI_endometrial heterogeneity', 'CI_uterine cavity occupation','CI_uterine cavity occupying lesion with rich blood flow', 'CI_uterine cavity fluid']]

# 填补缺失值
df_1_filled = df_1.fillna(df_1.mean())
df_2_filled = df_2.fillna(df_2.mode().iloc[0])

# 合并数据
df3 = pd.concat([df_1_filled, df_2_filled], axis=1)

# 使用U标准化器对测试集df3进行标准化
df3_scaled = pd.DataFrame(scalers['C'].transform(df3), columns=df3.columns)

# 读取外部测试数据
data = pd.read_csv('95CM.csv')

# 提取样本名称（从 'Samples' 列提取）
sample_names = data['Samples']

# 提取指定列
columns_to_extract = [
    'CM4160.0','CM727.0','CM889.0','CM7441.0','CM995.0','CM7440.0','CM7439.0','CM734.0',
    'CM1857.0','CM6407.0','CM2920.0','CM729.0','CM628.0'
]
extracted_df = data[columns_to_extract]

# 合并数据集
df_concat = pd.concat([df3_scaled, extracted_df], axis=1)
df_concat['target'] = data['target']

# 提取特征和目标
y = df_concat['target']
x = df_concat.drop(['target'], axis=1)

# ————————————————加载模型————————————————
loaded_model = joblib.load('xgboost.pkl')  #catboost.pkl

# 对提取的特征数据进行预测
predictions = loaded_model.predict(x)

# 显示预测结果
print(predictions)

# 输出模型的完整评价指标
print(classification_report(np.array(y).astype(int), predictions))

# 输出并保存预测的样品名字、真实值和预测结果标签
result_df = pd.DataFrame({
    'Sample Name': sample_names,  # 使用正确的 'Samples' 列名
    'True Target': y,
    'Predicted Label': predictions
})
print(result_df)  # 打印输出结果
result_df.to_csv('prediction_results.csv', index=False)  # 保存到CSV文件

# 计算混淆矩阵
conf_matrix = confusion_matrix(y, predictions)

# 提取混淆矩阵的四个值：TN, FP, FN, TP
TN, FP, FN, TP = conf_matrix.ravel()

# 计算准确度、敏感度、特异度
accuracy = accuracy_score(y, predictions)
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # 避免除零错误
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0   # 避免除零错误
ppv = precision_score(y, predictions, zero_division=0)  # TP / (TP + FP), 处理除零情况
npv = TN / (TN + FN) if (TN + FN) > 0 else 0           # NPV, 阴性预测值
f1 = f1_score(y, predictions)  # F1 分数

# 将所有指标存储到一个 DataFrame 中
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 'PPV (Precision)', 'NPV', 'F1 Score'],
    'Value': [accuracy, sensitivity, specificity, ppv, npv, f1]
})

# 输出指标表格
print(metrics_df)

# 保存指标到CSV文件
metrics_df.to_csv('model_metrics.csv', index=False)
print("模型指标已保存到 model_metrics.csv 文件中。")

# 绘制ROC曲线
y_score = loaded_model.predict_proba(x)[:, 1]  # 获取阳性类的概率分数
fpr, tpr, _ = roc_curve(y, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("ROC曲线图.pdf", format='pdf', bbox_inches='tight')
plt.show()


# ————————————————构建 shap解释器————————————————
# 构建 shap解释器
explainer = shap.TreeExplainer(loaded_model)

# # 计算基础版shap值
shap_values = explainer.shap_values(x)
shap_values_numpy = explainer.shap_values(x)

# # 二分类模型基础版shap值计算
# #SHAP 生成的 shap_values 通常是一个列表（或者说数组）包含两个数组，每个数组是二维的.(n_samples, n_features),
# # shap_values_0 = shap_values[:,:,0] # 类别0的值
# # shap_values_1 = shap_values[:,:,1] # 类别1的值
# # 你可以直接通过 shap_values[0] 获取类别 0 的 SHAP 值，通过 shap_values[1] 获取类别 1 的 SHAP 值。
# shap_values_0 = shap_values[0]  # 类别 0 的 SHAP 值
# shap_values_1 = shap_values[1]  # 类别 1 的 SHAP 值

# # 计算进阶版shap值
shap_values_Explanation = explainer(x)

# 特征标签
labels = x.columns
print(shap_values.shape)

# ————————————————绘制单个样本的SHAP解释（Force Plot）————————————————
sample_index = 0  # 选择一个样本索引进行解释
shap.force_plot(explainer.expected_value, shap_values[sample_index], x.iloc[sample_index], matplotlib=True, show=False) # 如果对类比1做explainer.expected_value[0]修改为explainer.expected_value[1] shap_values_0[sample_index]修改为shap_values_1[sample_index]
plt.savefig("shap力图.pdf", bbox_inches='tight', dpi=1200)
plt.show()

force_plot = shap.force_plot(explainer.expected_value, shap_values_Explanation.values, x)

# # ————————————————单个样本决策图可视化————————————————
# # 获取模型的期望输出值（平均预测值）
# expected_value = explainer.expected_value
# # 选择第1个样本的 SHAP 值
# shap_values = shap_values_numpy[1]
# # 决策图的特征名展示
# features_display = x
# # 绘制 SHAP 决策图
# plt.figure(figsize=(10, 5), dpi=1200)
# shap.decision_plot(expected_value, shap_values, features_display, show=False)
# # 保存图像为 PDF
# plt.savefig("shap_decision_plot_samples.pdf", bbox_inches='tight')
# plt.tight_layout()


# ————————————————单个样本瀑布图可视化————————————————
plt.figure(figsize=(10, 5), dpi=1200)
# 绘制第1个样本的 SHAP 瀑布图，并设置 show=False 以避免直接显示
shap.plots.waterfall(shap_values_Explanation[72], show=False, max_display=13)
# 保存图像为 PDF 文件
plt.savefig("SHAP_Waterfall_Plot_Sample_1.pdf", format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()

# # # ————————————————特征散点图可视化————————————————
# # 指定特征的名称为 'cp'（胸痛类型）
# feature_name = 'CM5141.0'
# # 找到指定特征的索引
# feature_index = shap_values_Explanation.feature_names.index(feature_name)
# plt.figure(figsize=(10, 5), dpi=1200)
# # 使用 SHAP 的 scatter 方法绘制指定特征的散点图
# shap.plots.scatter(shap_values_Explanation[:, feature_index], show=False)
# plt.title(f'SHAP Scatter Plot for Feature: {feature_name}')
# plt.savefig(f"SHAP_Scatter_Plot_{feature_name}.pdf", format='pdf', bbox_inches='tight')
# plt.tight_layout()
# plt.show()
