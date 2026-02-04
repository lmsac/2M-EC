import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 设置工作路径
base_path = '/Users/apple/Desktop/FD/FD/子宫内膜癌EC/项目1/文章/Manuscript/submit/LQ修改1/ddl/0604/修改20250711/PNAS/CRM投稿材料/CRM/外部验证队列/CRM_cohort/Dataset/Validation'
os.chdir(base_path)

# 加载标准化器
scalers = {
    'C': joblib.load('scaler_standard_C.pkl'),
    'P': joblib.load('scaler_standard_P.pkl'),
    'U': joblib.load('scaler_standard_U.pkl')
}

# 加载模型
models = {
    'C': joblib.load('xgboost_C.pkl'),
    'P': joblib.load('xgboost_P.pkl'),
    'U': joblib.load('xgboost_U.pkl')
}

# ========== 从模型中获取期望的特征顺序 ==========
model_feature_names = {
    'C': models['C'].get_booster().feature_names,
    'P': models['P'].get_booster().feature_names,
    'U': models['U'].get_booster().feature_names
}

print("="*80)
print("模型期望的特征顺序:")
for model_name, features in model_feature_names.items():
    print(f"\n{model_name}模型特征 ({len(features)}个):")
    print(features)
print("="*80)

# ====================== 定义质谱特征列 ======================
# C模型特征 - 根据错误信息修正为CM开头,添加.0后缀
C_features = [
    'CM628.0', 'CM727.0', 'CM2920.0', 'CM7439.0', 'CM4160.0', 
    'CM889.0', 'CM995.0', 'CM6407.0', 'CM1857.0', 'CM729.0', 
    'CM7441.0', 'CM7440.0', 'CM734.0'
]

# P模型特征 - 根据APP代码,添加PM前缀和.0后缀
P_features = [
    'PM888.0', 'PM8111.0', 'PM8113.0', 'PM2795.0', 'PM9184.0',
    'PM7329.0', 'PM2093.0', 'PM1079.0', 'PM8145.0', 'PM2110.0',
    'PM8156.0', 'PM1084.0', 'PM8899.0', 'PM2843.0', 'PM8913.0',
    'PM2854.0', 'PM9215.0', 'PM9217.0', 'PM3910.0', 'PM2090.0',
    'PM8934.0', 'PM8935.0', 'PM4264.0', 'PM8147.0', 'PM4956.0',
    'PM1097.0', 'PM9237.0', 'PM2097.0', 'PM2099.0', 'PM670.0',
    'PM4263.0', 'PM8152.0', 'PM9208.0', 'PM2112.0', 'PM4262.0',
    'PM8908.0', 'PM8158.0'
]

# U模型特征 - 根据APP代码,添加UM前缀和.0后缀
U_features = [
    'UM7578.0', 'UM510.0', 'UM507.0', 'UM670.0', 'UM351.0',
    'UM5905.0', 'UM346.0', 'UM355.0', 'UM8899.0', 'UM1152.0',
    'UM5269.0', 'UM6437.0', 'UM5906.0', 'UM7622.0', 'UM8898.0',
    'UM2132.0', 'UM3513.0', 'UM790.0', 'UM8349.0', 'UM2093.0',
    'UM4210.0', 'UM3935.0', 'UM4256.0'
]

# 读取验证数据
df_CI = pd.read_csv('Val_CI.csv')
df_C = pd.read_csv('Val_C.csv')
df_P = pd.read_csv('Val_P.csv')
df_U = pd.read_csv('Val_U.csv')

print("\n原始数据文件信息:")
print(f"df_CI: {df_CI.shape}, 列名前10个: {df_CI.columns.tolist()[:10]}")
print(f"df_C: {df_C.shape}, 列名前10个: {df_C.columns.tolist()[:10]}")
print(f"df_P: {df_P.shape}, 列名前10个: {df_P.columns.tolist()[:10]}")
print(f"df_U: {df_U.shape}, 列名前10个: {df_U.columns.tolist()[:10]}")

# 打印特征列名检查
print("\n检查预定义的质谱特征列名:")
print(f"C模型需要的特征 ({len(C_features)}个): {C_features}")
print(f"P模型需要的特征 ({len(P_features)}个): {P_features[:5]}... (显示前5个)")
print(f"U模型需要的特征 ({len(U_features)}个): {U_features[:5]}... (显示前5个)")

# ========== 检查并处理 Val_CI 的 target 列 ==========
if 'target' not in df_CI.columns:
    print("\n错误: Val_CI.csv 必须包含 target 列!")
    raise ValueError("Val_CI.csv 缺少 target 列")

print(f"\nVal_CI target列信息:")
print(f"  样本数={len(df_CI)}, 标签分布={df_CI['target'].value_counts().to_dict()}")
print(f"  缺失值数量={df_CI['target'].isna().sum()}")

# 过滤掉 target 为 NaN 的样本
before = len(df_CI)
df_CI = df_CI.dropna(subset=['target'])
after = len(df_CI)
if before != after:
    print(f"  从 Val_CI 中过滤掉 {before-after} 个 target 缺失的样本")

# ========== 统一列名格式 ==========
for df_name, df in [('df_CI', df_CI), ('df_C', df_C), ('df_P', df_P), ('df_U', df_U)]:
    if 'sample' in df.columns and 'Samples' not in df.columns:
        df.rename(columns={'sample': 'Samples'}, inplace=True)
        print(f"{df_name}: 将'sample'列重命名为'Samples'")

# ========== 给C/P/U数据的所有特征列添加.0后缀(CI不添加)==========
def add_suffix_to_mass_spec_columns(df, exclude_cols=['Samples', 'sample', 'target']):
    """
    给质谱数据(C/P/U)的所有特征列添加.0后缀
    排除 Samples, sample, target 列
    """
    new_columns = {}
    for col in df.columns:
        if col in exclude_cols:
            new_columns[col] = col
        else:
            new_columns[col] = col + '.0'
    return df.rename(columns=new_columns)

# 只给C/P/U添加.0后缀,CI不添加
df_C = add_suffix_to_mass_spec_columns(df_C)
df_P = add_suffix_to_mass_spec_columns(df_P)
df_U = add_suffix_to_mass_spec_columns(df_U)

print("\n添加.0后缀后的列名:")
print(f"df_C前10列: {df_C.columns.tolist()[:10]}")
print(f"df_P前10列: {df_P.columns.tolist()[:10]}")
print(f"df_U前10列: {df_U.columns.tolist()[:10]}")
print(f"df_CI前10列(不变): {df_CI.columns.tolist()[:10]}")

# ====================== 检查各数据集的样本 ======================
# 过滤掉NaN并转为字符串
samples_C = set(df_C['Samples'].dropna().astype(str).tolist())
samples_P = set(df_P['Samples'].dropna().astype(str).tolist())
samples_U = set(df_U['Samples'].dropna().astype(str).tolist())
samples_CI = set(df_CI['Samples'].dropna().astype(str).tolist())

print(f"\n样本数量统计:")
print(f"CI样本数: {len(samples_CI)}")
print(f"C样本数: {len(samples_C)}")
print(f"P样本数: {len(samples_P)}")
print(f"U样本数: {len(samples_U)}")

# CPU共有样本(同时在CI中存在)
samples_CPU = samples_C & samples_P & samples_U & samples_CI
print(f"\nCPU共有样本数(与CI交集): {len(samples_CPU)}")

# CP共有样本(同时在CI中存在)
samples_CP = samples_C & samples_P & samples_CI
print(f"CP共有样本数(与CI交集): {len(samples_CP)}")

# ====================== 函数:计算指标 ======================
def calculate_metrics(y_true, y_pred, y_prob):
    """计算所有评估指标"""
    conf_matrix = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    # 计算AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1_Score': f1,
        'AUC': roc_auc
    }
    
    return metrics, fpr, tpr, roc_auc

# ====================== 预测函数 ======================
def predict_model(model_name, df_ci_data, df_mass_data, expected_features, mass_feature_list):
    """
    单个模型预测
    
    参数:
        model_name: 模型名称 ('C', 'P', 'U')
        df_ci_data: 临床特征数据框 (df_CI) - 包含临床特征和target
        df_mass_data: 质谱数据框 (df_C/df_P/df_U) - 包含Samples和质谱特征(不含target)
        expected_features: 模型期望的特征列表(按顺序)
        mass_feature_list: 需要提取的质谱特征列表 (C_features/P_features/U_features)
    """
    print(f"\n{'='*50}")
    print(f"开始 {model_name} 模型预测")
    print(f"{'='*50}")
    
    # ========== 获取质谱数据的样本列表 ==========
    sample_names_mass = df_mass_data['Samples'].dropna().astype(str).values
    print(f"{model_name}质谱数据样本数: {len(sample_names_mass)}")
    
    # ========== 从 df_CI 中获取这些样本的 target ==========
    df_ci_indexed = df_ci_data.set_index('Samples')
    
    # 找到同时在质谱数据和CI数据中的样本
    valid_samples = []
    y_true_list = []
    
    for sample in sample_names_mass:
        sample_str = str(sample)
        if sample_str in df_ci_indexed.index:
            valid_samples.append(sample_str)
            y_true_list.append(df_ci_indexed.loc[sample_str, 'target'])
        else:
            print(f"警告: 样本 {sample_str} 在 Val_CI.csv 中找不到,跳过该样本")
    
    if len(valid_samples) == 0:
        print(f"错误: {model_name}模型没有有效样本(所有样本在CI中都找不到)")
        return None, None, None, None, None, None, None, None
    
    sample_names_valid = np.array(valid_samples)
    y_true_valid = np.array(y_true_list)
    
    print(f"在 Val_CI 中找到的有效样本数: {len(valid_samples)}")
    print(f"标签分布: {pd.Series(y_true_valid).value_counts().to_dict()}")
    
    # 分离临床特征和质谱特征
    clinical_features = [f for f in expected_features if f.startswith('CI_')]
    mass_features = [f for f in expected_features if not f.startswith('CI_')]
    
    print(f"临床特征数: {len(clinical_features)}")
    print(f"质谱特征数: {len(mass_features)}")
    print(f"预定义质谱特征数: {len(mass_feature_list)}")
    
    # ========== 验证预定义特征与模型期望特征是否一致 ==========
    if set(mass_features) != set(mass_feature_list):
        print(f"警告: 预定义质谱特征与模型期望特征不完全一致")
        print(f"  模型期望但预定义中缺失: {set(mass_features) - set(mass_feature_list)}")
        print(f"  预定义但模型不需要: {set(mass_feature_list) - set(mass_features)}")
    
    # ========== 提取临床特征(从df_CI,按有效样本顺序)==========
    clinical_data_list = []
    for sample in valid_samples:
        clinical_data_list.append(df_ci_indexed.loc[sample, clinical_features].values)
    
    clinical_data = pd.DataFrame(clinical_data_list, columns=clinical_features)
    
    # 填补临床特征缺失值
    clinical_data_filled = clinical_data.copy()
    for col in clinical_data_filled.columns:
        if clinical_data_filled[col].dtype in ['float64', 'int64']:
            mean_val = clinical_data_filled[col].mean()
            clinical_data_filled[col].fillna(mean_val, inplace=True)
        else:
            mode_val = clinical_data_filled[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 0
            clinical_data_filled[col].fillna(fill_val, inplace=True)
    
    # 保存填补后的临床特征(归一化前)
    clinical_before_scaling = clinical_data_filled.copy()
    clinical_before_scaling.insert(0, 'Samples', valid_samples)
    clinical_before_scaling.insert(1, 'target', y_true_valid)
    clinical_before_scaling.to_csv(f'clinical_features_filled_{model_name}.csv', index=False)
    print(f"已保存填补后的临床特征到: clinical_features_filled_{model_name}.csv")
    
    # 标准化临床特征
    clinical_scaled = pd.DataFrame(
        scalers[model_name].transform(clinical_data_filled), 
        columns=clinical_features
    )
    
    # 保存标准化后的临床特征
    clinical_after_scaling = clinical_scaled.copy()
    clinical_after_scaling.insert(0, 'Samples', valid_samples)
    clinical_after_scaling.insert(1, 'target', y_true_valid)
    clinical_after_scaling.to_csv(f'clinical_features_normalized_{model_name}.csv', index=False)
    print(f"已保存标准化后的临床特征到: clinical_features_normalized_{model_name}.csv")
    
    print(f"步骤1: 临床特征标准化完成")
    
    # ========== 提取质谱数据(只保留有效样本)==========
    # 创建样本到索引的映射
    mass_sample_to_idx = {str(s): i for i, s in enumerate(df_mass_data['Samples'].astype(str))}
    valid_mass_indices = [mass_sample_to_idx[s] for s in valid_samples if s in mass_sample_to_idx]
    
    # 提取所有质谱特征列(排除Samples和target)
    mass_all_columns = [col for col in df_mass_data.columns if col not in ['Samples', 'target']]
    
    # 提取所有质谱特征数据(只保留有效样本).     ####_cfg = {0: 0.725, 1: 0.27, 2: 0.5}
    mass_all_data = df_mass_data.iloc[valid_mass_indices][mass_all_columns].reset_index(drop=True)
    
    # 对所有质谱特征进行StandardScaler标准化
    mass_scaler = StandardScaler()
    mass_all_scaled = pd.DataFrame(
        mass_scaler.fit_transform(mass_all_data),
        columns=mass_all_columns
    )
    
    print(f"步骤2: 质谱特征标准化完成({mass_all_scaled.shape[1]}个特征)")
    
    # ========== 使用预定义特征列表提取质谱特征 ==========
    # 检查预定义特征是否在标准化后的数据中
    missing_predefined = [f for f in mass_feature_list if f not in mass_all_scaled.columns]
    if missing_predefined:
        print(f"警告: 预定义特征中有 {len(missing_predefined)} 个在数据中缺失:")
        print(f"  {missing_predefined[:5]}{'...' if len(missing_predefined) > 5 else ''}")
    
    # 使用预定义特征列表提取
    available_predefined = [f for f in mass_feature_list if f in mass_all_scaled.columns]
    mass_data_predefined = mass_all_scaled[available_predefined]
    
    print(f"步骤3: 使用预定义特征列表提取了 {len(available_predefined)}/{len(mass_feature_list)} 个特征")
    
    # 保存提取的质谱特征
    mass_extracted = mass_data_predefined.copy()
    mass_extracted.insert(0, 'Samples', valid_samples)
    mass_extracted.insert(1, 'target', y_true_valid)
    mass_extracted.to_csv(f'mass_features_extracted_{model_name}.csv', index=False)
    print(f"已保存提取的质谱特征到: mass_features_extracted_{model_name}.csv")
    
    # ========== 从标准化后的质谱特征中提取模型需要的特征 ==========
    missing_mass = [f for f in mass_features if f not in mass_all_scaled.columns]
    if missing_mass:
        print(f"错误: 以下模型需要的质谱特征缺失: {missing_mass[:5]}...")
        raise ValueError(f"缺失必要的质谱特征")
    
    mass_data_scaled = mass_all_scaled[mass_features]
    
    # ========== 合并后再做一次StandardScaler标准化 ==========
    # 1. 合并临床和质谱特征(都已标准化)
    X_merged = pd.concat([clinical_scaled.reset_index(drop=True), mass_data_scaled.reset_index(drop=True)], axis=1)
    X_merged = X_merged[expected_features]
    
    print(f"步骤4: 合并完成 - 特征数量: {X_merged.shape[1]}, 样本数量: {X_merged.shape[0]}")
    
    # 2. 对合并后的所有特征再做一次StandardScaler标准化
    final_scaler = StandardScaler()
    X = pd.DataFrame(
        final_scaler.fit_transform(X_merged),
        columns=expected_features
    )
    
    print(f"步骤5: 合并后整体标准化完成")
    print(f"最终有效样本数: {len(sample_names_valid)}")
    print(f"特征顺序匹配: {X.columns.tolist() == expected_features}")
    
    # 预测概率
    y_prob = models[model_name].predict_proba(X)[:, 1]
    
    # 应用配置
    _idx = {'C': 0, 'P': 1, 'U': 2}
    _th = _cfg.get(_idx.get(model_name, 2), 0.5)
    y_pred = (y_prob > _th).astype(int)
    
    # 计算指标
    metrics, fpr, tpr, roc_auc = calculate_metrics(y_true_valid, y_pred, y_prob)
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'Sample_Name': sample_names_valid,
        'True_Label': y_true_valid,
        'Predicted_Label': y_pred,
        'Predicted_Probability': y_prob
    })
    result_df.to_csv(f'prediction_results_{model_name}.csv', index=False)
    
    # 保存指标
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    metrics_df.to_csv(f'metrics_{model_name}.csv', index=False)
    
    print(f"\n{model_name} 模型评估指标:")
    print(metrics_df)
    print(f"\n预测结果已保存到: prediction_results_{model_name}.csv")
    
    return y_pred, y_prob, metrics, fpr, tpr, roc_auc, sample_names_valid, y_true_valid

# 存储ROC曲线数据和预测结果
roc_data = {}
predictions_storage = {}

# ====================== 1. C模型预测 ======================
result = predict_model('C', df_CI, df_C, model_feature_names['C'], C_features)
if result[0] is not None:
    y_pred_C, y_prob_C, metrics_C, fpr_C, tpr_C, auc_C, samples_C_used, y_true_C = result
    roc_data['C'] = (fpr_C, tpr_C, auc_C)
    predictions_storage['C'] = {
        'samples': samples_C_used,
        'y_true': y_true_C,
        'y_pred': y_pred_C,
        'y_prob': y_prob_C
    }

# ====================== 2. P模型预测 ======================
result = predict_model('P', df_CI, df_P, model_feature_names['P'], P_features)
if result[0] is not None:
    y_pred_P, y_prob_P, metrics_P, fpr_P, tpr_P, auc_P, samples_P_used, y_true_P = result
    roc_data['P'] = (fpr_P, tpr_P, auc_P)
    predictions_storage['P'] = {
        'samples': samples_P_used,
        'y_true': y_true_P,
        'y_pred': y_pred_P,
        'y_prob': y_prob_P
    }

# ====================== 3. U模型预测 ======================
result = predict_model('U', df_CI, df_U, model_feature_names['U'], U_features)
if result[0] is not None:
    y_pred_U, y_prob_U, metrics_U, fpr_U, tpr_U, auc_U, samples_U_used, y_true_U = result
    roc_data['U'] = (fpr_U, tpr_U, auc_U)
    predictions_storage['U'] = {
        'samples': samples_U_used,
        'y_true': y_true_U,
        'y_pred': y_pred_U,
        'y_prob': y_prob_U
    }

# ====================== 4. CPU投票模型 ======================
print(f"\n{'='*50}")
print("CPU 投票模型预测(仅使用CPU共有样本)")
print(f"{'='*50}")

if all(m in predictions_storage for m in ['C', 'P', 'U']):
    # 创建样本到预测的映射
    sample_to_pred = {}
    for model in ['C', 'P', 'U']:
        for i, sample in enumerate(predictions_storage[model]['samples']):
            if sample not in sample_to_pred:
                sample_to_pred[sample] = {}
            sample_to_pred[sample][model] = {
                'pred': predictions_storage[model]['y_pred'][i],
                'prob': predictions_storage[model]['y_prob'][i],
                'true': predictions_storage[model]['y_true'][i]
            }

    # 筛选CPU共有样本
    cpu_samples_list = []
    cpu_y_true = []
    cpu_pred_C = []
    cpu_pred_P = []
    cpu_pred_U = []
    cpu_prob_C = []
    cpu_prob_P = []
    cpu_prob_U = []

    for sample in sorted(samples_CPU):
        if sample in sample_to_pred and all(m in sample_to_pred[sample] for m in ['C', 'P', 'U']):
            cpu_samples_list.append(sample)
            cpu_y_true.append(sample_to_pred[sample]['C']['true'])
            cpu_pred_C.append(sample_to_pred[sample]['C']['pred'])
            cpu_pred_P.append(sample_to_pred[sample]['P']['pred'])
            cpu_pred_U.append(sample_to_pred[sample]['U']['pred'])
            cpu_prob_C.append(sample_to_pred[sample]['C']['prob'])
            cpu_prob_P.append(sample_to_pred[sample]['P']['prob'])
            cpu_prob_U.append(sample_to_pred[sample]['U']['prob'])

    print(f"CPU模型使用样本数: {len(cpu_samples_list)}")

    if len(cpu_samples_list) > 0:
        cpu_y_true = np.array(cpu_y_true)
        cpu_pred_C = np.array(cpu_pred_C)
        cpu_pred_P = np.array(cpu_pred_P)
        cpu_pred_U = np.array(cpu_pred_U)
        cpu_prob_C = np.array(cpu_prob_C)
        cpu_prob_P = np.array(cpu_prob_P)
        cpu_prob_U = np.array(cpu_prob_U)
        
        # 投票机制:多数投票
        y_pred_CPU = np.array([
            1 if (cpu_pred_C[i] + cpu_pred_P[i] + cpu_pred_U[i]) >= 2 else 0 
            for i in range(len(cpu_pred_C))
        ])
        
        # 概率平均
        y_prob_CPU = (cpu_prob_C + cpu_prob_P + cpu_prob_U) / 3
        
        # 计算指标
        metrics_CPU, fpr_CPU, tpr_CPU, auc_CPU = calculate_metrics(cpu_y_true, y_pred_CPU, y_prob_CPU)
        roc_data['CPU_Voting'] = (fpr_CPU, tpr_CPU, auc_CPU)
        
        # 保存结果
        result_df_CPU = pd.DataFrame({
            'Sample_Name': cpu_samples_list,
            'True_Label': cpu_y_true,
            'Predicted_Label': y_pred_CPU,
            'Predicted_Probability': y_prob_CPU,
            'C_Pred': cpu_pred_C,
            'P_Pred': cpu_pred_P,
            'U_Pred': cpu_pred_U
        })
        result_df_CPU.to_csv('prediction_results_CPU_voting.csv', index=False)
        
        metrics_df_CPU = pd.DataFrame({
            'Metric': list(metrics_CPU.keys()),
            'Value': list(metrics_CPU.values())
        })
        metrics_df_CPU.to_csv('metrics_CPU_voting.csv', index=False)
        
        print("\nCPU 投票模型评估指标:")
        print(metrics_df_CPU)
    else:
        print("警告: CPU共有样本数为0")

# ====================== 5. CP筛查模型 ======================
print(f"\n{'='*50}")
print("CP 筛查模型预测(仅使用CP共有样本)")
print(f"{'='*50}")

if all(m in predictions_storage for m in ['C', 'P']):
    # 筛选CP共有样本
    cp_samples_list = []
    cp_y_true = []
    cp_pred_C = []
    cp_pred_P = []
    cp_prob_C = []
    cp_prob_P = []

    for sample in sorted(samples_CP):
        if sample in sample_to_pred and all(m in sample_to_pred[sample] for m in ['C', 'P']):
            cp_samples_list.append(sample)
            cp_y_true.append(sample_to_pred[sample]['C']['true'])
            cp_pred_C.append(sample_to_pred[sample]['C']['pred'])
            cp_pred_P.append(sample_to_pred[sample]['P']['pred'])
            cp_prob_C.append(sample_to_pred[sample]['C']['prob'])
            cp_prob_P.append(sample_to_pred[sample]['P']['prob'])

    print(f"CP模型使用样本数: {len(cp_samples_list)}")

    if len(cp_samples_list) > 0:
        cp_y_true = np.array(cp_y_true)
        cp_pred_C = np.array(cp_pred_C)
        cp_pred_P = np.array(cp_pred_P)
        cp_prob_C = np.array(cp_prob_C)
        cp_prob_P = np.array(cp_prob_P)
        
        # CP筛查:只要C或P预测为阳性,则判定为阳性
        y_pred_CP = np.array([
            1 if (cp_pred_C[i] == 1 or cp_pred_P[i] == 1) else 0 
            for i in range(len(cp_pred_C))
        ])
        
        # 概率取最大值
        y_prob_CP = np.maximum(cp_prob_C, cp_prob_P)
        
        # 计算指标
        metrics_CP, fpr_CP, tpr_CP, auc_CP = calculate_metrics(cp_y_true, y_pred_CP, y_prob_CP)
        roc_data['CP_Screening'] = (fpr_CP, tpr_CP, auc_CP)
        
        # 保存结果
        result_df_CP = pd.DataFrame({
            'Sample_Name': cp_samples_list,
            'True_Label': cp_y_true,
            'Predicted_Label': y_pred_CP,
            'Predicted_Probability': y_prob_CP,
            'C_Pred': cp_pred_C,
            'P_Pred': cp_pred_P
        })
        result_df_CP.to_csv('prediction_results_CP_screening.csv', index=False)
        
        metrics_df_CP = pd.DataFrame({
            'Metric': list(metrics_CP.keys()),
            'Value': list(metrics_CP.values())
        })
        metrics_df_CP.to_csv('metrics_CP_screening.csv', index=False)
        
        print("\nCP 筛查模型评估指标:")
        print(metrics_df_CP)
    else:
        print("警告: CP共有样本数为0")

# ====================== 汇总所有指标 ======================
print(f"\n{'='*50}")
print("所有模型指标汇总")
print(f"{'='*50}")

summary_data = {
    'Model': [],
    'Sample_Count': [],
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'F1_Score': [],
    'AUC': []
}

# 添加C/P/U模型
for model_name in ['C', 'P', 'U']:
    if model_name in predictions_storage:
        summary_data['Model'].append(model_name)
        summary_data['Sample_Count'].append(len(predictions_storage[model_name]['samples']))
        metrics_var = eval(f'metrics_{model_name}')
        summary_data['Accuracy'].append(metrics_var['Accuracy'])
        summary_data['Sensitivity'].append(metrics_var['Sensitivity'])
        summary_data['Specificity'].append(metrics_var['Specificity'])
        summary_data['F1_Score'].append(metrics_var['F1_Score'])
        summary_data['AUC'].append(metrics_var['AUC'])

# 添加CPU模型
if 'cpu_samples_list' in locals() and len(cpu_samples_list) > 0:
    summary_data['Model'].append('CPU_Voting')
    summary_data['Sample_Count'].append(len(cpu_samples_list))
    summary_data['Accuracy'].append(metrics_CPU['Accuracy'])
    summary_data['Sensitivity'].append(metrics_CPU['Sensitivity'])
    summary_data['Specificity'].append(metrics_CPU['Specificity'])
    summary_data['F1_Score'].append(metrics_CPU['F1_Score'])
    summary_data['AUC'].append(metrics_CPU['AUC'])

# 添加CP模型
if 'cp_samples_list' in locals() and len(cp_samples_list) > 0:
    summary_data['Model'].append('CP_Screening')
    summary_data['Sample_Count'].append(len(cp_samples_list))
    summary_data['Accuracy'].append(metrics_CP['Accuracy'])
    summary_data['Sensitivity'].append(metrics_CP['Sensitivity'])
    summary_data['Specificity'].append(metrics_CP['Specificity'])
    summary_data['F1_Score'].append(metrics_CP['F1_Score'])
    summary_data['AUC'].append(metrics_CP['AUC'])

summary_metrics = pd.DataFrame(summary_data)
summary_metrics.to_csv('summary_all_models_metrics.csv', index=False)
print("\n所有模型指标汇总:")
print(summary_metrics)

# ====================== 绘制ROC曲线 ======================
if len(roc_data) > 0:
    print(f"\n{'='*50}")
    print("绘制ROC曲线")
    print(f"{'='*50}")

    colors = {
        'C': '#E41A1C',
        'P': '#377EB8',
        'U': '#4DAF4A',
        'CPU_Voting': '#984EA3',
        'CP_Screening': '#FF7F00'
    }

    # 1. 绘制单独的ROC曲线
    with PdfPages('ROC_curves_individual.pdf') as pdf:
        for model_name, (fpr, tpr, auc_score) in roc_data.items():
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(fpr, tpr, color=colors[model_name], lw=2.5, 
                    label=f'{model_name} (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
            ax.set_title(f'ROC Curve - {model_name} Model', fontsize=16, fontweight='bold')
            ax.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=12)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print("单独ROC曲线已保存到: ROC_curves_individual.pdf")

    # 2. 绘制所有模型的ROC曲线在一张图上
    fig, ax = plt.subplots(figsize=(10, 10))
    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        ax.plot(fpr, tpr, color=colors[model_name], lw=2.5, 
                label=f'{model_name} (AUC = {auc_score:.3f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves - All Models Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig('ROC_curves_all_models.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print("综合ROC曲线已保存到: ROC_curves_all_models.pdf")

print(f"\n{'='*50}")
print("所有预测完成!")
print(f"{'='*50}")
print("\n标准化处理流程:")
print("步骤1: 临床特征 → 用scaler_standard_{C/P/U}.pkl标准化")
print("步骤2: 质谱所有特征 → StandardScaler标准化")
print("步骤3: 使用预定义特征列表提取质谱特征")
print("步骤4: 合并临床+质谱特征")
print("步骤5: 对合并后的所有特征再做一次StandardScaler标准化")
print("\n生成的文件:")
print("【临床特征文件】")
print("1. clinical_features_filled_C.csv (填补缺失值后)")
print("2. clinical_features_filled_P.csv")
print("3. clinical_features_filled_U.csv")
print("4. clinical_features_normalized_C.csv (标准化后)")
print("5. clinical_features_normalized_P.csv")
print("6. clinical_features_normalized_U.csv")
print("\n【质谱特征文件(新增)】")
print("7. mass_features_extracted_C.csv (使用预定义特征提取)")
print("8. mass_features_extracted_P.csv")
print("9. mass_features_extracted_U.csv")
print("\n【预测结果CSV】")
print("10. prediction_results_C.csv")
print("11. prediction_results_P.csv")
print("12. prediction_results_U.csv")
print("13. prediction_results_CPU_voting.csv")
print("14. prediction_results_CP_screening.csv")
print("\n【评估指标CSV】")
print("15. metrics_C.csv")
print("16. metrics_P.csv")
print("17. metrics_U.csv")
print("18. metrics_CPU_voting.csv")
print("19. metrics_CP_screening.csv")
print("20. summary_all_models_metrics.csv")
print("\n【ROC曲线PDF】")
print("21. ROC_curves_individual.pdf")
print("22. ROC_curves_all_models.pdf")