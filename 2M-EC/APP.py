import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# 显示图片和标题
st.markdown("""
    <img src="https://github.com/Dandan-debug/2M-EC/raw/main/endometrial.svg" width="100" alt="Endometrial Cancer Model Image" style="display: block; margin: 0 auto 20px;">
    <h1 style="font-weight: bold; font-size: 50px; text-align: center; margin: 0;">
        2M-EC Predictive Platform
    </h1>
""", unsafe_allow_html=True)

# 显示描述文本
st.markdown("""
    <p style='text-align: left; font-size: 16px; margin-bottom: 28px;'>
        The 2M-EC (Bimodal Multilevel Endometrial Cancer) is designed for patient-centered minimally invasive ENDOM screening with high sensitivity and precise diagnosis.<br>
        Utilizes multiple models to calculate cancer risk probabilities, where:<br>
        • High-risk probability = Highest cancer probability across models<br>
        • Low-risk probability = 1 - Highest cancer probability<br>
    </p>
""", unsafe_allow_html=True)

# 尝试导入SHAP，如果失败则提供备用方案
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.sidebar.warning("SHAP library not available. Visualization features will be limited.")

# 加载标准器和模型
scalers = {
    'C': joblib.load('scaler_standard_C.pkl'),
    'P': joblib.load('scaler_standard_P.pkl'),
    'U': joblib.load('scaler_standard_U.pkl')
}

models = {
    'C': joblib.load('xgboost_C.pkl'),
    'P': joblib.load('xgboost_P.pkl'),
    'U': joblib.load('xgboost_U.pkl')
}

# 定义特征名称
display_features_to_scale = [
    'Age (years)',                                  # Age (e.g., 52 years)
    'Endometrial thickness (mm)',                   # Endometrial thickness in mm
    'HE4 (pmol/L)',                                 # HE4 level in pmol/L
    'Menopause (1=yes)',                            # Menopause status (1=yes)
    'HRT (Hormone Replacement Therapy, 1=yes)',     # HRT status (1=yes)
    'Endometrial heterogeneity (1=yes)',            # Endometrial heterogeneity (1=yes)
    'Uterine cavity occupation (1=yes)',            # Uterine cavity occupation (1=yes)
    'Uterine cavity occupying lesion with rich blood flow (1=yes)', # Uterine cavity occupying lesion with rich blood flow (1=yes)
    'Uterine cavity fluid (1=yes)'                  # Uterine cavity fluid (1=yes)
]

# 原始特征名称，用于标准化器
original_features_to_scale = [
    'CI_age', 'CI_endometrial thickness', 'CI_HE4', 'CI_menopause',
    'CI_HRT', 'CI_endometrial heterogeneity',
    'CI_uterine cavity occupation',
    'CI_uterine cavity occupying lesion with rich blood flow',
    'CI_uterine cavity fluid'
]

# 额外特征名称映射（移除 .0 后缀）
additional_features = {
    'C': ['CM4160.0','CM727.0','CM889.0','CM7441.0','CM995.0','CM7440.0','CM7439.0','CM734.0',
          'CM1857.0','CM6407.0','CM2920.0','CM729.0','CM628.0'],

    'P': ['PM816.0','PM846.0','PM120.0','PP408.0','PM883.0','PM801.0','PM578.0',
          'PP48.0','PM504.0','PP317.0','PM722.0','PM86.0','PP63.0','PP405.0',
          'PM574.0','PP434.0','PM163.0','PP81.0','PM461.0','PM571.0','PM88.0','PP378.0',
          'PM867.0','PP286.0','PM409.0','PP497.0','PM900.0','PM836.0','PP393.0',
          'PP653.0','PP456.0','PP75.0','PP488.0','PM887.0','PP640.0','PP344.0',
          'PM584.0','PM396.0','PM681.0','PP332.0','PM328.0','PM882.0','PM548.0',
          'PM832.0','PM232.0','PM285.0','PM104.0','PM379.0','PM782.0'],

    'U': ['UM7578.0', 'UM510.0', 'UM507.0', 'UM670.0', 'UM351.0',
          'UM5905.0', 'UM346.0', 'UM355.0', 'UM8899.0', 'UM1152.0',
          'UM5269.0', 'UM6437.0', 'UM5906.0', 'UM7622.0', 'UM8898.0',
          'UM2132.0', 'UM3513.0', 'UM790.0', 'UM8349.0', 'UM2093.0',
          'UM4210.0', 'UM3935.0', 'UM4256.0']
}

# SHAP可视化函数
def show_shap_waterfall(model, input_data, feature_names, model_name):
    """显示SHAP瀑布图来解释模型决策"""
    if not SHAP_AVAILABLE:
        st.warning("SHAP visualization not available. Please install shap and matplotlib.")
        return
    
    try:
        # 使用TreeExplainer（适用于XGBoost）
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer(input_data)
        
        # 创建瀑布图
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.title(f"SHAP Explanation for {model_name} Model", fontsize=16, pad=20)
        plt.tight_layout()
        
        # 在Streamlit中显示
        st.pyplot(fig)
        plt.close()
        
        # 添加解释文本
        st.markdown(f"""
        **Interpretation for {model_name} Model:**
        - **Blue bars**: Features that decrease cancer risk
        - **Red bars**: Features that increase cancer risk  
        - **Base value**: Average model prediction
        - **Final prediction**: Your individual prediction
        """)
        
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation for {model_name} model: {str(e)}")

# 模型选择
selected_models = st.multiselect(
    "Select the model(s) to be used (you can select one or more)",
    options=['U', 'C', 'P'],
    default=['U']
)

# 获取用户输入
user_input = {}

# 定义特征输入
for i, feature in enumerate(display_features_to_scale):
    if "1=yes" in feature:  # 对于分类变量，限制输入为0或1
        user_input[original_features_to_scale[i]] = st.selectbox(f"{feature}:", options=[0, 1])
    else:  # 对于连续变量，使用数值输入框
        user_input[original_features_to_scale[i]] = st.number_input(f"{feature}:", min_value=0.0, value=0.0)

# 为每个选定的模型定义额外特征
for model_key in selected_models:
    for feature in additional_features[model_key]:
        # 允许保留较多小数位的输入
        user_input[feature] = st.number_input(f"{feature} ({model_key}):", min_value=0.0, format="%.9f")

# 预测按钮
if st.button("Submit"):
    # 定义模型预测结果存储字典
    model_predictions = {}
    model_inputs = {}  # 存储每个模型的输入数据

    # 对选定的每个模型进行标准化和预测
    for model_key in selected_models:
        # 针对每个模型构建专用的输入数据
        model_input_df = pd.DataFrame([user_input])
        
        # 获取模型所需的特征列
        model_features = original_features_to_scale + additional_features[model_key]
        
        # 仅保留当前模型需要的特征
        model_input_df = model_input_df[model_features]
        
        # 保存原始输入用于SHAP分析
        original_input = model_input_df.copy()
        
        # 对需要标准化的特征进行标准化
        model_input_df[original_features_to_scale] = scalers[model_key].transform(model_input_df[original_features_to_scale])
        
        # 使用模型进行预测
        predicted_proba = models[model_key].predict_proba(model_input_df)[0]
        predicted_class = models[model_key].predict(model_input_df)[0]
        
        # 保存预测结果和输入数据
        model_predictions[model_key] = {
            'proba': predicted_proba,
            'class': predicted_class
        }
        model_inputs[model_key] = {
            'scaled': model_input_df,
            'features': model_features
        }

    # 用户选择1个模型时直接报错
    if len(selected_models) == 1:
        st.error("Please select at least 2 models for prediction")

    # 用户选择2个模型但不是C和P组合时也报错
    elif len(selected_models) == 2 and set(selected_models) != {'C', 'P'}:
        st.error("For 2 models, only C and P combination is supported")

    # 仅当选择2个模型且为C和P时才处理
    elif len(selected_models) == 2 and set(selected_models) == {'C', 'P'}:
        # 检查是否有阳性预测（类别1）
        has_positive = any(model_predictions[model_key]['class'] == 1 for model_key in selected_models)

        if has_positive:
            # 取两个模型中预测癌症概率更高的值
            max_proba = max(model_predictions[model_key]['proba'][1] for model_key in selected_models)
            st.success(f"ENDOM screening：{max_proba * 100:.2f}%- high risk")
            
            # 显示SHAP瀑布图
            st.subheader("Model Decision Explanation")
            for model_key in selected_models:
                show_shap_waterfall(models[model_key], model_inputs[model_key]['scaled'], 
                                   model_inputs[model_key]['features'], model_key)
        else:
            # 取两个模型中预测癌症概率更高的值（虽然都是阴性）
            max_proba = max(model_predictions[model_key]['proba'][1] for model_key in selected_models)
            st.info(f"ENDOM screening：{max_proba * 100:.2f}%- low risk")

    # 用户选择3个模型
    elif len(selected_models) == 3:
        # 统计阳性预测数量
        positive_count = sum(model_predictions[model_key]['class'] == 1 for model_key in selected_models)

        if positive_count >= 2:  # 多数为阳性
            # 取三个模型中预测癌症概率最高的值
            max_proba = max(model_predictions[model_key]['proba'][1] for model_key in selected_models)
            st.success(f"ENDOM diagnosis：{max_proba * 100:.2f}%- high risk")
            
            # 显示SHAP瀑布图
            st.subheader("Model Decision Explanation")
            for model_key in selected_models:
                if model_predictions[model_key]['class'] == 1:  # 只显示阳性模型的SHAP
                    show_shap_waterfall(models[model_key], model_inputs[model_key]['scaled'], 
                                       model_inputs[model_key]['features'], model_key)
        else:  # 多数为阴性
            # 计算1减去三个模型中最高癌症概率
            max_proba = max(model_predictions[model_key]['proba'][1] for model_key in selected_models)
            low_risk_proba = (1 - max_proba) * 100
            st.info(f"ENDOM diagnosis：{low_risk_proba:.2f}%- low risk")

    # 其他情况也报错（比如选择0个或超过3个）
    else:
        st.error("Invalid model selection. Please select 2 or 3 models.")

# 添加SHAP安装说明
if not SHAP_AVAILABLE:
    st.sidebar.markdown("""
    **To enable SHAP visualization:**
    Add to your requirements.txt:
    ```
    shap>=0.41.0
    matplotlib>=3.3.0
    ```
    """)
