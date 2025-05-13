from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# 加载 CSV 数据
input_file = "ROMA-84.csv"  # 替换为你的 CSV 文件路径
data = pd.read_csv(input_file)

# 初始化 Selenium WebDriver
driver = webdriver.Chrome()  # 使用 Chrome 浏览器，需确保安装了 ChromeDriver
driver.get("https://xema.com.ua/en/roma/")

# 等待页面加载完成
wait = WebDriverWait(driver, 10)  # 增加等待时间以确保元素加载


# 函数：选择绝经状态
def select_menopausal_status(menopause):
    try:
        if menopause == 1:  # Postmenopausal
            print("Attempting to select Postmenopausal...")
            button = wait.until(EC.visibility_of_element_located((By.XPATH,
                                                                  "/html/body/div[6]/main/div[2]/section/div/div/div/div[2]/form/div[1]/div[2]/div/div[2]/label[2]/input")))
            driver.execute_script("arguments[0].scrollIntoView(true);", button)  # 滚动至可见区域
            driver.execute_script("arguments[0].click();", button)  # 强制点击
            print("Postmenopausal selected successfully.")
        elif menopause == 0:  # Premenopausal
            print("Attempting to select Premenopausal...")
            button = wait.until(EC.visibility_of_element_located((By.XPATH,
                                                                  "/html/body/div[6]/main/div[2]/section/div/div/div/div[2]/form/div[1]/div[2]/div/div[2]/label[2]/input")))
            driver.execute_script("arguments[0].scrollIntoView(true);", button)  # 滚动至可见区域
            driver.execute_script("arguments[0].click();", button)  # 强制点击
            print("Premenopausal selected successfully.")
        else:
            print(f"Invalid menopausal status: {menopause}")
    except Exception as e:
        raise Exception(f"Error selecting menopausal status: {e}")


# 函数：输入数据并获取结果
def get_risk_result(ca125, he4, menopause):
    try:
        # 选择绝经状态
        select_menopausal_status(menopause)

        # 输入 CA125 和 HE4 值
        print("Attempting to input CA125 and HE4 values...")
        ca125_input = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Put the value of CA125']")))
        he4_input = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Put the value of HE4']")))
        ca125_input.clear()
        he4_input.clear()
        ca125_input.send_keys(str(ca125))
        he4_input.send_keys(str(he4))
        print(f"CA125={ca125} and HE4={he4} values entered successfully.")

        # 使用 btn_primary-2 类名点击按钮
        print("Attempting to click the Calculate button using class 'btn_primary-2'...")
        calculate_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "btn_primary-2")))  # 使用类名定位计算按钮

        # 强制滚动并点击按钮
        driver.execute_script("arguments[0].scrollIntoView(true);", calculate_button)  # 滚动至按钮可见区域
        time.sleep(1)  # 稍作延迟确保页面稳定
        driver.execute_script("arguments[0].click();", calculate_button)  # 强制点击
        print("Calculate button clicked successfully.")

        # 等待计算结果出现，并增加等待时间
        print("Waiting for the result...")
        time.sleep(1)  # 增加等待时间以确保计算完成

        # 使用 .input.calc-res 类来获取计算结果
        result_element = wait.until(
            EC.visibility_of_element_located((By.CLASS_NAME, "input.calc-res")))  # 使用 class 定位结果区域
        result = result_element.text.strip()  # 获取计算结果并去除前后空白字符
        print(f"Raw result obtained: {result}")

        # 提取 High risk 或 Low risk
        if "High risk" in result:
            return "High risk"
        elif "Low risk" in result:
            return "Low risk"
        else:
            return "Unknown risk"  # 如果无法确定风险级别

    except Exception as e:
        print(f"Error getting risk result: {e}")
        raise Exception(f"Error getting risk result: {e}")


# 创建一个结果列表
results = []

# 遍历输入数据
for index, row in data.iterrows():
    ca125 = row["CA125"]  # 读取 CA125 值
    he4 = row["HE4"]  # 读取 HE4 值
    menopause = row["menopause"]  # 读取绝经状态（1 或 0）
    print(f"Processing row {index}: CA125={ca125}, HE4={he4}, Menopause={menopause}")

    try:
        # 获取结果
        risk_result = get_risk_result(ca125, he4, menopause)
        results.append(
            {"Samples": row["Samples"], "Group": row["Group"], "CA125": ca125, "HE4": he4, "Menopause": menopause,
             "Risk Result": risk_result})
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        results.append(
            {"Samples": row["Samples"], "Group": row["Group"], "CA125": ca125, "HE4": he4, "Menopause": menopause,
             "Risk Result": "Error"})

# 保存结果到文件
output_file = "output_results_ROMA-84.csv"
output_df = pd.DataFrame(results)
output_df.to_csv(output_file, index=False)

print(f"结果已保存到 {output_file}")

# 关闭浏览器
driver.quit()
