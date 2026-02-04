import numpy as np
import os
import pandas as pd


def simple_binning_txt_data(input_folder, output_csv_path, n_bins=9000, min_bin=100, max_bin=1000):
    """
    简化的分箱处理 - 处理有表头的txt文件
    """

    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    print(f"找到 {len(txt_files)} 个txt文件")

    all_binned_data = []
    sample_names = []

    for file in txt_files:
        try:
            file_path = os.path.join(input_folder, file)

            # 使用pandas读取有表头的txt文件,指定空格分隔
            df = pd.read_csv(file_path, sep='\s+', header=0, engine='python')  # \s+ 匹配一个或多个空格

            # 获取m/z和强度数据
            if df.shape[1] >= 2:
                mz_values = df.iloc[:, 0].values  # 第一列为m/z
                intensities = df.iloc[:, 1].values  # 第二列为强度
                print(
                    f"文件 {file}: 数据点数量 = {len(mz_values)}, m/z范围 = [{mz_values[0]:.2f}, {mz_values[-1]:.2f}]")
            else:
                print(f"文件 {file} 列数不足,跳过")
                continue

            # 分箱处理
            bin_edges = np.linspace(min_bin, max_bin, n_bins + 1)
            binned_intensities = []

            for i in range(len(bin_edges) - 1):
                mask = (mz_values >= bin_edges[i]) & (mz_values < bin_edges[i + 1])
                if np.any(mask):
                    # 取该bin内的加和强度值
                    sum_intensity = np.sum(intensities[mask])
                else:
                    sum_intensity = 0
                binned_intensities.append(sum_intensity)

            all_binned_data.append(binned_intensities)
            sample_names.append(os.path.splitext(file)[0])
            print(f"成功处理文件: {file}")

        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")

    if len(all_binned_data) == 0:
        print("没有成功处理任何文件!")
        return

    # 保存为CSV
    X_binned = np.array(all_binned_data)
    bin_columns = [f'bin_{i}' for i in range(1, X_binned.shape[1] + 1)]  # 从1开始编号

    # 创建DataFrame并保存
    output_df = pd.DataFrame(X_binned, columns=bin_columns)
    output_df.insert(0, 'sample_name', sample_names)
    output_df.to_csv(output_csv_path, index=False)

    print(f"分箱完成!数据已保存到: {output_csv_path}")
    print(f"数据形状: {X_binned.shape}")
    print(f"成功处理 {len(sample_names)} 个样本")


# 主程序
if __name__ == "__main__":
    input_folder = '/Users/apple/Desktop/FD/FD/子宫内膜癌EC/项目1/文章/Manuscript/submit/LQ修改1/ddl/0604/修改20250711/PNAS/CRM投稿材料/CRM/外部验证队列/CRM_cohort/Dataset/Bin_UM'
    output_csv = os.path.join(input_folder, 'binned_spectra.csv')

    simple_binning_txt_data(input_folder, output_csv)