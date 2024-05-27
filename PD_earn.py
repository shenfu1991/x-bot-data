import os
import pandas as pd

# 输入文件夹路径
input_folder = './'
# 输出文件路径
output_file = './output.csv'

# 初始化一个空的DataFrame来存储结果
result_df = pd.DataFrame()

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 构造文件的完整路径
        file_path = os.path.join(input_folder, filename)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 过滤列earn的值大于20的行
        filtered_df = df[df['earn'] > 20]
        # 将结果追加到result_df
        result_df = pd.concat([result_df, filtered_df], ignore_index=True)

# 将结果写入新的CSV文件
result_df.to_csv(output_file, index=False)

print(f"处理完成，结果已保存到 {output_file}")