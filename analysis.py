import os
import pandas as pd

arr = ["hz","jk","tc","yd","nl"]

def analyze_csv_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and file.startswith(tuple(arr)):
            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(file_path)

                # 校验列是否存在
                if 'earn' not in df.columns or 'symbol' not in df.columns:
                    print(f"{file}：缺少 earn 或 symbol 列，跳过")
                    continue

                # 去掉 NaN
                df = df[['symbol', 'earn']].dropna()

                # 转换类型（防止字符串）
                df['earn'] = pd.to_numeric(df['earn'], errors='coerce')
                df = df.dropna()

                # Top 10 盈利
                top_earn = df.sort_values(by='earn', ascending=False).head(10)

                # Top 10 亏损
                top_loss = df.sort_values(by='earn', ascending=True).head(10)

                print(f"\n文件名：{file}")
                print("earn top10：")
                for _, row in top_earn.iterrows():
                    print(f"{row['symbol']} : {row['earn']}")

                print("loss top10：")
                for _, row in top_loss.iterrows():
                    print(f"{row['symbol']} : {row['earn']}")

            except Exception as e:
                print(f"{file} 处理失败：{e}")


# 使用示例
folder_path = "/Users/xuanyuan/x-bot-data"  # 替换为你的目录
analyze_csv_folder(folder_path)
