import os
import pandas as pd

FOLDER_PATH = "/Users/xuanyuan/x-bot-data"
OUTPUT_DIR = "/Users/xuanyuan/x-bot-data"
SORT_COLUMNS = ["symbol", "openTime"]
PREFIX = "_sorted"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(FOLDER_PATH):
    if not file.lower().endswith(".csv"):
        continue

    src = os.path.join(FOLDER_PATH, file)

    name, ext = os.path.splitext(file)
    new_name = f"{name}{PREFIX}{ext}"
    dst = os.path.join(OUTPUT_DIR, new_name)

    try:
        df = pd.read_csv(src)

        if "symbol" not in df.columns or "openTime" not in df.columns:
            print(f"[跳过] {file} 缺少 symbol 或 openTime")
            continue

        # 将 openTime 转为 datetime（假设同一年，不影响相对顺序）
        df["_openTime_dt"] = pd.to_datetime(
            df["openTime"],
            format="%m-%d %H:%M:%S",
            errors="coerce"
        )

        # 排序：symbol -> openTime（由远及近）
        df_sorted = df.sort_values(
            by=["symbol", "_openTime_dt"],
            ascending=[True, True]
        )

        # 清理临时列
        df_sorted.drop(columns="_openTime_dt", inplace=True)

        df_sorted.to_csv(dst, index=False)
        print(f"[完成] {file} -> {new_name}")

    except Exception as e:
        print(f"[错误] {file}: {e}")
