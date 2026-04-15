import csv
import requests
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import pytz
import numpy as np
import os

mins = 15
fileName = "hzx_2.csv"
pading = 2
fontSize = 50
passLoss = False
# passLoss = True
minPassLoss = 30

if mins == 15:
    pading = 8
    fontSize = 25

# ---
# MACD 设置
macd_fast = 12
macd_slow = 26
macd_signal = 9
# ---

def add_dashed_lines(df, panel, levels):
    """添加虚线到图表"""
    if not isinstance(levels, (list, tuple, np.ndarray)):
        raise ValueError("levels must be a list, tuple or numpy array")

    add_plots = []
    for level in levels:
        if not isinstance(level, (int, float)):
            continue
        add_plot = mpf.make_addplot([level] * len(df), panel=panel, color='gray', linestyle='--', secondary_y=False)
        add_plots.append(add_plot)
    return add_plots

def calculate_dmi(high, low, close, window=14):
    """计算 DMI 指标 (返回 ADX, PDI, MDI)"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up = high - high.shift()
    down = low.shift() - low

    pos_dm = np.where((up > down) & (up > 0), up, 0)
    neg_dm = np.where((down > up) & (down > 0), down, 0)

    pos_dm = pd.Series(pos_dm, index=up.index)
    neg_dm = pd.Series(neg_dm, index=down.index)

    tr_smooth = true_range.ewm(alpha=1/window, adjust=False).mean()
    pos_dm_smooth = pos_dm.ewm(alpha=1/window, adjust=False).mean()
    neg_dm_smooth = neg_dm.ewm(alpha=1/window, adjust=False).mean()

    pos_di = 100 * pos_dm_smooth / tr_smooth
    neg_di = 100 * neg_dm_smooth / tr_smooth

    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-8)
    adx = dx.ewm(alpha=1/window, adjust=False).mean()

    return pd.Series(adx, name='ADX'), pd.Series(pos_di, name='PDI'), pd.Series(neg_di, name='MDI')

def round_to_nearest_3min(dt):
    """四舍五入到最近的时间间隔"""
    if not isinstance(dt, datetime):
        raise ValueError("Input must be a datetime object")
    rounded = dt - timedelta(minutes=dt.minute % mins,
                             seconds=dt.second,
                             microseconds=dt.microsecond)
    return rounded

def calculate_macd(data, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def process_csv(file_path):
    """处理CSV文件并生成图表"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            try:
                symbol = row['symbol']
                open_time_str = row['openTime']
                close_time_str = row['closeTime']
                boot_time_str = row.get('GBFirstBootTime', '0')
                add_time1_str = row['addTime1']
                add_time2_str = row.get('addTime2', 'none')
                add_time3_str = row.get('addTime3', 'none')
                side = row['side']
                earn = float(row['earn'])
                me = float(row['maxEarn'])

                if passLoss and earn < minPassLoss:
                    continue

                earnRate = row['earnRate']
                cp = row['closeType']
                maxEarnRate = row['maxEarnRate']
                funcName = row["funcName"]

                current_year = datetime.now().year
                tz = pytz.timezone('Asia/Shanghai')

                open_time = tz.localize(datetime.strptime(f"{open_time_str}", '%Y-%m-%d %H:%M:%S'))
                close_time = tz.localize(datetime.strptime(f"{close_time_str}", '%Y-%m-%d %H:%M:%S'))

                rounded_boot_timestamp = None
                if boot_time_str and boot_time_str not in ['0', '00-00 00:00:00']:
                    boot_time = tz.localize(datetime.strptime(f"{boot_time_str}", '%Y-%m-%d %H:%M:%S'))
                    rounded_boot_time = round_to_nearest_3min(boot_time)
                    rounded_boot_timestamp = int(rounded_boot_time.timestamp() * 1000)

                open_timestamp = int(open_time.timestamp() * 1000)
                close_timestamp = int(close_time.timestamp() * 1000)

                dateBoot = (rounded_boot_timestamp if rounded_boot_timestamp else open_timestamp) - 1000 * 60 * 60 * pading
                dateClose = close_timestamp + 1000 * 60 * 60 * pading

                try:
                    klines_url = f"https://fapi.binance.com/fapi/v1/continuousKlines?interval={mins}m&contractType=PERPETUAL&pair={symbol}&startTime={dateBoot}&endTime={dateClose}"
                    response = requests.get(klines_url)
                    response.raise_for_status()
                    klines_data = response.json()
                except requests.RequestException as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    continue

                if len(klines_data) > 0:
                    df = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    df = df.astype(float)

                    utc_plus_8 = pytz.timezone('Asia/Shanghai')
                    df.index = df.index.tz_localize(pytz.utc).tz_convert(utc_plus_8)

                    # --- 计算不同高度的偏移逻辑 ---
                    price_range = df['high'].max() - df['low'].min()
                    # 每一阶单位设为总振幅的 3%
                    step = price_range * 0.03

                    rounded_open_time = round_to_nearest_3min(open_time)
                    rounded_close_time = round_to_nearest_3min(close_time)
                    rounded_open_timestamp = int(rounded_open_time.timestamp() * 1000)
                    rounded_close_timestamp = int(rounded_close_time.timestamp() * 1000)

                    add_timestamps = []
                    for t_str in [add_time1_str, add_time2_str, add_time3_str]:
                        if t_str and t_str.lower() != 'none':
                            try:
                                add_time = tz.localize(datetime.strptime(f"{current_year}-{t_str}", '%Y-%m-%d %H:%M:%S'))
                                rounded_add_time = round_to_nearest_3min(add_time)
                                add_timestamps.append(int(rounded_add_time.timestamp() * 1000))
                            except ValueError:
                                continue

                    open_close_markers = [np.nan] * len(df)
                    add_markers = [np.nan] * len(df)
                    boot_markers = [np.nan] * len(df)

                    for i, r_val in df.iterrows():
                        ts = int(i.timestamp() * 1000)
                        idx = df.index.get_loc(i)

                        # 蓝色点：第1层高度 (High + 1个单位)
                        if ts == rounded_open_timestamp or ts == rounded_close_timestamp:
                            open_close_markers[idx] = r_val['high'] + step

                        # 黑色点：第2层高度 (High + 2个单位)
                        if ts in add_timestamps:
                            add_markers[idx] = r_val['high'] + step * 2

                        # 紫色点：第3层高度 (High + 3个单位)
                        if rounded_boot_timestamp and ts == rounded_boot_timestamp:
                            boot_markers[idx] = r_val['high'] + step * 3

                    # --- 指标计算 ---
                    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
                    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
                    df['EMA30'] = df['close'].ewm(span=30, adjust=False).mean()
                    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
                    df['custom_mid'] = (df['open'] + df['close']) / 2
                    df['custom_sma'] = df['custom_mid'].rolling(window=10).mean()

                    macd, signal_line, histogram = calculate_macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                    adx, pdi, mdi = calculate_dmi(df['high'], df['low'], df['close'])

                    add_plots = []
                    # 绘图：开平仓 (蓝色)
                    if not np.all(np.isnan(open_close_markers)):
                        add_plots.append(mpf.make_addplot(open_close_markers, type='scatter', markersize=300, marker='o', color='blue'))
                    # 绘图：加仓 (黑色)
                    if not np.all(np.isnan(add_markers)):
                        add_plots.append(mpf.make_addplot(add_markers, type='scatter', markersize=300, marker='o', color='black'))
                    # 绘图：Boot (紫色)
                    if not np.all(np.isnan(boot_markers)):
                        add_plots.append(mpf.make_addplot(boot_markers, type='scatter', markersize=300, marker='o', color='purple'))

                    # 主图指标
                    add_plots.extend([
                        mpf.make_addplot(df['EMA5'], linestyle='--', color='red'),
                        mpf.make_addplot(df['EMA20'], linestyle='--', color='orange'),
                        mpf.make_addplot(df['EMA30'], linestyle='--', color='purple'),
                        mpf.make_addplot(df['EMA50'], linestyle='--', color='green'),
                        mpf.make_addplot(df['custom_sma'], color='cyan', width=1.5)
                    ])

                    # MACD & DMI Panel
                    colors = ['red' if h < 0 else 'green' for h in histogram]
                    add_plots.extend([
                        mpf.make_addplot(macd, panel=1, color='fuchsia', ylabel='MACD'),
                        mpf.make_addplot(signal_line, panel=1, color='dodgerblue'),
                        mpf.make_addplot(histogram, type='bar', panel=1, color=colors),
                        mpf.make_addplot(adx, panel=2, color='black', ylabel='DMI/ADX'),
                        mpf.make_addplot(pdi, panel=2, color='green'),
                        mpf.make_addplot(mdi, panel=2, color='red'),
                    ])
                    add_plots.extend(add_dashed_lines(df, panel=2, levels=[20, 40]))

                    # 绘图参数
                    mc = mpf.make_marketcolors(up='green', down='red', edge='i', wick='i', volume='in', ohlc='i')
                    s = mpf.make_mpf_style(marketcolors=mc)
                    additional_text = f"{funcName} {symbol} {open_time_str} -> {close_time_str} {side} {earn}/{me} {earnRate}/{maxEarnRate} {cp}"
                    num_candles = len(df)
                    fig_width = max(15, num_candles // 2)
                    fig_height = fig_width / 1.2

                    try:
                        fig, ax = mpf.plot(df, type='candle', volume=False, returnfig=True,
                                         style=s, addplot=add_plots, figsize=(fig_width, fig_height),
                                         panel_ratios=(6, 2, 2))

                        ax[0].set_title(additional_text, fontsize=fontSize, pad=40) # 增加 pad 防止标题重叠点
                        ax[0].tick_params(axis='both', labelsize=20)

                        safe_symbol = "".join(c for c in symbol if c.isalnum() or c in ('_', '-'))
                        safe_open_time = open_time_str.replace(':', '-').replace(' ', '_')
                        safe_close_time = close_time_str.replace(':', '-').replace(' ', '_')
                        output_file = f"{safe_symbol}_{safe_open_time}_{safe_close_time}.png"

                        fig.savefig(output_file)
                        print(f"K-line chart saved to {output_file}")
                    except Exception as e:
                        print(f"Error generating chart for {symbol}: {e}")
                else:
                    print(f"No data found for {symbol}")
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

if __name__ == "__main__":
    print(f"Processing file: {fileName}")
    try:
        process_csv(fileName)
    except Exception as e:
        print(f"Error: {e}")
