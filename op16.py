import csv
import requests
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import pytz
import numpy as np
import os

mins = 15
<<<<<<< HEAD
fileName = "tcx_3.csv"
=======
fileName = "tcx_4.csv"
>>>>>>> 50fe7cc9dd2e4ac51c3f89204ad9bc94a04e363f
pading = 2
fontSize = 50
passLoss = False

if mins == 15:
    pading = 8
    fontSize = 25

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

def calculate_adx(high, low, close, window=14):
    """计算ADX指标"""
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

    return pd.Series(adx, name='ADX')

def round_to_nearest_3min(dt):
    """四舍五入到最近的3分钟间隔"""
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
                add_time1_str = row['addTime1']
                add_time2_str = row['addTime3']
                add_time3_str = row['addTime3']
                side = row['side']
                SLstatus = row['SLstatus']

                # 转换收益值为浮点数
                earn = float(row['earn'])
                me = float(row['maxEarn'])

                if passLoss and earn < 20:
                    continue

                earnRate = row['earnRate']
                cp = row['closeType']
                maxEarnRate = row['maxEarnRate']
                funcName = row["funcName"]

                current_year = datetime.now().year
                # 添加时区信息
                tz = pytz.timezone('Asia/Shanghai')
                open_time = tz.localize(datetime.strptime(f"{current_year}-{open_time_str}", '%Y-%m-%d %H:%M:%S'))
                close_time = tz.localize(datetime.strptime(f"{current_year}-{close_time_str}", '%Y-%m-%d %H:%M:%S'))

                open_timestamp = int(open_time.timestamp() * 1000)
                close_timestamp = int(close_time.timestamp() * 1000)

                dateOpen = open_timestamp - 1000 * 60 * 60 * pading
                dateClose = close_timestamp + 1000 * 60 * 60 * pading

                try:
                    klines_url = f"https://fapi.binance.com/fapi/v1/continuousKlines?interval={mins}m&contractType=PERPETUAL&pair={symbol}&startTime={dateOpen}&endTime={dateClose}"
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

                    # 确保时区转换正确
                    utc_plus_8 = pytz.timezone('Asia/Shanghai')
                    df.index = df.index.tz_localize(pytz.utc).tz_convert(utc_plus_8)

                    additional_text = f"【{SLstatus}】{funcName} {symbol} {open_time_str} ----> {close_time_str}  {side}  {earn}/{me} {earnRate}/{maxEarnRate} {cp}"

                    mc = mpf.make_marketcolors(up='green', down='red', edge='i', wick='i', volume='in', ohlc='i')
                    s = mpf.make_mpf_style(marketcolors=mc)

                    rounded_open_time = round_to_nearest_3min(open_time)
                    rounded_close_time = round_to_nearest_3min(close_time)

                    rounded_open_timestamp = int(rounded_open_time.timestamp() * 1000)
                    rounded_close_timestamp = int(rounded_close_time.timestamp() * 1000)

                    add_timestamps = []
                    for add_time_str in [add_time1_str, add_time2_str, add_time3_str]:
                        if add_time_str.lower() != 'none':
                            try:
                                add_time = tz.localize(datetime.strptime(f"{current_year}-{add_time_str}", '%Y-%m-%d %H:%M:%S'))
                                rounded_add_time = round_to_nearest_3min(add_time)
                                add_timestamps.append(int(rounded_add_time.timestamp() * 1000))
                            except ValueError:
                                continue

                    open_close_markers = [np.nan] * len(df)
                    add_markers = [np.nan] * len(df)
                    for i, row in df.iterrows():
                        ts = int(i.timestamp() * 1000)
                        if ts == rounded_open_timestamp or ts == rounded_close_timestamp:
                            open_close_markers[df.index.get_loc(i)] = row['high']
                        if ts in add_timestamps:
                            add_markers[df.index.get_loc(i)] = row['high']

                    # 添加EMA指标
                    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
                    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
                    df['EMA30'] = df['close'].ewm(span=30, adjust=False).mean()
                    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

                    # 计算MACD指标
                    macd, signal_line, histogram = calculate_macd(df['close'])

                    valid_open_close_markers = [marker for marker in open_close_markers if not np.isnan(marker)]
                    valid_add_markers = [marker for marker in add_markers if not np.isnan(marker)]

                    if valid_open_close_markers or valid_add_markers:
                        add_plots = []
                        if valid_open_close_markers:
                            add_plot_open_close = mpf.make_addplot(open_close_markers, type='scatter', markersize=400, marker='o', color='blue')
                            add_plots.append(add_plot_open_close)
                        if valid_add_markers:
                            add_plot_additional = mpf.make_addplot(add_markers, type='scatter', markersize=400, marker='o', color='black')
                            add_plots.append(add_plot_additional)

                        # 添加EMA线
                        add_plot_ema5 = mpf.make_addplot(df['EMA5'], linestyle='--', color='red')
                        add_plot_ema20 = mpf.make_addplot(df['EMA20'], linestyle='--', color='orange')
                        add_plot_ema30 = mpf.make_addplot(df['EMA30'], linestyle='--', color='purple')
                        add_plot_ema50 = mpf.make_addplot(df['EMA50'], linestyle='--', color='green')
                        add_plots.extend([add_plot_ema5, add_plot_ema20, add_plot_ema30, add_plot_ema50])

                        # 添加MACD指标到副图
                        ap0 = [
                            mpf.make_addplot(macd, panel=1, color='fuchsia', ylabel='MACD'),
                            mpf.make_addplot(signal_line, panel=1, color='dodgerblue'),
                        ]

                        # 为MACD直方图设置颜色
                        colors = ['red' if h < 0 else 'green' for h in histogram]
                        ap1 = [
                            mpf.make_addplot(histogram, type='bar', panel=1, color=colors)
                        ]

                        add_plots.extend(ap0)
                        add_plots.extend(ap1)

                        num_candles = len(df)
                        fig_width = max(15, num_candles // 2)
                        fig_height = fig_width / 1.4

                        try:
                            fig, ax = mpf.plot(df, type='candle', volume=False, returnfig=True, style=s, addplot=add_plots, figsize=(fig_width, fig_height))
                            ax[0].set_title(additional_text, fontsize=fontSize, pad=20)

                            ax[0].tick_params(axis='x', labelsize=20)
                            ax[0].tick_params(axis='y', labelsize=20)

                            # 确保文件名有效
                            safe_symbol = "".join(c for c in symbol if c.isalnum() or c in ('_', '-'))
                            safe_open_time = open_time_str.replace(':', '-').replace(' ', '_')
                            safe_close_time = close_time_str.replace(':', '-').replace(' ', '_')
                            output_file = f"{safe_symbol}_{safe_open_time}_{safe_close_time}.png"

                            fig.savefig(output_file)
                            print(f"K-line chart saved to {output_file}")
                        except Exception as e:
                            print(f"Error generating chart for {symbol}: {e}")
                    else:
                        print(f"No valid markers found for {symbol} from {open_time_str} to {close_time_str}")
                else:
                    print(f"No data found for {symbol} from {open_time_str} to {close_time_str}")
            except Exception as e:
                print(f"Error processing row: {row}. Error: {e}")
                continue

# 主程序
if __name__ == "__main__":
    print(f"Processing file: {fileName}")
    try:
        process_csv(fileName)
    except Exception as e:
        print(f"Error: {e}")
