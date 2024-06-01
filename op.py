import csv
import requests
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import pytz
import numpy as np

def round_to_nearest_5min(dt):
    rounded = dt - timedelta(minutes=dt.minute % 5,
                             seconds=dt.second,
                             microseconds=dt.microsecond)
    return rounded

def process_csv(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            symbol = row['symbol']
            open_time_str = row['openTime']
            close_time_str = row['closeTime']
            add_time1_str = row['addTime1']
            add_time2_str = row['addTime2']
            add_time3_str = row['addTime3']
            side = row['side']
            earn = row['earn']
            earnRate = row['earnRate']
            cp = row['closeType']

            current_year = datetime.now().year
            open_time = datetime.strptime(f"{current_year}-{open_time_str}", '%Y-%m-%d %H:%M:%S')
            close_time = datetime.strptime(f"{current_year}-{close_time_str}", '%Y-%m-%d %H:%M:%S')
            
            open_timestamp = int(open_time.timestamp() * 1000)
            close_timestamp = int(close_time.timestamp() * 1000)
            
            dateOpen = open_timestamp - 1000 * 60 * 60 * 2
            dateClose = close_timestamp + 1000 * 60 * 60 * 2

            klines_url = f"https://fapi.binance.com/fapi/v1/continuousKlines?interval=5m&contractType=PERPETUAL&pair={symbol}&startTime={dateOpen}&endTime={dateClose}"
            response = requests.get(klines_url)
            klines_data = response.json()

            if len(klines_data) > 0:
                df = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close']]
                df = df.astype(float)
                
                # 将时间转换为UTC+8时区
                utc_plus_8 = pytz.timezone('Asia/Shanghai')
                df.index = df.index.tz_localize(pytz.utc).tz_convert(utc_plus_8)
                
                additional_text = f"{symbol} {open_time_str} ----> {close_time_str}  {side}  {earn} {earnRate} {cp}"
                
                # 设置蜡烛图颜色
                mc = mpf.make_marketcolors(up='green', down='red', edge='i', wick='i', volume='in', ohlc='i')
                s = mpf.make_mpf_style(marketcolors=mc)

                # 向下取整至5分钟的倍数
                rounded_open_time = round_to_nearest_5min(open_time)
                rounded_close_time = round_to_nearest_5min(close_time)
                
                # 转换为UTC时间戳
                rounded_open_timestamp = int(rounded_open_time.timestamp() * 1000)
                rounded_close_timestamp = int(rounded_close_time.timestamp() * 1000)

                # 添加时间的处理
                add_timestamps = []
                for add_time_str in [add_time1_str, add_time2_str, add_time3_str]:
                    if add_time_str != 'none':
                        add_time = datetime.strptime(f"{current_year}-{add_time_str}", '%Y-%m-%d %H:%M:%S')
                        rounded_add_time = round_to_nearest_5min(add_time)
                        add_timestamps.append(int(rounded_add_time.timestamp() * 1000))
                
                # 在总数据中找到对应的时间戳并标记
                open_close_markers = [np.nan] * len(df)
                add_markers = [np.nan] * len(df)
                for i, row in df.iterrows():
                    if row.name.timestamp() * 1000 == rounded_open_timestamp or row.name.timestamp() * 1000 == rounded_close_timestamp:
                        open_close_markers[df.index.get_loc(i)] = row['high']
                    if row.name.timestamp() * 1000 in add_timestamps:
                        add_markers[df.index.get_loc(i)] = row['high']
                
                # 计算WMA10、WMA16和WMA25指标
                df['WMA10'] = df['close'].rolling(window=10).apply(lambda x: np.sum(np.arange(1, 11) * x) / 55, raw=False)
                df['WMA16'] = df['close'].rolling(window=16).apply(lambda x: np.sum(np.arange(1, 17) * x) / 136, raw=False)
                df['WMA25'] = df['close'].rolling(window=25).apply(lambda x: np.sum(np.arange(1, 26) * x) / 325, raw=False)

                # 过滤有效的标记数据
                valid_open_close_markers = [marker for marker in open_close_markers if not np.isnan(marker)]
                valid_add_markers = [marker for marker in add_markers if not np.isnan(marker)]

                # 检查是否有有效数据
                if valid_open_close_markers or valid_add_markers:
                    add_plots = []
                    if valid_open_close_markers:
                        add_plot_open_close = mpf.make_addplot(open_close_markers, type='scatter', markersize=400, marker='o', color='blue')
                        add_plots.append(add_plot_open_close)
                    if valid_add_markers:
                        add_plot_additional = mpf.make_addplot(add_markers, type='scatter', markersize=400, marker='o', color='black')
                        add_plots.append(add_plot_additional)
                    
                    add_plot_wma = mpf.make_addplot(df[['WMA10', 'WMA16', 'WMA25']])
                    add_plots.append(add_plot_wma)

                      # 根据数据的长度调整图形大小
                    num_candles = len(df)
                    fig_width = max(15, num_candles // 3)  # 每10根蜡烛图增加1英寸，最小宽度为15英寸

                    # 根据数据的长度调整图形大小
                    # fig_width = 20  # 固定宽度为20英寸
                    fig_height = fig_width / 1.4  # 高度为宽度的1.4倍
                    
                    fig, ax = mpf.plot(df, type='candle', volume=False, returnfig=True, style=s, addplot=add_plots, figsize=(fig_width, fig_height))
                    ax[0].set_title(additional_text, fontsize=20, pad=20)
                    
                    # 修改x轴和y轴的字体大小
                    ax[0].tick_params(axis='x', labelsize=20)
                    ax[0].tick_params(axis='y', labelsize=20)

                    fig.savefig(f"{symbol}_{open_time_str}_{close_time_str}.png")
                    
                    print(f"K-line chart generated for {symbol} from {open_time_str} to {close_time_str}")
                else:
                    print(f"No valid markers found for {symbol} from {open_time_str} to {close_time_str}")
            else:
                print(f"No data found for {symbol} from {open_time_str} to {close_time_str}")

# 调用函数并传入CSV文件路径
process_csv('njx_1.csv')

