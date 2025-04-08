import csv
import requests
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import pytz
import numpy as np

mins = 5

def add_dashed_lines(df, panel, levels):
    add_plots = []
    for level in levels:
        add_plot = mpf.make_addplot([level] * len(df), panel=panel, color='gray', linestyle='--', secondary_y=False)
        add_plots.append(add_plot)
    return add_plots

def calculate_adx(high, low, close, window=14):
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
    rounded = dt - timedelta(minutes=dt.minute % mins,
                             seconds=dt.second,
                             microseconds=dt.microsecond)
    return rounded

def calculate_macd(data, fast=14, slow=30, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

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
            me = row['maxEarn']
            maxEarnRate = row['maxEarnRate']
            funcName = row["funcName"]

            current_year = datetime.now().year
            open_time = datetime.strptime(f"{current_year}-{open_time_str}", '%Y-%m-%d %H:%M:%S')
            close_time = datetime.strptime(f"{current_year}-{close_time_str}", '%Y-%m-%d %H:%M:%S')
            
            open_timestamp = int(open_time.timestamp() * 1000)
            close_timestamp = int(close_time.timestamp() * 1000)
            
            dateOpen = open_timestamp - 1000 * 60 * 60 * 2
            dateClose = close_timestamp + 1000 * 60 * 60 * 2

            klines_url = f"https://fapi.binance.com/fapi/v1/continuousKlines?interval={mins}m&contractType=PERPETUAL&pair={symbol}&startTime={dateOpen}&endTime={dateClose}"
            response = requests.get(klines_url)
            klines_data = response.json()

            if len(klines_data) > 0:
                df = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df = df.astype(float)
                
                utc_plus_8 = pytz.timezone('Asia/Shanghai')
                df.index = df.index.tz_localize(pytz.utc).tz_convert(utc_plus_8)
                
                additional_text = f"{funcName} {symbol} {open_time_str} ----> {close_time_str}  {side}  {earn}/{me} {earnRate}/{maxEarnRate} {cp}"
                
                mc = mpf.make_marketcolors(up='green', down='red', edge='i', wick='i', volume='in', ohlc='i')
                s = mpf.make_mpf_style(marketcolors=mc)

                rounded_open_time = round_to_nearest_3min(open_time)
                rounded_close_time = round_to_nearest_3min(close_time)
                
                rounded_open_timestamp = int(rounded_open_time.timestamp() * 1000)
                rounded_close_timestamp = int(rounded_close_time.timestamp() * 1000)

                add_timestamps = []
                for add_time_str in [add_time1_str, add_time2_str, add_time3_str]:
                    if add_time_str != 'none':
                        add_time = datetime.strptime(f"{current_year}-{add_time_str}", '%Y-%m-%d %H:%M:%S')
                        rounded_add_time = round_to_nearest_3min(add_time)
                        add_timestamps.append(int(rounded_add_time.timestamp() * 1000))
                
                open_close_markers = [np.nan] * len(df)
                add_markers = [np.nan] * len(df)
                for i, row in df.iterrows():
                    if row.name.timestamp() * 1000 == rounded_open_timestamp or row.name.timestamp() * 1000 == rounded_close_timestamp:
                        open_close_markers[df.index.get_loc(i)] = row['high']
                    if row.name.timestamp() * 1000 in add_timestamps:
                        add_markers[df.index.get_loc(i)] = row['high']
                
                # 注释掉 WMA 的计算
                # df['WMA10'] = df['close'].rolling(window=10).apply(lambda x: np.sum(np.arange(1, 11) * x) / 55, raw=False)
                # df['WMA16'] = df['close'].rolling(window=16).apply(lambda x: np.sum(np.arange(1, 17) * x) / 136, raw=False)
                # df['WMA25'] = df['close'].rolling(window=25).apply(lambda x: np.sum(np.arange(1, 26) * x) / 325, raw=False)
                
                # 注释掉 SMA80 的计算
                # df['SMA80'] = df['close'].rolling(window=80).mean()

                # 添加 EMA20、EMA30 和 EMA50 的计算
                df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
                df['EMA30'] = df['close'].ewm(span=30, adjust=False).mean()
                df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
             
                # 注释掉 ADX 的计算
                # df['adx'] = calculate_adx(df['high'], df['low'], df['close'])

                # 注释掉 RSI 的计算
                # delta = df['close'].diff()
                # gain = delta.where(delta > 0, 0)
                # loss = -delta.where(delta < 0, 0)
                # avg_gain = gain.rolling(window=14).mean()
                # avg_loss = loss.rolling(window=14).mean()
                # rs = avg_gain / avg_loss
                # df['rsi'] = 100 - (100 / (1 + rs))

                # 注释掉 MACD 的计算
                # df['macd'], df['signal'], df['histogram'] = calculate_macd(df['close'], fast=14, slow=30)

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
                    
                    # 注释掉 WMA 的绘图
                    # add_plot_wma = mpf.make_addplot(df[['WMA10', 'WMA16', 'WMA25']])
                    # add_plots.append(add_plot_wma)
                    
                    # 添加 EMA20、EMA30 和 EMA50 的绘图
                    add_plot_ema20 = mpf.make_addplot(df['EMA20'], linestyle='--', color='orange')
                    add_plot_ema30 = mpf.make_addplot(df['EMA30'], linestyle='--', color='purple')
                    add_plot_ema50 = mpf.make_addplot(df['EMA50'], linestyle='--', color='green')
                    add_plots.extend([add_plot_ema20, add_plot_ema30, add_plot_ema50])
                    
                    num_candles = len(df)
                    fig_width = max(15, num_candles // 2)
                    fig_height = fig_width / 1.4
                    
                    fig, ax = mpf.plot(df, type='candle', volume=False, returnfig=True, style=s, addplot=add_plots, figsize=(fig_width, fig_height))
                    ax[0].set_title(additional_text, fontsize=50, pad=20)
                    
                    ax[0].tick_params(axis='x', labelsize=20)
                    ax[0].tick_params(axis='y', labelsize=20)

                    fig.savefig(f"{symbol}_{open_time_str}_{close_time_str}.png")
                    
                    print(f"K-line chart with indicators generated for {symbol} from {open_time_str} to {close_time_str}")
                else:
                    print(f"No valid markers found for {symbol} from {open_time_str} to {close_time_str}")
            else:
                print(f"No data found for {symbol} from {open_time_str} to {close_time_str}")

# Call the function with the CSV file path
process_csv('qcx_4.csv')
#process_csv('earnings.csv')

#process_csv('negative_earnings.csv')