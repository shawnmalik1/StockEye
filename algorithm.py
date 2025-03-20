import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings

def validate_data(df, min_rows=300):
    if df.empty:
        raise ValueError("Dataframe is empty. Check the ticker symbol or the data source.")
    if len(df) < min_rows:
        warnings.warn(f"Data has only {len(df)} rows, which may be insufficient for robust modeling.")
    if df[['Close', 'Volume']].isnull().any().any():
        warnings.warn("Missing values found in 'Close' or 'Volume' columns. Consider cleaning the data.")
    return True

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

ticker = input("Enter Stock Ticker: ")
data = yf.download(ticker, start="2015-01-01")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.dropna(inplace=True)
validate_data(data)
data['RSI'] = compute_rsi(data['Close'])
data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])
data['SMA_20'], data['BB_upper'], data['BB_lower'] = compute_bollinger_bands(data['Close'])

data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

data.dropna(inplace=True)

df_prophet = data.reset_index()[['Date', 'Close', 'RSI', 'MACD', 'MACD_Signal', 'BB_upper', 'BB_lower', 'SMA_50', 'SMA_200', 'Volume']]
df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')

model = Prophet()
model.add_regressor('RSI')
model.add_regressor('MACD')
model.add_regressor('MACD_Signal')
model.add_regressor('BB_upper')
model.add_regressor('BB_lower')
model.add_regressor('SMA_50')
model.add_regressor('SMA_200')
model.add_regressor('Volume')
model.fit(df_prophet)

future = model.make_future_dataframe(periods=30, freq='B')

last_row = df_prophet.iloc[-1]

future['RSI'] = last_row['RSI']
future['MACD'] = last_row['MACD']
future['MACD_Signal'] = last_row['MACD_Signal']
future['BB_upper'] = last_row['BB_upper']
future['BB_lower'] = last_row['BB_lower']
future['SMA_50'] = last_row['SMA_50']
future['SMA_200'] = last_row['SMA_200']
future['Volume'] = last_row['Volume']

forecast = model.predict(future)

fig1 = model.plot(forecast)
plt.title(f'{ticker} Stock Price Forecast with Advanced Features')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
fig2 = model.plot_components(forecast)
plt.show()
