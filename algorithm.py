import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
import joblib

ticker = input("Enter Stock Ticker: ")
stock_data = yf.download(ticker, start="2015-01-01")

print("Stock Data Sample:")
print(stock_data.head())

stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
stock_data = stock_data.dropna()

features = stock_data[['Close', 'SMA_50', 'SMA_200']].values
target = stock_data['Close'].shift(-1).dropna().values
features = features[:-1]

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

print("\nScaled Features Sample:")
print(features_scaled[:5])

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, shuffle=False)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, predictions))
print(f"\nRoot Mean Squared Error (RMSE): {rmse}")

plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Prices', color='blue')
plt.plot(stock_data.index[-len(y_test):], predictions, label='Predicted Prices', color='red')

last_row = stock_data[['Close', 'SMA_50', 'SMA_200']].iloc[-1:]
last_scaled = scaler.transform(last_row.values)

n_days = 10
future_predictions = []
current_history = stock_data.copy()

for _ in range(n_days):
    pred = model.predict(last_scaled)[0]
    future_predictions.append(pred)

    new_date = current_history.index[-1] + pd.Timedelta(days=1)
    new_row = pd.DataFrame({
        'Close': [pred]
    }, index=[new_date])
    current_history = current_history = pd.concat([current_history, new_row])

    current_history['SMA_50'] = current_history['Close'].rolling(window=50, min_periods=1).mean()
    current_history['SMA_200'] = current_history['Close'].rolling(window=200, min_periods=1).mean()

    last_features = current_history[['Close', 'SMA_50', 'SMA_200']].iloc[-1:].values
    last_scaled = scaler.transform(last_features)

future_dates = pd.date_range(start=current_history.index[-n_days], periods=n_days, freq='B')
plt.plot(future_dates, future_predictions, label='Future Predictions', color='green', linestyle='--')

plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())

plt.show()

joblib.dump(model, 'stock_price_predictor.pkl')
print("\nModel saved as 'stock_price_predictor.pkl'")

loaded_model = joblib.load('stock_price_predictor.pkl')
new_predictions = loaded_model.predict(X_test)