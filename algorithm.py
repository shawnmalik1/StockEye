# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
import joblib

ticker = "AMZN"
stock_data = yf.download(ticker, start="2015-01-01", end="2025-01-01")

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

plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

joblib.dump(model, 'stock_price_predictor.pkl')
print("\nModel saved as 'stock_price_predictor.pkl'")


loaded_model = joblib.load('stock_price_predictor.pkl')
new_predictions = loaded_model.predict(X_test)
