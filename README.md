# StockEye 📈👁️
StockEye is a stock price predictor that uses a linear regression machine learning model to predict the short-term success of a given stock. 

StockEye uses historical data from the past 10 years to train its model.

## Description 🔮
StockEye allows the user to input a stock ticker, fetching historical stock data from Yahoo Finance via the ```yfinance``` library, and  calculating rolling averages (50-day and 200-day SMAs) for feature engineering. 
A linear regression model is trained on these features to predict the following day’s closing price. 
The script also provides a short-term forecast (10 days) by iteratively updating and recalculating the SMAs. 

StockEye provides the user with an intuitive and interactive graph of the predicted closing prices for the near future.

## Technologies 🖥️
- Python
- Pandas & NumPy
- yfinance
- scikit-learn
- matplotlib

## Authors ✍️
- Developed by Shawn Malik

## License 📝

This project is licensed under the [MIT License](./LICENSE)
