# stock_price_prediction
This project is a deep learning based stock price prediction model built using LSTM networks while also using a statistical baseline model ARIMA, and made a prototype for both through Streamlit. It allows users to input a stock ticker (like AAPL or TSLA) and visualizes the actual vs predicted closing prices based on historical stock data from Yahoo Finance.

Features:

-> Fetches real-time historical data from 2020–present using yfinance

-> Uses a pre-trained LSTM model to predict future stock prices

Visualizations:

-> Raw closing prices

-> Actual vs predicted prices

Displays error metrics:

-> MAE (Mean Absolute Error)

-> RMSE (Root Mean Squared Error)

Model Architecture:

-> LSTM trained on past 100 days of closing prices to predict the next day's price

-> Data normalized using MinMaxScaler

-> Evaluation done on the last 15% of the dataset

Note: The LSTM model (pred_model.keras) must be pre-trained and placed in the working directory.
