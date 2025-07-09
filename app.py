# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from datetime import datetime
import streamlit as st

# Title
st.title('Stock Price Prediction')

# Input box for user to enter stock ticker symbol
user_input = st.text_input('Enter Stock Ticker' , 'TSLA')

# Fetching data from Yahoo Finance for the selected stock
data = yf.Ticker(user_input)
today = datetime.today().strftime('%Y-%m-%d')
stock = data.history(start='2020-01-01', end=today)
df = stock[['Close']]
df = df.reset_index(drop=True) 

# Showing basic stats of the stock data
st.subheader('Data from 2020 - 2025')
st.write(stock.describe()) 

# Plotting raw Closing Prices over time
st.subheader('Closing Price vs Time chart') 
fig = plt.figure(figsize = (12,6))
plt.plot(stock.Close)
st.pyplot(fig)

# Normalizing the close prices between 0 and 1
scaler = MinMaxScaler(feature_range = (0,1)) 
scaled_data = scaler.fit_transform(df)

# Preparing sequences for LSTM (100-day window as input, next day as label)
X = []
y = []
sequence_length = 100
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])
X = np.array(X)
y = np.array(y)

# Using last 15% of data for testing
val_size = int(0.85 * len(X))
X_test, y_test = X[val_size:], y[val_size:]
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Loading pre-trained LSTM model
model = load_model('pred_model.keras')

# Making predictions using the model & Inversing normalization to get actual price values
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test)

# Plotting actual vs predicted prices with real calendar dates
dates = stock.index
test_start_index = sequence_length + val_size 
test_dates = dates[test_start_index:test_start_index + len(y_test)]
st.subheader('Actual Price vs Predicted Price')
fig2 = plt.figure(figsize=(14,6))
plt.plot(test_dates, real_prices, label='Actual Price')
plt.plot(test_dates, predicted_prices, label='Predicted Price')
plt.xlabel('Time (in Months)')
plt.ylabel('Stock Price (in USD)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
st.pyplot(fig2)

# Calculating and displaying MAE and RMSE for model performance
mae = mean_absolute_error(real_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
st.write(f"MAE: {mae:.4f}")
st.write(f"RMSE: {rmse:.4f}")