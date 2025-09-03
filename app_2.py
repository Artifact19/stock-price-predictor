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
st.title('Stock Price Prediction with LSTM')

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
st.subheader('Closing Price vs Time Chart') 
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


# Future Predictions
future_days = 30

# Start with the last 100 days from scaled data (the model input window)
last_sequence = scaled_data[-sequence_length:]
future_input = last_sequence.reshape(1, sequence_length, 1)

future_predictions = []
for _ in range(future_days):
    next_pred = model.predict(future_input)[0]
    future_predictions.append(next_pred)

    # updating the input sequence with the new prediction
    future_input = np.append(future_input[:, 1:, :], [[next_pred]], axis=1)

# Converting predictions back to original scale
future_predictions = scaler.inverse_transform(future_predictions)

# Creating date index for future predictions
last_date = stock.index[-1]
future_dates = pd.date_range(last_date, periods=future_days+1, freq="B")[1:]

# Plot future predictions 
st.subheader('Future 30-Day Forecast')

zoom_history = stock["Close"].last("180D")   # last 180 days 

fig3 = plt.figure(figsize=(14,6))
plt.plot(zoom_history.index, zoom_history.values, label="Recent Historical Price")
plt.plot(future_dates, future_predictions, label="Future Predictions", color="red")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title(f"Next {future_days} Days Forecast for {user_input}")
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot(fig3)


# Future Predictions Stats
current_price = stock["Close"].iloc[-1]
predicted_high = float(future_predictions.max())
predicted_low = float(future_predictions.min())
predicted_final = float(future_predictions[-1])
predicted_change = predicted_final - current_price
predicted_pct_change = (predicted_change / current_price) * 100

st.subheader("Future Forecast Summary")
st.write(f"Current Price: **${current_price:.2f}**")
st.write(f"Predicted Highest Price: **${predicted_high:.2f}**")
st.write(f"Predicted Lowest Price: **${predicted_low:.2f}**")
st.write(f"Predicted Price after {future_days} days: **${predicted_final:.2f}**")
st.write(f"Predicted Change: **${predicted_change:.2f} ({predicted_pct_change:.2f}%)**")