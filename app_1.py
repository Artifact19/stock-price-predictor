import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import streamlit as st

# --- Streamlit Title ---
st.title("Stock Price Prediction with ARIMA")

# --- User Input ---
ticker = st.text_input("Enter Stock Ticker", "NVDA")
start_date = "2020-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# --- Load Data ---
data = yf.Ticker(ticker).history(start=start_date, end=end_date)
close = data["Close"].dropna()
close = close.asfreq("B").ffill()

# --- Plot Historical Data ---
st.subheader("Closing Price vs Time Chart")
fig1 = plt.figure(figsize=(12,6))
plt.plot(close, label="Close Price")
plt.title(f"{ticker} Stock Price")
plt.xlabel("Date"); plt.ylabel("Price (USD)"); plt.legend()
st.pyplot(fig1)

# --- Train-Test Split for Evaluation ---
train_size = int(len(close) * 0.8)
train, test = close[:train_size], close[train_size:]

# --- Fit ARIMA Model (on train set for evaluation) ---
p, d, q = 1, 1, 1
model = ARIMA(train, order=(p,d,q))
model_fit = model.fit()

# --- Forecast on Test Data ---
forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

# --- Plot Actual vs Forecast ---
st.subheader("Actual Price vs Predicted Price")
fig2 = plt.figure(figsize=(12,6))
plt.plot(train, label="Train")
plt.plot(test, label="Test", color="orange")
plt.plot(forecast, label="Forecast", color="green")
plt.title(f"ARIMA({p},{d},{q}) Forecast vs Actual")
plt.xlabel("Date"); plt.ylabel("Price"); plt.legend()
st.pyplot(fig2)

# --- Evaluation ---
results = pd.DataFrame({"Actual": test, "Forecast": forecast}).dropna()
mse = mean_squared_error(results["Actual"], results["Forecast"])
rmse = math.sqrt(mse)
mae = mean_absolute_error(results["Actual"], results["Forecast"])

st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")


# --- Refit on Full Data for Future Predictions ---
full_model = ARIMA(close, order=(p,d,q))
full_fit = full_model.fit()

# --- Future Forecast (30 Days) ---
future_days = 30
future_forecast = full_fit.forecast(steps=future_days)

# Create future dates
last_date = close.index[-1]
future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days, freq="B")
future_forecast = pd.Series(future_forecast, index=future_dates)

# --- Plot Future Forecast ---
st.subheader("Future 30-Day Forecast")
fig3 = plt.figure(figsize=(12,6))
plt.plot(close.last("180D"), label="Recent History")
plt.plot(future_forecast, label="Future Forecast", color="red")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.title(f"{ticker} - Next {future_days} Business Days")
plt.legend(); plt.grid(True)
st.pyplot(fig3)

# --- Future Forecast Stats ---
current_price = close.iloc[-1]
predicted_high = future_forecast.max()
predicted_low = future_forecast.min()
predicted_final = float(future_forecast[-1])

absolute_change = predicted_final - current_price
percent_change = (absolute_change / current_price) * 100

st.subheader("Future Forecast Summary")
st.write(f"Current Price: **${current_price:.2f}**")
st.write(f"Predicted Highest Price: **${predicted_high:.2f}**")
st.write(f"Predicted Lowest Price: **${predicted_low:.2f}**")
st.write(f"Predicted Price after {future_days} days: **${predicted_final:.2f}**")
st.write(f"Predicted Change: **${absolute_change:.2f} ({percent_change:.2f}%)**")
