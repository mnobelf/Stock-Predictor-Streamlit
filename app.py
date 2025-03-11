import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import datetime

def load_data(ticker):
    # Use today's date as the end date for downloading data
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # Download data from 2018-01-01 until today
    data = yf.download(ticker, start="2018-01-01", end=today)
    
    # Feature Engineering: Calculate moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    # Create Target: Next day's closing price
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    return data

# Streamlit App Layout
st.title("Real-Time Stock Price Prediction")

# User input for ticker symbol
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

if ticker:
    # Load data until today
    data = load_data(ticker)
    
    # Define features and target variable
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'MA20']
    X = data[features]
    y = data['Target']
    
    # Split data (80% training, 20% testing) preserving time order
    split_index = int(len(data) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    # Train the XGBoost model
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         learning_rate=0.1,
                         random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Model Performance on Test Data")
    st.write(f"**Test RMSE:** {rmse:.2f}")
    st.write(f"**Test MAE:** {mae:.2f}")
    st.write(f"**R-squared:** {r2:.4f}")
    
    # Plot Actual vs. Predicted Prices
    st.subheader("Actual vs. Predicted Prices")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_test.index, y_test, label="Actual Price", color="blue")
    ax.plot(y_test.index, y_pred, label="Predicted Price", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    
    # Predict next day closing price using the latest available data
    last_row = X.tail(1)
    next_day_pred = model.predict(last_row)
    st.subheader("Next Day Prediction")
    st.write(f"Next day predicted closing price: **{next_day_pred[0]:.2f}**")
