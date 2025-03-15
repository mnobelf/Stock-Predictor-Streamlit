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
    today = datetime.datetime.now()

    start_date = today - datetime.timedelta(days=5*365)
    # Download data from 5 years ago until today
    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"))
    
    # Resample to weekly data (using Friday's closing price)
    weekly_data = data.resample('W-FRI').last()
    weekly_data = weekly_data.ffill()  # Forward fill missing values
    
    # Feature Engineering: Calculate weekly moving averages
    weekly_data['MA5'] = weekly_data['Close'].rolling(window=5).mean()
    weekly_data['MA10'] = weekly_data['Close'].rolling(window=10).mean()
    weekly_data['MA20'] = weekly_data['Close'].rolling(window=20).mean()
    
    # Create Target: Next week's closing price
    weekly_data['Target'] = weekly_data['Close'].shift(-1)
    weekly_data.dropna(inplace=True)
    
    return weekly_data

# Streamlit App Layout
st.title("Weekly Stock Price Prediction")

# User input for ticker symbol
ticker = st.text_input("Enter Stock Ticker (e.g., BBCA.JK)", "BBCA.JK")

if ticker:
    # Load weekly data
    data = load_data(ticker)
    
    # Feature Engineering
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'MA20']
    X = data[features]
    y = data['Target']
    
    # Split data (80% training, 20% testing) preserving time order
    split_index = int(len(X) * 0.8)
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
    
    direction_actual = np.sign(y_test.values - X_test['Close'].values)
    direction_pred = np.sign(y_pred - X_test['Close'].values)
    direction_accuracy = np.mean(direction_actual == direction_pred)
    st.write(f"**Directional Accuracy:** {direction_accuracy * 100:.2f}%")

    # Plot Actual vs. Predicted Prices
    st.subheader("Actual vs. Predicted Prices")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_test.index, y_test, label="Actual Price", color="blue")
    ax.plot(y_test.index, y_pred, label="Predicted Price", color="red")
    ax.set_title(f"Weekly {ticker} Stock Price Prediction using XGBoost")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    
    # Predict next week closing price using the latest available data
    latest_week = data.iloc[-1]
    next_week_date = latest_week.name + pd.Timedelta(weeks=1)
    
    # Prepare features for prediction
    prediction_features = latest_week[features].to_frame().T
    
    next_week_pred = model.predict(prediction_features)
    st.subheader("Next Week Prediction")
    direction_next_pred = np.sign(next_week_pred[0] - latest_week['Close'])
    st.write(f"Week ending {next_week_date.strftime('%Y-%m-%d')} Predicted direction : " + 
             "Up" if direction_next_pred == 1 else "Down")
    st.write(f"Week ending {next_week_date.strftime('%Y-%m-%d')} predicted closing price: {next_week_pred[0]:.2f}")