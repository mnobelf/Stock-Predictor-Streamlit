import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------
# Helper Function: Data Loading and Preprocessing
# ----------------------------
def load_data(ticker):
    """
    Downloads historical stock data from yfinance and computes moving averages.
    :param ticker: Stock ticker symbol (e.g., 'AAPL').
    :return: Processed DataFrame with technical indicators and target variable.
    """
    # Download historical data
    data = yf.download(ticker, start="2018-01-01", end="2023-01-01")
    
    # Create moving average columns
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    # Create the target variable: next day's closing price
    data['Target'] = data['Close'].shift(-1)
    
    # Remove any rows with missing values (from moving averages or target)
    data.dropna(inplace=True)
    return data

# ----------------------------
# Streamlit App Layout
# ----------------------------

# Set the title of the app
st.title("Stock Price Prediction App")

# Input widget: Let the user enter a stock ticker symbol.
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

# Run the prediction only when a valid ticker is provided
if ticker:
    # Load and preprocess the data
    data = load_data(ticker)
    
    # Define features and target variable
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'MA20']
    X = data[features]
    y = data['Target']
    
    # Split the data into training (80%) and testing (20%) sets
    split_index = int(len(data) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    # ----------------------------
    # Model Training with XGBoost
    # ----------------------------
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         learning_rate=0.1,
                         random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Model Performance on Test Data")
    st.write(f"**Test RMSE:** {rmse:.2f}")
    st.write(f"**Test MAE:** {mae:.2f}")
    st.write(f"**R-squared:** {r2:.4f}")
    
    # ----------------------------
    # Directional Accuracy Calculation
    # ----------------------------
    # This checks if the model correctly predicted the direction (up/down) relative to the last closing price.
    direction_actual = np.sign(y_test.values - X_test['Close'].values)
    direction_pred = np.sign(y_pred - X_test['Close'].values)
    direction_accuracy = np.mean(direction_actual == direction_pred)
    st.write(f"**Directional Accuracy:** {direction_accuracy * 100:.2f}%")
    
    # ----------------------------
    # Plot Actual vs. Predicted Prices
    # ----------------------------
    st.subheader("Actual vs. Predicted Prices")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.index, y_test, label='Actual Price', color='blue')
    ax.plot(y_test.index, y_pred, label='Predicted Price', color='red')
    ax.set_title(f"{ticker} Stock Price Prediction using XGBoost")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    
    # ----------------------------
    # Predict Next Day's Closing Price
    # ----------------------------
    last_row = X.tail(1)
    next_day_pred = model.predict(last_row)
    st.subheader("Next Day Prediction")
    st.write(f"Next day predicted closing price: **{next_day_pred[0]:.2f}**")
