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

    start_date = today - datetime.timedelta(days=10*365)
    # Download data from 2018-01-01 until today
    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"))
    
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the ticker symbol and try again.")
    
    # Feature Engineering: Calculate moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    last = data.iloc[-1]
    # Create Target: Next day's closing price
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    return data, last

# Streamlit App Layout
st.title("Daily Stock Price Prediction")

# User input for ticker symbol
ticker = st.text_input("Enter Stock Ticker (e.g., BBCA.JK)", "BBCA.JK")

if ticker:
    try:
        data, last = load_data(ticker)

        # Define features and target variable
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'MA20']
        X = data[features]
        y = data['Target']
        
        # Create alternating training and testing sets:
        # For example, rows 1-4 for training, row 5 for testing, row 6-9 for training, row 10 for testing, etc.
        indices = np.arange(len(X))
        test_mask = (indices % 5 == 4)  # Using 0-index: 0,1,2,3 -> train; 4 -> test; 5,6,7,8 -> train; 9 -> test; etc.
        train_mask = ~test_mask

        X_train = X.iloc[train_mask]
        X_test = X.iloc[test_mask]
        y_train = y.iloc[train_mask]
        y_test = y.iloc[test_mask]
        
        # Train the XGBoost model
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.05,  # Reduced learning rate
            max_depth=5,         # Reduced tree depth
            subsample=0.8,       # Subsample ratio of training data
            colsample_bytree=0.8,# Subsample ratio of columns
            gamma=0.1,           # Minimum loss reduction required
            reg_alpha=0.1,       # L1 regularization term
            reg_lambda=0.1,      # L2 regularization term
            random_state=42
        )
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
        st.subheader("Actual vs. Predicted")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(y_test.index, y_test, label="Actual Next Day Prices", color="blue")
        ax.plot(y_test.index, y_pred, label="Predicted Next Day Prices", color="red")
        ax.set_title(f"{ticker} Stock Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
        
        # Predict next week closing price using the latest available data
        latest_week = last
        next_day_date = last.name + pd.Timedelta(days=1)
        
        # Prepare features for prediction
        prediction_features = last[features].to_frame().T

        next_day_pred = model.predict(prediction_features)
        st.subheader("Next Day Prediction")
        direction_next_pred = np.sign(next_day_pred[0] - prediction_features['Close'].values[0][0])
        st.write(f"{next_day_date.strftime('%Y-%m-%d')} Predicted direction : " + "Up" if direction_next_pred == 1 else "Down")
        st.write(f"{next_day_date.strftime('%Y-%m-%d')} predicted closing price: {next_day_pred[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
