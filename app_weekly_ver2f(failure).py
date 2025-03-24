import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import datetime

def load_data(ticker):
    # Use today's date as the end date for downloading data
    today = datetime.datetime.now()

    start_date = today - datetime.timedelta(days=10*365)
    # Download data from 5 years ago until today
    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"), group_by=f"'{ticker}'")
    
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the ticker symbol and try again.")
    
    data = data[ticker]

    # Resample to weekly data with proper aggregation:
    weekly_data = data.resample('W-FRI').agg({
        'Open': 'first',    # Monday's open (or first available)
        'High': 'max',      # Highest price in the week
        'Low': 'min',       # Lowest price in the week
        'Close': 'last',    # Friday's close (or last available)
        'Volume': 'sum'     # Total volume for the week
    })
    weekly_data = weekly_data.ffill()  # Forward fill missing values
    
    # Feature Engineering: Calculate weekly moving averages
    weekly_data['MA5'] = weekly_data['Close'].rolling(window=5).mean()
    weekly_data['MA10'] = weekly_data['Close'].rolling(window=10).mean()
    weekly_data['MA20'] = weekly_data['Close'].rolling(window=20).mean()

    # Create Target: Next week's closing price
    weekly_data['Target'] = weekly_data['Close'].shift(-1)
    weekly_data.dropna(inplace=True)
    last = weekly_data.iloc[-1]
    return weekly_data, last

# Streamlit App Layout
st.title("Weekly Stock Price Prediction")

# User input for ticker symbol
ticker = st.text_input("Enter Stock Ticker (e.g., BMRI.JK)", "BMRI.JK")

if ticker:
    try:
        # Load weekly data
        data, last = load_data(ticker)
        
        # Feature Engineering
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
        
        # # Train the XGBoost model
        # model = XGBRegressor(objective='reg:squarederror',
        #                      n_estimators=100,
        #                      learning_rate=0.1,
        #                      random_state=42)

        # Modified model with regularization parameters
        model = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')

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

        # Plot current week's actual price
        # current_prices = data['Close'].iloc[split_index:]
        # ax.plot(current_prices.index, current_prices, label="Current Week's Actual Price", color="green")

        ax.plot(y_test.index, y_test, label="Actual Next Week Price", color="blue")
        ax.plot(y_test.index, y_pred, label="Predicted Next Week Price", color="red")
        ax.set_title(f"Weekly {ticker} Stock Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
        
        # Predict next week closing price using the latest available data
        latest_week = last
        next_week_date = latest_week.name + pd.Timedelta(weeks=1)
        
        # Prepare features for prediction
        prediction_features = latest_week[features].to_frame().T
        
        next_week_pred = model.predict(prediction_features)
        st.subheader("Next Week Prediction")
        direction_next_pred = np.sign(next_week_pred[0] - prediction_features['Close'].values[0])
        st.write(f"Week ending {next_week_date.strftime('%Y-%m-%d')} Predicted direction : " + 
                "Up" if direction_next_pred == 1 else "Down")
        st.write(f"Week ending {next_week_date.strftime('%Y-%m-%d')} predicted closing price: {next_week_pred[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")