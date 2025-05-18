# Automatically install missing packages (only for Jupyter or Colab)
try:
    import yfinance as yf
except ImportError:
    import sys
    !{sys.executable} -m pip install yfinance
    import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Install other required packages
try:
    import sklearn
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    import sys
    !{sys.executable} -m pip install scikit-learn
    from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
except ImportError:
    import sys
    !{sys.executable} -m pip install tensorflow
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM

# Step 1: Download historical stock data
ticker = 'AAPL'  # Change to any stock symbol you want
data = yf.download(ticker, start='2015-01-01', end='2024-01-01')

# Use only the 'Close' price
data = data[['Close']]

# Step 2: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3: Prepare training data
look_back = 60  # Number of previous days used for prediction
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape into 3D input for LSTM: [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 4: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Step 6: Predict on training data
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Step 7: Plot actual vs predicted prices
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()