# Ex.No: 08     MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 25-10-25


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
# EXPERIMENT 8 - TIME SERIES ANALYSIS
# Edited for the dataset: car_price_prediction (1).csv
# Structure remains identical to the original PDF code.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# ------------------------------------------------
# 1. Read the Dataset
# ------------------------------------------------

data = pd.read_csv("/content/car_price_prediction (1).csv")
print("Dataset loaded successfully!\n")

print("First 10 rows:")
print(data.head(10))

# ------------------------------------------------
# 2. Select one numeric column to simulate time series
# ------------------------------------------------
# We'll use the 'Price' column as our target time series.
# Index will be the record order to mimic time steps.

ts = data['Price']
ts.index = pd.RangeIndex(start=1, stop=len(ts)+1, step=1)
ts = ts.astype(float)

print("\nTime series data (Price) shape:", ts.shape)
print(ts.head())

# ------------------------------------------------
# 3. Plot Original Data
# ------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(ts, label='Original Price Data', color='blue')
plt.title("Original Price Data (treated as time series)")
plt.xlabel("Record Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------
# 4. Moving Averages (window = 5 and 10)
# ------------------------------------------------

moving_avg_5 = ts.rolling(window=5).mean()
moving_avg_10 = ts.rolling(window=10).mean()

print("\nFirst 10 values of 5-point moving average:")
print(moving_avg_5.head(10))

print("\nFirst 20 values of 10-point moving average:")
print(moving_avg_10.head(20))

plt.figure(figsize=(12,5))
plt.plot(ts, label='Original Price')
plt.plot(moving_avg_5, label='Moving Average (5)')
plt.plot(moving_avg_10, label='Moving Average (10)')
plt.title("Moving Average of Car Prices")
plt.xlabel("Record Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------
# 5. Normalize (MinMaxScaler) to match PDF code logic
# ------------------------------------------------

scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
ts_scaled = pd.Series(ts_scaled.flatten(), index=ts.index)

# ------------------------------------------------
# 6. Split into Train and Test (1/4 test as per PDF)
# ------------------------------------------------

split_index = int(len(ts_scaled) * 0.75)
train, test = ts_scaled[:split_index], ts_scaled[split_index:]

print("\nTraining data length:", len(train))
print("Testing data length:", len(test))

# ------------------------------------------------
# 7. Holt-Winters Exponential Smoothing
# ------------------------------------------------

model = ExponentialSmoothing(train, trend='add', seasonal=None)
model_fit = model.fit()
pred_test = model_fit.forecast(len(test))

# Plot results
plt.figure(figsize=(12,5))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(pred_test, label='Predicted')
plt.title("Holt-Winters Exponential Smoothing on Car Price Data")
plt.xlabel("Record Index")
plt.ylabel("Scaled Price")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------
# 8. Evaluate Model
# ------------------------------------------------

rmse = np.sqrt(mean_squared_error(test, pred_test))
print("\nRMSE on Test Data:", rmse)

# ------------------------------------------------
# 9. Refit on full data and forecast next 1/4
# ------------------------------------------------

model_full = ExponentialSmoothing(ts_scaled, trend='add', seasonal=None)
fit_full = model_full.fit()
forecast_future = fit_full.forecast(int(len(ts_scaled)/4))

plt.figure(figsize=(12,5))
plt.plot(ts_scaled, label='Full Data')
plt.plot(forecast_future, label='Forecast')
plt.title("Forecasted Car Prices (Next 1/4 Data Points)")
plt.xlabel("Record Index")
plt.ylabel("Scaled Price")
plt.legend()
plt.grid(True)
plt.show()

print("\nForecasted Scaled Prices:")
print(forecast_future.head(10))

```
### OUTPUT:

Moving Average
<img width="1373" height="582" alt="image" src="https://github.com/user-attachments/assets/288c3b34-b95d-41eb-9baf-9edd4b15d565" />

Plot Transform Dataset
<img width="1448" height="593" alt="image" src="https://github.com/user-attachments/assets/c100025f-59ca-4742-8c54-82c67fd6eee9" />


Exponential Smoothing
<img width="1424" height="575" alt="image" src="https://github.com/user-attachments/assets/021a7b5e-8443-4509-9401-f106406044f6" />





### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
