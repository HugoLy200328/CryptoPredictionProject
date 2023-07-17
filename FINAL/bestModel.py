import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


# Get data from Yahoo finance ============================== START
# Set the start and end dates for the data
start_date = dt.datetime(2018, 1, 1)
end_date = dt.date.today()
# Retrieve ETH data
eth_data = yf.download('ETH-USD', start=start_date, end=end_date)
# Get data from Yahoo finance ========== END


# Pre-processing dataset ============================== START
# Resample the DataFrame to fill any missing dates
eth_data = eth_data.resample('D').ffill()

# Create X and y
X = eth_data.drop('Close', axis=1)
y = eth_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# Creating XGBoost train and test data
dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
dtest = xgb.DMatrix(X_test.values, label=y_test.values)
# PRE-PROCESSING DATASET ============================== END


# Training modelll ============================== START
# Defining hyperparameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 9,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'verbosity': 0
}
# Training the XGBoost model
num_round = 1000
xgb_model = xgb.train(params, dtrain, num_round)
# Training modelll ============================== END


# Predicting on testing dataset ============================== START
# Perform prediction
prediction = xgb_model.predict(dtest)
# Convert dataframe to an array for ploting graphs purposes
y_test_transformed = y_test.values
plt.figure(figsize=(20, 10))
plt.plot(prediction[:100], color='blue', label='Prediction')
plt.plot(y_test_transformed[:100], color='red', label='Actual')
plt.title('ETH PRICE PREDICTION')
plt.legend(loc='best', fontsize=16)
plt.show()
# Predicting on testing dataset ============================== END


# Predicting the future ============================== START
# Get the data for the previous 30 days
previous_data = eth_data.iloc[-31:-1]

# Separate the features and the target variable (Close price)
X_previous = previous_data.drop('Close', axis=1)
y_previous = previous_data['Close']

# Create DMatrix for the previous data
dprevious = xgb.DMatrix(X_previous.values)

# Use the trained model to predict the next day's price
predicted_price = xgb_model.predict(dprevious)

# Display the predicted ETH prices for the first 1 to 30 days
# Change the range value (only within 1 to 30) ==> can you a variable for storing users choice

theDayRange = 30
for i in range(theDayRange):
    print(f"Day {i+1} - Predicted ETH price: {predicted_price[i]}")

# Predicting the future ============================== END
