import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# read data from csv files
data = pd.read_csv("Cases.csv").iloc[41:, :] # get the data after 2020-03-06

# convert all columns to float64 except for date column
cols = data.columns.drop('date')
data[cols] = data[cols].apply(np.float64)
data.index = pd.to_datetime(data['date']) # set date column to index
del data["date"] # delete the column 'date' since the date is set to index

# data cleansing
data.fillna(value = 0, inplace = True) # fill all NaN with 0

# create new dataframe only store for the selected columns
df = data[['cases_new', 'cases_recovered', 'cases_fvax', 'daily']].copy()
df.fillna(value = 0, inplace = True)

# check whether it is stationary or not, p-value less than 0.05 can dy
ad_fuller_result_1 = adfuller(df['cases_new'].diff()[1:])
# print(f'ADF Statistic: {ad_fuller_result_1[0]}')
# print(f'p-value: {ad_fuller_result_1[1]}')

# 4 (lag value) is rondomly pick one 
# print('cases_new causes daily?\n')
# granger_1 = grangercausalitytests(df[['cases_new', 'daily']], 4)
# print('\ndaily causes cases_new?\n')
# granger_2 = grangercausalitytests(df[['daily', 'cases_new']], 4)

# split into train test dataset
train_df = df[:int(len(df) * 0.8)]
test_df = df[int(len(df) * 0.8):]

# didn't include the target variable
model = VAR(train_df.diff()[1:])

# check the lowest value, the lowest the best
# sorted_order = model.select_order(maxlags = 20)
# print(sorted_order.summary()) # in this case, the lowest value fall on the lag 10

# VARMAX is easier to do forecasting
var_model = VARMAX(train_df, order = (10, 0), enforce_stationarity = True)
fitted_model = var_model.fit(disp = False)
print(fitted_model.summary())

# prediction
n_forecast = 14
predict = fitted_model.get_prediction(start = test_df.index[-(n_forecast + 1)] , end = test_df.index[-1])

predictions = predict.predicted_mean
predictions.columns = ['Predicted New Cases', 'Predicted Recovered', 'Predicted Fully Vaccinated Cases', 'Predicted Daily Vaccinated']
print(predictions)

# combine 2 dataframes into 1
test_vs_pred = pd.concat([test_df, predictions], axis = 1)

# plot 
plt.title("Predicted New Daily Cases")
test_vs_pred['Predicted New Cases'].plot(label = 'Predicted New Daily Cases')
test_df['cases_new'].plot(label = 'Actual New Daily Cases')
X = plt.gca().xaxis # decalared X variable to x-axis
X.set_major_locator(mdates.MonthLocator()) # set every month
X.set_major_formatter(mdates.DateFormatter('%b %Y')) # Specify the format of date to be displayed
plt.xticks(rotation = 45) # rotate the x-axis label so that can be seen clearly
plt.ylabel("Cases") # label y-axis
plt.legend()
plt.show()

# predict
def predict(num_prediction, model):
    n_forecast = num_prediction
    predict = model.get_prediction(start = len(df), end = len(df) + n_forecast - 1)

    predictions = predict.predicted_mean
    predictions.columns = ['Predicted New Cases', 'Predicted Recovered', 'Predicted Fully Vaccinated Cases', 'Predicted Daily Vaccinated']

    # combine 2 dataframes into 1
    final_pred = pd.concat([df, predictions], axis = 1)

    # plot 
    plt.title("Predicted New Daily Cases In Penang")
    final_pred['Predicted New Cases'].plot(label = 'Predicted New Daily Cases')
    df['cases_new'].plot(label = 'Actual New Daily Cases') 
    X = plt.gca().xaxis # decalared X variable to x-axis
    X.set_major_locator(mdates.MonthLocator()) # set every month
    X.set_major_formatter(mdates.DateFormatter('%b %Y')) # Specify the format of date to be displayed
    plt.xticks(rotation = 45) # rotate the x-axis label so that can be seen clearly
    plt.ylabel("Cases") # label y-axis
    plt.legend()
    plt.show()

# assign value (day to be predicted)
num_prediction = 14
predict(num_prediction, fitted_model)