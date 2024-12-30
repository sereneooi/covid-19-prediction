import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# read data from csv files
data = pd.read_csv("Cases.csv").iloc[41:, :] # get the data after 2020-03-06

# convert all columns to float64 except for date column
cols = data.columns.drop('date')
data[cols] = data[cols].apply(np.float64)
data.index = pd.to_datetime(data['date']) # set date column to index
del data["date"] # delete the column 'date' since the date is set to index

# data cleansing
data.fillna(value = 0, inplace = True) # fill all NaN with 0

def showTrends(title, x_variable, y_variable):
    plt.title(title)
    plt.plot(x_variable, y_variable)
    X = plt.gca().xaxis # decalared X variable to x-axis
    X.set_major_locator(mdates.MonthLocator()) # set every month
    X.set_major_formatter(mdates.DateFormatter('%b %Y')) # Specify the format of date to be displayed
    plt.xticks(rotation = 45) # rotate the x-axis label so that can be seen clearly
    plt.ylabel("Cases") # label y-axis
    plt.show()

smooth_data = data.rolling(7).mean().round(5)

# create new dataframe only store for the selected columns
df = data[['cases_new', 'cases_fvax', 'daily']].copy()
df.fillna(value = 0, inplace = True)

# normalized the data 
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

features = data_scaled
target = data_scaled[:, 0] # target variable is = 'cases_new' which is located at the 0 index

generator = TimeseriesGenerator(features, target, length = 7, sampling_rate = 1, batch_size = 1)[0]

# x = the whole shape including columns while y = only hold for rows
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 123, shuffle = False)

win_length = 60
batch_size = 130 # the higher the more accurate but slower
num_features = len(df.columns)
train_generator = TimeseriesGenerator(x_train, y_train, length = win_length, sampling_rate = 1, batch_size = batch_size) # sampling rate is the slider
test_generator = TimeseriesGenerator(x_test, y_test, length = win_length, sampling_rate = 1, batch_size = batch_size)

# build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape = (win_length, num_features), return_sequences = True))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.5))
model.add(tf.keras.layers.LSTM(128, return_sequences = True))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(64, return_sequences = False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

model.summary()

# stop the training if the performance does not improve after 2 iterations
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2, mode = 'min')
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = [tf.metrics.MeanAbsoluteError()])
history = model.fit(train_generator, epochs = 50, validation_data = test_generator, shuffle = False, callbacks = [early_stopping])
model.evaluate(test_generator, verbose = 0)

# prediction
predictions = model.predict(test_generator)
#print(predictions.shape[0])
df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:, 1:][win_length:])], axis = 1)
rev_trans = scaler.inverse_transform(df_pred)
df_final = df[predictions.shape[0]*-1:]

# store the predicted value into the new column
df_final['cases_new_pred'] = rev_trans[:, 0] # 0 means the index, which is cases_new

# plot 
plt.title("Predicted New Daily Cases")
df_final['cases_new_pred'].plot(label = 'Predicted New Daily Cases')

# df_final will only display the actual cases that is being used by the prediction, if want show all then change to df instead of df_final
df['cases_new'].plot(label = 'Actual New Daily Cases') 
X = plt.gca().xaxis # decalared X variable to x-axis
X.set_major_locator(mdates.MonthLocator()) # set every month
X.set_major_formatter(mdates.DateFormatter('%b %Y')) # Specify the format of date to be displayed
plt.xticks(rotation = 45) # rotate the x-axis label so that can be seen clearly
plt.ylabel("Cases") # label y-axis
plt.legend()
plt.show()


models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
os.makedirs(models_dir, exist_ok=True)

# save model
model_path = os.path.join(models_dir, "lstm.h5")

# save model
model.save(model_path)

# def predict(num_prediction, model):
#     prediction_list = data_scaled[-num_prediction-1:, :].copy()
    
#     for i in range(num_prediction):
#         y = np.zeros((data_scaled.shape[0], )) # create a target variable, not important since the 
#         gen = TimeseriesGenerator(data_scaled, y, length = win_length, sampling_rate = 1, batch_size = batch_size)
#         result = model.predict(gen)
#         prd = pd.concat([pd.DataFrame(result), pd.DataFrame(data_scaled[:, 1:][win_length:])], axis = 1)
#         reverse_trans = scaler.inverse_transform(prd)
#         final = df[prd.shape[0]*-1:]
#         final['pred'] = reverse_trans[:, 0] 

#         prediction_list = np.append(prediction_list, final['pred'])

#     prediction_list = prediction_list[-num_prediction-1:]
    
#     return prediction_list
    
# def predict_dates(num_prediction):
#     last_date = df.index.values[-1]
#     prediction_dates = pd.date_range(last_date, periods = num_prediction + 1).tolist()

#     return prediction_dates

# # assign value
# num_prediction = 14
# forecast = predict(num_prediction, model)
# forecast_dates = predict_dates(num_prediction)

# fct = pd.DataFrame(forecast_dates, columns = ['dates'])
# fct['pred'] = forecast.tolist()
# fct.index = pd.to_datetime(fct['dates'])
# del fct["dates"] 

# # plot predicted cases for future days
# plt.title("Predicted New Daily Cases")
# df['cases_new'].plot(label = 'Actual New Daily Cases') 
# fct['pred'].plot(label = 'Predicted New Daily Cases') 
# plt.ylabel("Cases") # label y-axis
# plt.legend()
# plt.show()

'''
# prediction for future days
def predict(num_prediction, model):
    prediction_list = df.iloc[-num_prediction-1:, :].copy()
    
    for i in range(num_prediction):
        prd = model.predict(test_generator)
        prd = pd.concat([pd.DataFrame(prd), pd.DataFrame(x_test[:, 1:][win_length:])], axis = 1)
        reverse_trans = scaler.inverse_transform(prd)
        final = df[prd.shape[0]*-1:]
        final['pred'] = reverse_trans[:, 0] 

        prediction_list = np.append(prediction_list, final['pred'])

    prediction_list = prediction_list[-num_prediction-1:]
    
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df.index.values[-1]
    prediction_dates = pd.date_range(last_date, periods = num_prediction + 1).tolist()

    return prediction_dates

# assign value
num_prediction = 14
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

fct = pd.DataFrame(forecast_dates, columns = ['dates'])
fct['pred'] = forecast.tolist()
fct.index = pd.to_datetime(fct['dates'])
del fct["dates"] 

# plot predicted cases for future days
plt.title("Predicted New Daily Cases")
#df['cases_new'].plot(label = 'Actual New Daily Cases') 
fct['pred'].plot(label = 'Predicted New Daily Cases') 
plt.ylabel("Cases") # label y-axis
plt.legend()
plt.show()
'''