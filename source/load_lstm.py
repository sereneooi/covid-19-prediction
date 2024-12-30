from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import warnings
warnings.filterwarnings("ignore")

def predict(num_prediction, model):
    win_length = 60 
    batch_size = 130
    prediction_list = data_scaled[-num_prediction-1:, :].copy()
    for i in range(num_prediction):
        y = np.zeros((data_scaled.shape[0], )) # create a target variable, not important since the 
        gen = TimeseriesGenerator(data_scaled, y, length = win_length, sampling_rate = 1, batch_size = batch_size)
        result = model.predict(gen)
        prd = pd.concat([pd.DataFrame(result), pd.DataFrame(data_scaled[:, 1:][win_length:])], axis = 1)
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

# laod the saved model
model = load_model('lstm.h5')

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
df = data[['cases_new', 'cases_fvax', 'daily']].copy()
df.fillna(value = 0, inplace = True)

# normalized the data 
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

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
df['cases_new'].plot(label = 'Actual New Daily Cases') 
fct['pred'].plot(label = 'Predicted New Daily Cases') 
plt.ylabel("Cases") # label y-axis
plt.legend()
plt.show()