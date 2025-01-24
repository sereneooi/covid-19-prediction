from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def predict(num_prediction, model, data_scaled):
    prediction_list = data_scaled[-num_prediction-1:, :].copy()
    for i in range(num_prediction):
        result = model.predict(data_scaled)
        prd = pd.concat([pd.DataFrame(result), pd.DataFrame(data_scaled[:, 1:])], axis = 1)
        reverse_trans = scaler.inverse_transform(prd)
        final = df[prd.shape[0]*-1:]
        final['pred'] = reverse_trans[:, 0] 

        prediction_list = np.append(prediction_list, final['pred'])

    prediction_list = prediction_list[-num_prediction-1:]
    
    return prediction_list
    
def predict_dates(df, num_prediction):
    last_date = df.index.values[-1]
    prediction_dates = pd.date_range(last_date, periods = num_prediction + 1).tolist()

    return prediction_dates

def main():
    colours = ['teal', 'darkturquoise', 'cadetblue', 'powderblue', 'lightblue',
            'deepskyblue', 'steelblue', 'lightskyblue', 'dodgerblue', 'darkslategrey', 
            'royalblue', 'paleturquoise', 'mediumturquoise', 'forestgreen', 'darkseagreen', 
            'green', 'seagreen', 'limegreen']

    # laod the saved model
    model = load_model('models/gru.h5')

    # read data from csv files
    data = pd.read_csv("data/Cases.csv").iloc[41:, :] # get the data after 2020-03-06

    # convert all columns to float64 except for date column
    cols = data.columns.drop('date')
    data[cols] = data[cols].apply(np.float64)
    data.index = pd.to_datetime(data['date']) # set date column to index
    del data["date"] # delete the column 'date' since the date is set to index

    # data cleansing
    data.fillna(value = 0, inplace = True) # fill all NaN with 0

    # create new dataframe only store for the selected columns
    df = data[['cases_new', 'cases_recovered', 'cases_active', 'cases_pvax', 'deaths_new']].copy()
    df.fillna(value = 0, inplace = True)

    # normalized the data 
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # assign value
    num_prediction = 14
    forecast = predict(num_prediction, model, data_scaled)
    forecast_dates = predict_dates(df, num_prediction)

    fct = pd.DataFrame(forecast_dates, columns = ['dates'])
    fct['pred'] = forecast.tolist()
    fct.index = pd.to_datetime(fct['dates'])
    del fct["dates"] 

    # plot predicted cases for future days
    plt.figure(figsize = (25, 15), facecolor = "#627D78")
    plt.title("Predicted New Daily Cases")
    df['cases_new'].plot(label = 'Actual New Daily Cases', color = colours[7]) 
    fct['pred'].plot(label = 'Predicted New Daily Cases', color = colours[2]) 
    plt.ylabel("Cases") # label y-axis
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()